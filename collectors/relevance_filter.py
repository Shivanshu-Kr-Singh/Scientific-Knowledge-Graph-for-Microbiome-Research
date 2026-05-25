"""
3-stage relevance filter pipeline.
PIPELINE:
Stage 1 — Metadata filter (PubMed MeSH only, highest precision)
Stage 2 — Weighted rule scorer(all sources, reads from organisms.yaml)
Stage 3 — ML classifier(sentence-transformers+LogisticRegression)
Metagenomics gate — project-specific requirement

FLOW PER PAPER:
PubMed papers →Stage 1 first (MeSH is reliable), then Stage 2 if borderline
Other sources → Stage 2 directly, then Stage 3 if borderline
All papers → metagenomics gate as final check

DESIGN PRINCIPLES:
- All term lists live in config/organisms.yaml — not hardcoded here
- ML model is optional — system works rule-only until enough data to train
- Every decision is logged with reason — fully auditable
- Review queue for borderline papers — researcher makes final call
"""

import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

import yaml
from loguru import logger
from tqdm import tqdm

from models import PaperRecord
from collectors.audit_logger import AuditLogger
from collections import Counter
import json
from pathlib import Path

#Lazy import—LLMVerifier only instantiated if needed
_llm_verifier = None

def _get_llm_verifier():
    global _llm_verifier
    if _llm_verifier is None:
        from collectors.llm_verifier import LLMVerifier
        _llm_verifier = LLMVerifier()
    return _llm_verifier


CONFIG_PATH = Path(__file__).parent.parent / "config" / "organisms.yaml"
MODEL_PATH  = Path(__file__).parent.parent / "config" / "relevance_model.pkl"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class FilterVerdict:
    """Full decision record for one paper — stored for audit."""
    keep:    bool
    score:   float
    stage:   str          # "stage1_mesh" | "stage2_rules" | "stage3_ml" | "gate"
    reason:  str
    review:  bool = False


class RelevanceFilter:
    """
    3-stage pipeline filter. Instantiate once, call filter() on each run.
    """

    def __init__(self):
        self.cfg = _load_config()
        self.pos_terms  = self.cfg.get("positive_terms", {})
        self.neg_terms  = self.cfg.get("negative_terms", {})
        self.thresholds = self.cfg.get("thresholds", {"keep": 0.70, "review": 0.40})
        self.gate_cfg   = self.cfg.get("metagenomics_gate", {"enabled": True, "terms": []})
        self.mesh_keep  = [m.lower() for m in self.cfg.get("mesh_keep", [])]
        self.mesh_human = [m.lower() for m in self.cfg.get("mesh_human_signal", [])]
        self.mesh_animal= [m.lower() for m in self.cfg.get("mesh_animal_only", [])]

        # ML model — loaded only if trained model file exists
        self._ml_model     = None
        self._ml_vectorizer = None
        if MODEL_PATH.exists():
            self._load_ml_model()
            logger.info("[filter] ML classifier loaded")
        else:
            logger.info("[filter] No ML model found — using rules only (Stage 3 skipped)")

    # ── Public interface ───────────────────────────────────────────────────────

    def filter(
        self,
        papers: List[PaperRecord],
    ) -> Tuple[List[PaperRecord], List[PaperRecord], List[Tuple]]:
        """
        Runs the full 3-stage pipeline on a list of papers.
        Returns: (kept, removed, review_queue)
        """
        kept, removed, review = [], [], []
        stage_counts = {"stage1": 0, "stage2": 0, "stage3": 0, "gate": 0}

        for paper in tqdm(papers, desc="Relevance filtering"):
            verdict = self._evaluate(paper)
            AuditLogger.log(paper, verdict)
            stage_counts[verdict.stage.split("_")[0]] = \
                stage_counts.get(verdict.stage.split("_")[0], 0) + 1

            if verdict.review:
                review.append((paper, verdict))
            if verdict.keep:
                kept.append(paper)
            else:
                removed.append(paper)

        self._log_summary(papers, kept, removed, review)
        self._save_removed(removed)
        return kept, removed, review

    # ── Evaluation pipeline ────────────────────────────────────────────────────

    def _evaluate(self, paper: PaperRecord) -> FilterVerdict:
        """Routes each paper through the appropriate stages."""
        source = (paper.source or "").lower()

        # PubMed papers → Stage 1 first (MeSH is human-curated and reliable)
        if "pubmed" in source:
            v = self._stage1_mesh(paper)
            if v.score >= self.thresholds["keep"] or v.score < self.thresholds["review"]:
                return v
            # Borderline PubMed → fall through to Stage 2

        # All sources → Stage 2 (weighted rule scorer)
        v = self._stage2_rules(paper)

        # Confident reject
        if v.score < self.thresholds["review"]:
            return v

        # Run gate
        v = self._metagenomics_gate(paper,v)

        # Borderline → send to LLM
        if v.review:
            return self._stage4_llm(paper,v)

        # High confidence keep
        if v.score >= self.thresholds["keep"]:
            return v

        # Stage3 ML
        if self._ml_model is not None:
            v = self._stage3_ml(paper,v)
            if v.score >= self.thresholds[
                "keep"]:
                return v

        # Stage4
        return self._stage4_llm(paper,v)

    # ── Stage 1: MeSH metadata filter ─────────────────────────────────────────

    def _stage1_mesh(self, paper: PaperRecord) -> FilterVerdict:
        """
        Uses PubMed's human-curated MeSH terms.

        LOGIC:
          Has microbiome MeSH AND human MeSH → confident keep (0.90)
          Has microbiome MeSH, no animal MeSH → likely human (0.70)
          Has animal MeSH, no human MeSH → reject (0.05)
          Has microbiome MeSH AND animal MeSH → borderline (0.45)
        """
        mesh_lower = [m.lower() for m in (paper.mesh_terms or [])]
        if not mesh_lower:
            return FilterVerdict(keep=False, score=0.0, stage="stage1_mesh",
                                 reason="no_mesh_terms")

        has_microbiome = any(m in mesh_lower for m in self.mesh_keep)
        has_human      = any(m in mesh_lower for m in self.mesh_human)
        has_animal     = any(m in mesh_lower for m in self.mesh_animal)

        if not has_microbiome:
            return FilterVerdict(keep=False, score=0.10, stage="stage1_mesh",
                                 reason="no_microbiome_mesh")

        if has_human and not has_animal:
            return FilterVerdict(keep=True, score=0.90, stage="stage1_mesh",
                                 reason="microbiome+human_mesh")

        if has_human and has_animal:
            return FilterVerdict(keep=True, score=0.70, stage="stage1_mesh",
                                 reason="microbiome+human+animal_mesh",
                                 review=False)

        if has_animal and not has_human:
            return FilterVerdict(keep=False, score=0.05, stage="stage1_mesh",
                                 reason="animal_only_mesh")

        # Has microbiome MeSH but no human/animal signal → borderline
        return FilterVerdict(keep=False, score=0.45, stage="stage1_mesh",
                             reason="microbiome_mesh_no_human_signal",
                             review=True)

    # ── Stage 2: Weighted rule scorer ─────────────────────────────────────────

    def _stage2_rules(self, paper: PaperRecord) -> FilterVerdict:
        """
        Additive weighted scoring from organisms.yaml terms.
        Checks title + abstract combined.
        """
        text = (
            (paper.title or "") + " " +
            (paper.abstract or "") + " " +
            " ".join(paper.mesh_terms or []) + " " +
            " ".join(paper.keywords or [])
        ).lower()

        score = 0.0
        matched_pos = []
        matched_neg = []

        # Positive terms (multi-word terms checked first for correct matching)
        for term, weight in sorted(self.pos_terms.items(),
                                   key=lambda x: len(x[0]), reverse=True):
            if term.lower() in text:
                score += weight
                matched_pos.append(term)

        # Negative terms
        for term, weight in sorted(self.neg_terms.items(),
                                   key=lambda x: len(x[0]), reverse=True):
            if term.lower() in text:
                score += weight   # weight is already negative
                matched_neg.append(term)

        # Clamp
        score = max(-1.0, min(1.5, score))
        # Normalize to [0, 1] for consistent thresholding
        score_norm = max(0.0, min(1.0, score / 1.5))

        keep   = score_norm >= self.thresholds["keep"]
        review = self.thresholds["review"] <= score_norm < self.thresholds["keep"]

        reason = (
            f"pos:[{','.join(matched_pos[:3])}] "
            f"neg:[{','.join(matched_neg[:3])}]"
        )

        return FilterVerdict(keep=keep, score=round(score_norm, 3),
                             stage="stage2_rules", reason=reason, review=review)

    # ── Stage 3: ML classifier ────────────────────────────────────────────────

    def _stage3_ml(self, paper: PaperRecord, prev: FilterVerdict) -> FilterVerdict:
        """
        Uses trained sentence-transformers + LogisticRegression classifier.
        Falls back to prev verdict if prediction fails.
        """
        try:
            text = f"{paper.title or ''} {paper.abstract or ''}"
            embedding = self._ml_vectorizer.encode([text])
            prob = self._ml_model.predict_proba(embedding)[0][1]  # prob of class 1 (relevant)

            # Blend ML confidence with rule score for robustness
            blended = 0.4 * prev.score + 0.6 * float(prob)

            keep   = blended >= self.thresholds["keep"]
            review = self.thresholds["review"] <= blended < self.thresholds["keep"]

            return FilterVerdict(
                keep=keep, score=round(blended, 3), stage="stage3_ml",
                reason=f"ml_prob={prob:.3f} rule_score={prev.score}",
                review=review
            )
        except Exception as e:
            logger.warning(f"[filter] ML inference failed: {e} — using rule score")
            return prev

    # ── Stage 4: LLM verifier ────────────────────────────────────────────────

    def _stage4_llm(self, paper: PaperRecord, prev: FilterVerdict) -> FilterVerdict:
        """
        Calls LLM API for borderline papers that Stage 2+3 couldn't resolve.
        COST CONTROL:
          - Only called when score is in [review_threshold, keep_threshold]
          - All responses cached by content hash
          - Typically 5-10% of total papers
        PROVIDER: Set LLM_PROVIDER=gemini in .env"""
        verifier = _get_llm_verifier()

        if not verifier.is_available:
            # No API key configured — flag for human review
            prev.review = True
            prev.stage  = "stage4_llm_unavailable"
            return prev

        try:
            verdict = verifier.verify(paper.title, paper.abstract)
            AuditLogger.log_llm(paper,verdict)
            cached_str = " (cached)" if verdict.cached else ""

            keep   = verdict.confidence >= 0.70 and verdict.keep
            review = 0.50 <= verdict.confidence < 0.70

            logger.debug(
                f"[stage4_llm{cached_str}] '{paper.title[:50]}' → "
                f"keep={verdict.keep} conf={verdict.confidence:.2f} "
                f"reason={verdict.reason}"
            )

            return FilterVerdict(
                keep=keep, score=round(verdict.confidence, 3),
                stage=f"stage4_llm{cached_str}",
                reason=f"llm:{verdict.reason}",
                review=review,
            )

        except Exception as e:
            logger.warning(
                f"[stage4_llm] Failed: {e}")

            prev.keep = False
            prev.review = True
            prev.stage = "stage4_llm_failed"
            prev.reason = (
                f"llm_failed:{str(e)[:80]}")
            return prev

    # ── Metagenomics gate ──────────────────────────────────────────────────────
    def _metagenomics_gate(self, paper: PaperRecord, prev: FilterVerdict) -> FilterVerdict:
        """
        Project-specific gate: paper must mention at least one sequencing/
        data term to pass. This enforces the project objective:
        'metagenomics + literature mining + data availability'.
        Disabled if metagenomics_gate.enabled = false in organisms.yaml.
        """
        if not self.gate_cfg.get("enabled", True):
            return prev

        text = (
            (paper.title or "") + " " + (paper.abstract or "")
        ).lower()

        gate_terms = self.gate_cfg.get("terms", [])
        passed = any(t in text for t in gate_terms)

        if passed:
            return prev   # Gate passed — keep original verdict

        # Gate failed — downgrade to review
        return FilterVerdict(
            keep=False, score=prev.score * 0.5,
            stage="gate_metagenomics",
            reason=f"no_sequencing_terms (prev_score={prev.score})",
            review=True   # Flag for review — don't silently reject
        )

    # ── ML training ───
    def train_ml_model(self, papers: List[PaperRecord]):
        """
        Bootstraps an ML classifier from the collected papers.

        TRAINING STRATEGY — self-supervised bootstrap:
          1. Run Stage 2 rules on all collected papers
          2. High-confidence keeps (score > 0.85) → label as 1 (relevant)
          3. High-confidence rejects (score < 0.15) → label as 0 (off-topic)
          4. Discard borderline papers from training
          5. Train sentence-transformers + LogisticRegression
          6. Save model to config/relevance_model.pkl

        WHY SELF-SUPERVISED?
          We don't have labeled training data yet. The rules give us
          high-confidence labels on the extremes. The ML model then
          generalizes to handle the borderline cases the rules struggle with.
          This is called label propagation/pseudo-labeling.

        WHEN TO RUN:
          After your first large collection (MAX_PER_SOURCE=500+).
          You'll have ~1000+ papers → enough for a solid classifier.
          Re-train monthly as more papers accumulate.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            import numpy as np
        except ImportError:
            logger.error("Run: pip install sentence-transformers scikit-learn")
            return

        logger.info("[filter] Bootstrapping ML training data from rule scores...")

        texts, labels = [], []
        for paper in papers:
            v = self._stage2_rules(paper)
            if v.score >= 0.85:
                texts.append(f"{paper.title or ''} {paper.abstract or ''}")
                labels.append(1)
            elif v.score <= 0.15:
                texts.append(f"{paper.title or ''} {paper.abstract or ''}")
                labels.append(0)

        if len(texts) < 50:
            logger.warning(f"[filter] Only {len(texts)} training samples — need 50+. "
                           "Collect more papers before training.")
            return

        pos = labels.count(1)
        neg = labels.count(0)
        logger.info(f"[filter] Training samples: {pos} relevant, {neg} off-topic")

        # Encode with sentence-transformers
        logger.info("[filter] Encoding papers with sentence-transformers...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        X = encoder.encode(texts, show_progress_bar=True, batch_size=32)
        y = np.array(labels)

        # Train LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
        logger.info(f"[filter] Cross-val F1: {scores.mean():.3f} ± {scores.std():.3f}")
        clf.fit(X, y)

        # Save model + encoder
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": clf, "encoder_name": "all-MiniLM-L6-v2"}, f)

        self._ml_model      = clf
        self._ml_vectorizer = encoder
        logger.success(f"[filter] ML model trained and saved → {MODEL_PATH}")
        logger.info(f"[filter] F1 score: {scores.mean():.3f} — "
                    f"{'excellent' if scores.mean() > 0.9 else 'good' if scores.mean() > 0.8 else 'retrain with more data'}")

    def _load_ml_model(self):
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        from sentence_transformers import SentenceTransformer
        self._ml_model      = saved["model"]
        self._ml_vectorizer = SentenceTransformer(saved["encoder_name"])

    # ── Utilities ──────────────────────────────────────────────────────────────
    def _log_summary(self, papers, kept, removed, review):
        from collections import Counter
        logger.info("─" * 45)
        logger.info("RELEVANCE FILTER SUMMARY")
        logger.info(f"  Input:           {len(papers)}")
        logger.info(f"  Kept:            {len(kept)} ({100*len(kept)//max(len(papers),1)}%)")
        logger.info(f"  Removed:         {len(removed)} ({100*len(removed)//max(len(papers),1)}%)")
        logger.info(f"  Review queue:    {len(review)}")

        # Stage breakdown
        all_verdicts = []
        stage_counts = Counter()

        # Show which stages resolved papers:------
        audit = Path("data/audit")
        stage_counter = Counter()
        for f in ["kept.json","rejected.json","review.json"]:

            path = audit / f

            if not path.exists():
                continue
            try:
                with open(path) as fp: data = json.load(fp)
                for x in data:
                    stage_counter[x["stage"]] += 1
            except:
                pass

        logger.info("  Stage breakdown:")
        for stage, cnt in sorted(
            stage_counter.items()):
            logger.info(
                f"    {stage:<25} {cnt}")
            

        if removed:
            logger.info(" Sample removed:")
            for p in removed[:3]:
                logger.info(f"    ✗ {p.title[:65]}")
        if review:
            logger.info("  Flagged for review:")
            for p, v in review[:3]:
                logger.info(f"    ? [{v.score:.2f}] {p.title[:55]}")

        # LLM stats
        try:
            verifier = _get_llm_verifier()
            stats = verifier.cache_stats()
            logger.info(f"  LLM cache: {stats['total_cached']} verdicts cached")
        except Exception:
            pass

        logger.info("─" * 45)

    def _save_removed(self, removed: List[PaperRecord]):
        """Saves rejected papers for audit — so nothing is silently lost."""
        if not removed:
            return
        from config import PROC_DIR
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = PROC_DIR / f"rejected_{ts}.json"
        import json
        with open(path, "w") as f:
            json.dump([p.model_dump() for p in removed], f, indent=2,
                      default=str)
        logger.info(f"[filter] Rejected papers saved → {path}")

# ── Unit tests ───
def run_tests():
    f = RelevanceFilter()
    cases = [
        ("Gut microbiome in Crohn disease patients: a cohort study",
         "We recruited 200 patients. 16S rRNA sequencing was performed.",
         ["Gastrointestinal Microbiome", "Humans", "Crohn Disease"],
         True, "human IBD cohort"),

        ("Zebrafish gut microbiome response to dietary changes",
         "Adult zebrafish were fed different diets for 8 weeks.",
         ["Animals", "Zebrafish"], False, "zebrafish study"),

        ("The Microbiota of Homemade Tepache fermented beverage",
         "Tepache is a Mexican fermented beverage from pineapple.",
         [], False, "food fermentation"),

        ("Gut microbiota of gopher tortoises in Florida",
         "Fecal samples from wild-caught tortoises were sequenced.",
         [], False, "tortoise study"),

        ("Shotgun metagenomics reveals gut microbiome changes in T2D",
         "Participants with type 2 diabetes had altered gut microbiota. "
         "Shotgun sequencing of fecal samples was performed. PRJNA123456.",
         ["Microbiota", "Humans"], True, "human T2D metagenomics"),

        ("Soil microbiome diversity in agricultural fields",
         "Rhizosphere samples were collected from wheat fields.",
         [], False, "soil/plant study"),

        ("FMT in antibiotic-treated ICU patients: a pilot RCT",
         "Patients received fecal microbiota transplantation. 16S rRNA.",
         ["Fecal Microbiota Transplantation", "Humans"], True, "human FMT RCT"),
    ]

    print("3-Stage Relevance Filter — Unit Tests")
    print("─" * 60)
    passed = 0
    for title, abstract, mesh, expected, desc in cases:
        paper = PaperRecord(title=title, abstract=abstract, mesh_terms=mesh)
        v = f._evaluate(paper)
        ok = v.keep == expected
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        print(f"         Stage={v.stage} Score={v.score} Review={v.review}")
        print(f"         {v.reason[:70]}")
    print("─" * 60)
    print(f"Result: {passed}/{len(cases)} passed "
          f"{'✅' if passed == len(cases) else '❌'}")


if __name__ == "__main__":
    run_tests()
