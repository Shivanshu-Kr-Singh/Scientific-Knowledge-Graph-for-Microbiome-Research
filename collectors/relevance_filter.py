"""
3-stage relevance filter pipeline.
PIPELINE:
Stage 1 — Metadata filter (PubMed MeSH only, highest precision)
Stage 2 — Weighted rule scorer(all sources, reads from stage2_rules.yaml)
Stage 3 — ML classifier(sentence-transformers+LogisticRegression)
Metagenomics gate — project-specific requirement

FLOW PER PAPER:
PubMed papers →Stage 1 first (MeSH is reliable), then Stage 2 if borderline
Other sources → Stage 2 directly, then Stage 3 if borderline
All papers → metagenomics gate as final check

DESIGN PRINCIPLES:
- All term lists live in config/stage1_mesh.yaml (Stage 1) and config/stage2_rules.yaml (Stage 2) — not hardcoded here
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
from collectors.metadata_filter import MetadataFilter
from collectors.ml_classifier import MLClassifier
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


CONFIG_PATH = Path(__file__).parent.parent / "config" / "stage2_rules.yaml"
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

        # Standalone stage modules
        self._metadata_filter = MetadataFilter()
        self._ml_classifier   = MLClassifier()

        # Legacy inline ML (kept for train_ml_model compatibility)
        self._ml_model      = self._ml_classifier._model
        self._ml_vectorizer = self._ml_classifier._encoder

        status = "active" if self._ml_classifier.trained else "inactive (not trained yet)"
        logger.info(f"[filter] Pipeline ready | Stage3 ML: {status}")

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
        """
        Routes each paper through stages in correct order:
        Stage1(MeSH) → Stage2(rules) → Gate → Stage3(ML) → Stage4(LLM)

        Cheap/precise stages first. Each stage only runs if the previous
        stage couldn't make a confident decision.
        """
        source = (paper.source or "").lower()

        # ── Stage 1: MeSH metadata filter (PubMed papers only) ───────────────
        # Use the standalone MetadataFilter module for clean separation
        if "pubmed" in source:
            mv = self._metadata_filter.evaluate(paper)
            if mv.decision == "KEEP":
                return FilterVerdict(keep=True,  score=mv.score,
                                     stage=mv.stage, reason=mv.reason)
            if mv.decision == "REJECT":
                return FilterVerdict(keep=False, score=mv.score,
                                     stage=mv.stage, reason=mv.reason)
            # UNKNOWN → fall through to Stage 2

        # ── Stage 2: Weighted rule scorer (all sources) ───────────────────────
        v = self._stage2_rules(paper)

        # Confident reject from rules
        if v.score < self.thresholds["review"]:
            return v

        # Confident keep from rules → apply gate first, then return
        if v.score >= self.thresholds["keep"]:
            return self._metagenomics_gate(paper, v)

        # ── Borderline [review, keep) — continue through remaining stages ────

        # ── Metagenomics gate BEFORE ML/LLM ──────────────────────────────────
        # Eliminates papers lacking sequencing terms cheaply
        # No point sending soil/animal papers to ML or LLM
        gate_v = self._metagenomics_gate(paper, v)
        if not gate_v.keep and not gate_v.review:
            # Gate confidently rejected — stop here
            return gate_v

        # ── Stage 3: ML classifier ────────────────────────────────────────────
        ml_verdict = self._ml_classifier.evaluate(paper)
        if ml_verdict.decision == "KEEP":
            return FilterVerdict(keep=True, score=ml_verdict.prob,
                                 stage=ml_verdict.stage, reason=ml_verdict.reason)
        if ml_verdict.decision == "REJECT":
            return FilterVerdict(keep=False, score=ml_verdict.prob,
                                 stage=ml_verdict.stage, reason=ml_verdict.reason)
        # BORDERLINE or UNTRAINED → Stage 4 LLM

        # ── Stage 4: LLM verifier (only truly uncertain papers) ───────────────
        return self._stage4_llm(paper, gate_v)

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
        Additive weighted scoring from stage2_rules.yaml terms.
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
        Disabled if metagenomics_gate.enabled = false in stage2_rules.yaml.
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

    # ── ML training ───────────────────────────────────────────────────────────

    def train_ml_model(self, papers: List[PaperRecord],
                       rejected_papers: Optional[List[PaperRecord]] = None):
        """
        Trains ML classifier with balanced, quality-focused sampling.

        WHAT CHANGED FOR 60k SCALE:
          - `papers` should now be ALL collected papers across ALL runs
            (pass orchestrator.load_all() not load_latest())
          - `rejected_papers` can be pre-loaded via orchestrator.load_all_rejected()
            to avoid re-reading every rejected_*.json file on each train run
          - Deduplication by content_hash / title before training — prevents
            duplicate papers from multiple runs from inflating counts
          - Encoding batch_size raised to 128 for GPU / 64 for CPU throughput
          - Hard negative cap raised: fills up to 3× positives instead of 2×
            to give the model more ambiguous examples at scale
          - Min samples raised: 100+ needed before training at production scale
          - After training, the live MLClassifier instance is hot-reloaded so
            the new weights are used immediately without a process restart

        STRATEGY:
          Positives: all collected papers with rule score ≥ 0.85
          Negatives: from rejected files, prioritising HARD cases (score 0.30–0.65)
                     over medium (0.15–0.30) over easy (< 0.15)
          Balance:   caps negatives at 3× positives
          Model:     sentence-transformers (all-MiniLM-L6-v2) + LogisticRegression
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.utils import shuffle as sk_shuffle
            import numpy as np
        except ImportError:
            logger.error("[filter] Run: pip install sentence-transformers scikit-learn")
            return

        logger.info("[filter] Building balanced training dataset across all runs...")

        # ── Deduplicate incoming papers by content_hash / dedup_key ──────────
        seen_keys: set = set()
        unique_papers: List[PaperRecord] = []
        for paper in papers:
            key = (paper.content_hash or paper.get_dedup_key())
            if key not in seen_keys:
                seen_keys.add(key)
                unique_papers.append(paper)

        logger.info(f"[filter] Collected papers after dedup: {len(unique_papers)} "
                    f"(from {len(papers)} total across all runs)")

        # ── Collect positives: rule score ≥ 0.85 ─────────────────────────────
        pos_texts: List[str] = []
        for paper in unique_papers:
            v = self._stage2_rules(paper)
            if v.score >= 0.85:
                pos_texts.append(
                    f"{paper.title or ''} {paper.abstract or ''}".strip()
                )

        MIN_POSITIVES = 50  # raised from 30 — too few produces unreliable models
        if len(pos_texts) < MIN_POSITIVES:
            logger.warning(
                f"[filter] Only {len(pos_texts)} positive samples — need {MIN_POSITIVES}+. "
                f"Collect more papers (MAX_PER_SOURCE=500+) before training."
            )
            return

        logger.info(f"[filter] Positives: {len(pos_texts)} papers (rule score ≥ 0.85)")

        # ── Collect negatives from rejected papers ────────────────────────────
        # If pre-loaded externally, use those. Otherwise fall back to globbing
        # rejected_*.json files — but deduplicate across files first.
        if rejected_papers is None:
            from config import PROC_DIR
            import glob as _glob

            seen_rej: set = set()
            rejected_papers = []
            for path in _glob.glob(str(PROC_DIR / "rejected_*.json")):
                try:
                    with open(path, encoding="utf-8") as f:
                        batch = json.load(f)
                    for r in batch:
                        try:
                            p   = PaperRecord(**r)
                            key = p.content_hash or p.get_dedup_key()
                            if key not in seen_rej:
                                seen_rej.add(key)
                                rejected_papers.append(p)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"[filter] Could not load {path}: {e}")

            logger.info(f"[filter] Rejected papers loaded: {len(rejected_papers)} unique")

        # Bucket rejected papers by difficulty (rule score)
        hard_negs:   List[str] = []   # 0.30–0.65 — model needs these most
        medium_negs: List[str] = []   # 0.15–0.30
        easy_negs:   List[str] = []   # < 0.15 — trivially irrelevant

        for paper in rejected_papers:
            title    = paper.title    or ""
            abstract = paper.abstract or ""
            if not title and not abstract:
                continue
            try:
                v    = self._stage2_rules(paper)
                text = f"{title} {abstract}".strip()
                if v.score >= 0.30:
                    hard_negs.append(text)
                elif v.score >= 0.15:
                    medium_negs.append(text)
                else:
                    easy_negs.append(text)
            except Exception:
                easy_negs.append(f"{title} {abstract}".strip())

        logger.info(
            f"[filter] Negatives available — hard: {len(hard_negs)}, "
            f"medium: {len(medium_negs)}, easy: {len(easy_negs)}"
        )

        # Fill up to 3× positives (more at scale → better decision boundary)
        import random
        random.seed(42)

        total_neg_available = len(hard_negs) + len(medium_negs) + len(easy_negs)
        target_neg = min(len(pos_texts) * 3, total_neg_available)

        selected_negs: List[str] = []
        selected_negs.extend(
            random.sample(hard_negs, min(len(hard_negs), target_neg))
        )
        remaining = target_neg - len(selected_negs)
        if remaining > 0:
            selected_negs.extend(
                random.sample(medium_negs, min(len(medium_negs), remaining))
            )
        remaining = target_neg - len(selected_negs)
        if remaining > 0:
            selected_negs.extend(
                random.sample(easy_negs, min(len(easy_negs), remaining))
            )

        logger.info(
            f"[filter] Selected {len(selected_negs)} negatives "
            f"(target: {target_neg}, hard: {min(len(hard_negs), target_neg)})"
        )

        # ── Build final dataset ───────────────────────────────────────────────
        texts  = pos_texts + selected_negs
        labels = [1] * len(pos_texts) + [0] * len(selected_negs)
        pos    = labels.count(1)
        neg    = labels.count(0)

        logger.info(
            f"[filter] Final training set: {pos} relevant, {neg} off-topic "
            f"(ratio {neg / max(pos, 1):.1f}:1)"
        )

        MIN_TOTAL = 100   # raised from 50 — need sufficient volume at 60k scale
        if len(texts) < MIN_TOTAL:
            logger.warning(
                f"[filter] Only {len(texts)} total samples — need {MIN_TOTAL}+. "
                f"Collect more data before training."
            )
            return

        # ── Encode with sentence-transformers ─────────────────────────────────
        # batch_size=128 is faster on GPU; use 64 if you see OOM errors
        logger.info(
            f"[filter] Encoding {len(texts)} texts with "
            f"sentence-transformers (all-MiniLM-L6-v2)..."
        )
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        X = encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=128,          # was 32 — 4× faster encoding at scale
            convert_to_numpy=True,
        )
        y      = np.array(labels)
        X, y   = sk_shuffle(X, y, random_state=42)

        # ── Train with stratified cross-validation ────────────────────────────
        cv_folds = min(5, min(pos, neg))
        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,           # raised from 1000 — large datasets need more iters
            class_weight="balanced",
            random_state=42,
            solver="saga",           # faster for large n_samples vs default lbfgs
            n_jobs=-1,               # use all CPU cores
        )
        cv     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1", n_jobs=-1)
        logger.info(
            f"[filter] Cross-val F1 ({cv_folds}-fold): "
            f"{scores.mean():.3f} ± {scores.std():.3f}"
        )
        clf.fit(X, y)

        # ── Save ──────────────────────────────────────────────────────────────
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":        clf,
                "encoder_name": "all-MiniLM-L6-v2",
                "trained_on":   len(texts),
                "n_samples":    len(texts),
                "pos_samples":  pos,
                "neg_samples":  neg,
                "f1":           float(scores.mean()),
            }, f)

        # ── Hot-reload the live MLClassifier so new weights are used NOW ──────
        # Without this, the new model only takes effect after a process restart
        # because _ml_classifier still holds the old (or untrained) instance.
        self._ml_model      = clf
        self._ml_vectorizer = encoder
        self._ml_classifier._model   = clf
        self._ml_classifier._encoder = encoder
        self._ml_classifier.trained  = True

        quality = (
            "excellent"              if scores.mean() > 0.92 else
            "good — ready to use"    if scores.mean() > 0.85 else
            "moderate — collect more data and retrain"
        )
        logger.success(
            f"[filter] Model saved → {MODEL_PATH} | "
            f"trained on {len(texts)} samples | "
            f"F1={scores.mean():.3f} — {quality}"
        )

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
        with open(path, "w", encoding="utf-8") as f:
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
