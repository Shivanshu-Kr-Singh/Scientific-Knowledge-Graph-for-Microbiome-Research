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
from collectors.embedding_filter import EmbeddingVerdict
from collectors.embedding_store import EmbeddingStore, EmbeddingMetadata
from config import (
    BLENDED_CONFIDENCE_LOW,
    BLENDED_CONFIDENCE_HIGH,
    GROWTH_KEEP_THRESHOLD,
    GROWTH_REJECT_THRESHOLD,
)
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


# Lazy singletons for embedding model and store — keeps __init__ lightweight
_embedding_model = None
_embedding_store = None


def _get_embedding_model():
    """Lazily initialize the shared EmbeddingModel singleton."""
    global _embedding_model
    if _embedding_model is None:
        from collectors.embedding_model import EmbeddingModel
        _embedding_model = EmbeddingModel()
    return _embedding_model


def _get_embedding_store():
    """Lazily initialize the shared EmbeddingStore singleton."""
    global _embedding_store
    if _embedding_store is None:
        from collectors.embedding_store import EmbeddingStore
        _embedding_store = EmbeddingStore()
    return _embedding_store


CONFIG_PATH = Path(__file__).parent.parent / "config" / "stage2_rules.yaml"
GATE_CONFIG_PATH = Path(__file__).parent.parent / "config" / "metagenomics_gate.yaml"
MODEL_PATH  = Path(__file__).parent.parent / "config" / "relevance_model.pkl"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_gate_config() -> dict:
    """Load metagenomics gate configuration from its dedicated YAML file.
    The enabled/disabled state is controlled by METAGENOMICS_GATE_ENABLED in .env.
    """
    from config import METAGENOMICS_GATE_ENABLED
    if GATE_CONFIG_PATH.exists():
        with open(GATE_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        cfg["enabled"] = METAGENOMICS_GATE_ENABLED
        return cfg
    return {"enabled": METAGENOMICS_GATE_ENABLED, "terms": []}


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
        self.gate_cfg   = _load_gate_config()
        self.mesh_keep  = [m.lower() for m in self.cfg.get("mesh_keep", [])]
        self.mesh_human = [m.lower() for m in self.cfg.get("mesh_human_signal", [])]
        self.mesh_animal= [m.lower() for m in self.cfg.get("mesh_animal_only", [])]

        # Load Stage 2 thresholds — auto-calibrated values take priority
        # over stage2_rules.yaml static values. Falls back to yaml if no
        # calibration file exists yet.
        from collectors.stage2_calibrator import load_calibrated_thresholds
        yaml_thresholds = self.cfg.get("thresholds", {"keep": 0.70, "review": 0.40})
        cal_keep, cal_review = load_calibrated_thresholds()

        self.thresholds = {
            "keep":   cal_keep,
            "review": cal_review,
        }

        # Log which source the thresholds came from
        from collectors.stage2_calibrator import CALIBRATION_PATH
        source = "calibration file" if CALIBRATION_PATH.exists() else "stage2_rules.yaml defaults"
        logger.info(
            f"[filter] Stage 2 thresholds: "
            f"keep≥{self.thresholds['keep']:.3f}  "
            f"review≥{self.thresholds['review']:.3f}  "
            f"(source: {source})"
        )

        # Standalone stage modules
        self._metadata_filter = MetadataFilter()
        self._ml_classifier   = MLClassifier()

        # Legacy inline ML (kept for train_ml_model compatibility)
        self._ml_model      = self._ml_classifier._model
        self._ml_vectorizer = self._ml_classifier._encoder

        # ── Layer 1 scale components (lazy, fail-graceful) ────────────────────
        # All components below are optional — if sentence-transformers or any
        # dependency is missing, the pipeline falls back to the original
        # Stage 4 LLM path.
        self._embedding_model: Optional[object] = None
        self._embedding_store: Optional[object] = None
        self._embedding_filter: Optional[object] = None
        self._semantic_cache: Optional[object] = None
        self._batched_verifier: Optional[object] = None
        self._metrics_logger: Optional[object] = None

        try:
            from collectors.embedding_model import EmbeddingModel as _EM
            self._embedding_model = _EM()
        except Exception as e:
            logger.debug(f"[filter] EmbeddingModel not available: {e}")

        try:
            from collectors.embedding_store import EmbeddingStore as _ES
            self._embedding_store = _ES()
        except Exception as e:
            logger.debug(f"[filter] EmbeddingStore not available: {e}")

        try:
            from collectors.embedding_filter import EmbeddingFilter as _EF
            if self._embedding_model and self._embedding_store:
                self._embedding_filter = _EF(
                    embedding_model=self._embedding_model,
                    embedding_store=self._embedding_store,
                )
            else:
                logger.debug("[filter] EmbeddingFilter skipped — model or store unavailable")
        except Exception as e:
            logger.debug(f"[filter] EmbeddingFilter not available: {e}")

        try:
            from collectors.llm_verifier import SemanticCache as _SC
            self._semantic_cache = _SC()
        except Exception as e:
            logger.debug(f"[filter] SemanticCache not available: {e}")

        try:
            from collectors.llm_verifier import BatchedVerifier as _BV
            self._batched_verifier = _BV()
        except Exception as e:
            logger.debug(f"[filter] BatchedVerifier not available: {e}")

        try:
            from collectors.metrics_logger import MetricsLogger as _ML
            self._metrics_logger = _ML()
        except Exception as e:
            logger.debug(f"[filter] MetricsLogger not available: {e}")

        status = "active" if self._ml_classifier.trained else "inactive (not trained yet)"
        stage35_status = "active" if self._embedding_filter else "inactive"
        logger.info(
            f"[filter] Pipeline ready | Stage3 ML: {status} | "
            f"Stage3.5 Embedding: {stage35_status}"
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def filter(
        self,
        papers: List[PaperRecord],
    ) -> Tuple[List[PaperRecord], List[PaperRecord], List[Tuple]]:
        """
        Runs the full multi-stage pipeline on a list of papers.

        After evaluating all papers:
          1. Invokes embedding store growth for each (paper, verdict) pair
          2. Records pipeline run metrics (stage counts, LLM stats, store size)

        Returns: (kept, removed, review_queue)
        """
        kept, removed, review = [], [], []
        stage_counts = {
            "stage1": 0,
            "stage2": 0,
            "stage3": 0,
            "stage3_5": 0,
            "stage4": 0,
            "gate": 0,
        }

        # Track LLM-related statistics for metrics
        llm_calls = 0
        semantic_cache_hits = 0

        # Collect (paper, verdict) pairs for post-loop processing
        paper_verdicts: List[Tuple[PaperRecord, FilterVerdict]] = []

        for paper in tqdm(papers, desc="Relevance filtering"):
            verdict = self._evaluate(paper)
            AuditLogger.log(paper, verdict)

            # Categorize stage for counting
            stage_key = verdict.stage.split("_")[0]
            # Normalize stage3.5 variants (stage3_5_embedding → stage3_5)
            if "stage3_5" in verdict.stage or "stage3.5" in verdict.stage:
                stage_key = "stage3_5"
            elif "stage4" in verdict.stage:
                stage_key = "stage4"
                # Count LLM calls vs semantic cache hits
                if "semantic_cache" in verdict.stage:
                    semantic_cache_hits += 1
                else:
                    llm_calls += 1
            stage_counts[stage_key] = stage_counts.get(stage_key, 0) + 1

            paper_verdicts.append((paper, verdict))

            if verdict.review:
                review.append((paper, verdict))
            if verdict.keep:
                kept.append(paper)
            else:
                removed.append(paper)

        # ── Post-loop: Embedding Store Growth ─────────────────────────────────
        # Feed confident verdicts back into the store for progressive learning
        for paper, verdict in paper_verdicts:
            try:
                self._embedding_store_growth(paper, verdict.score)
            except Exception as e:
                logger.debug(f"[filter] Store growth failed for '{(paper.title or '')[:40]}': {e}")

        # ── Post-loop: Pipeline Metrics ───────────────────────────────────────
        if self._metrics_logger is not None:
            try:
                from collectors.metrics_logger import PipelineMetrics

                # Embedding store sizes at run completion
                store_positive = 0
                store_negative = 0
                avg_latency_ms = 0.0
                p95_latency_ms = 0.0

                if self._embedding_store is not None:
                    try:
                        store_positive = self._embedding_store.positive_count
                        store_negative = self._embedding_store.negative_count
                    except Exception:
                        pass
                    try:
                        latency_stats = self._embedding_store.query_latency_stats()
                        avg_latency_ms = latency_stats.get("avg_ms", 0.0)
                        p95_latency_ms = latency_stats.get("p95_ms", 0.0)
                    except Exception:
                        pass

                metrics = PipelineMetrics(
                    timestamp=datetime.utcnow().isoformat(),
                    total_papers=len(papers),
                    stage1_resolved=stage_counts.get("stage1", 0),
                    stage2_resolved=stage_counts.get("stage2", 0),
                    gate_resolved=stage_counts.get("gate", 0),
                    stage3_resolved=stage_counts.get("stage3", 0),
                    stage3_5_resolved=stage_counts.get("stage3_5", 0),
                    stage4_resolved=stage_counts.get("stage4", 0),
                    llm_calls=llm_calls,
                    semantic_cache_hits=semantic_cache_hits,
                    batch_count=0,  # Batch stats tracked in future batch mode
                    batch_retries=0,
                    embedding_store_positive=store_positive,
                    embedding_store_negative=store_negative,
                    avg_embedding_latency_ms=avg_latency_ms,
                    p95_embedding_latency_ms=p95_latency_ms,
                )
                self._metrics_logger.record(metrics)
                logger.info(
                    f"[filter] Pipeline metrics recorded: {len(papers)} papers, "
                    f"LLM calls={llm_calls}, cache hits={semantic_cache_hits}"
                )
            except Exception as e:
                logger.warning(f"[filter] Failed to record pipeline metrics: {e}")

        self._log_summary(papers, kept, removed, review)
        self._save_removed(removed)
        self._export_rejected_csv(paper_verdicts)
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
        # BORDERLINE or UNTRAINED → Stage 3.5 embedding filter

        # ── Stage 3.5: Embedding-based similarity filter ──────────────────────
        emb_verdict = self._stage3_5_embedding(paper)
        if emb_verdict is not None and emb_verdict.decision in ("KEEP", "REJECT"):
            # Stage 3.5 has a confident decision — check disagreement router
            blended_confidence = (v.score + emb_verdict.pos_similarity) / 2.0
            route_to_llm, route_reason = self._disagreement_router(
                paper, v, emb_verdict, blended_confidence
            )
            if not route_to_llm:
                # Router says no LLM needed — accept Stage 3.5 verdict
                keep = emb_verdict.decision == "KEEP"
                score = emb_verdict.pos_similarity if keep else (1.0 - emb_verdict.neg_similarity)
                return FilterVerdict(
                    keep=keep,
                    score=round(score, 3),
                    stage=emb_verdict.stage,
                    reason=f"{emb_verdict.reason} | router: {route_reason}",
                )
            # Router says LLM needed — fall through to Stage 4

        # ── Stage 4: LLM verifier (only truly uncertain papers) ───────────────
        # Try semantic cache first to avoid redundant LLM calls
        if self._semantic_cache is not None and self._embedding_model is not None:
            try:
                paper_embedding = self._embedding_model.encode_paper(
                    paper.title or "", paper.abstract
                )
                cached_verdict = self._semantic_cache.lookup(paper_embedding)
                if cached_verdict is not None:
                    keep = cached_verdict.confidence >= 0.70 and cached_verdict.keep
                    return FilterVerdict(
                        keep=keep,
                        score=round(cached_verdict.confidence, 3),
                        stage="stage4_llm (semantic_cache)",
                        reason=f"semantic_cache_hit: {cached_verdict.reason}",
                    )
            except Exception as e:
                logger.debug(f"[stage4] Semantic cache lookup failed: {e}")

        return self._stage4_llm(paper, gate_v)

    # ── Stage 3.5: Embedding filter ─────────────────────────────────────────────

    def _stage3_5_embedding(self, paper: PaperRecord) -> Optional[EmbeddingVerdict]:
        """
        Delegates to the EmbeddingFilter for Stage 3.5 classification.

        Returns the EmbeddingVerdict if the filter is available and produces
        a result, otherwise returns None (pipeline falls through to Stage 4).
        """
        if self._embedding_filter is None:
            return None

        try:
            verdict = self._embedding_filter.evaluate(paper)
            return verdict
        except Exception as e:
            logger.warning(f"[stage3_5] Embedding filter failed: {e}")
            return None

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

        SCANNED FIELDS (covers all 6 collectors):
          title + abstract     → all sources
          mesh_terms           → PubMed, EuropePMC
          keywords             → PubMed, EuropePMC author keywords
                                 OpenAlex concept display names (e.g. "Gut flora")
                                 Crossref funder names (e.g. "NIH", "Wellcome Trust")
          article_types        → Semantic Scholar types ("JournalArticle", "Review")
                                 OpenAlex types ("article")
                                 Other source categories ("microbiology", "new results")
          journal              → journal name as additional signal
        """
        text = (
            (paper.title or "")                    + " " +
            (paper.abstract or "")                 + " " +
            " ".join(paper.mesh_terms or [])       + " " +
            " ".join(paper.keywords or [])         + " " +
            " ".join(paper.article_types or [])    + " " +
            (paper.journal or "")
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
        Calls Ollama LLM for borderline papers that Stage 2+3 couldn't resolve.
        COST CONTROL:
          - Only called when score is in [review_threshold, keep_threshold]
          - All responses cached by content hash
          - Typically 5-10% of total papers
        REQUIRES: Ollama running locally (ollama serve)"""
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

    # ── Disagreement Router ───────────────────────────────────────────────────

    def _disagreement_router(
        self,
        paper: PaperRecord,
        stage2_verdict: FilterVerdict,
        stage3_5_verdict: EmbeddingVerdict,
        blended_confidence: float,
    ) -> tuple:
        """
        Determines whether to route a paper to the LLM verifier.

        Routing conditions (any triggers LLM):
          1. Stage 2 and Stage 3.5 disagree (one keeps, other rejects)
          2. Blended Confidence falls in the uncertain zone [0.40, 0.70]

        If neither condition is met, the Stage 3.5 verdict is accepted as final.

        Returns
        -------
        tuple[bool, str]
            (route_to_llm, reason) — True if LLM verification is needed.
        """
        # Determine Stage 2 keep/reject based on threshold
        stage2_keeps = stage2_verdict.score >= self.thresholds["keep"]

        # Determine Stage 3.5 keep/reject
        stage3_5_keeps = stage3_5_verdict.decision == "KEEP"
        stage3_5_rejects = stage3_5_verdict.decision == "REJECT"

        # Condition 1: Verdict disagreement
        # Disagreement = one stage says keep while the other says reject
        verdicts_disagree = (
            (stage2_keeps and stage3_5_rejects)
            or (not stage2_keeps and stage3_5_keeps)
        )

        if verdicts_disagree:
            reason = (
                f"disagreement: stage2={'keep' if stage2_keeps else 'reject'}, "
                f"stage3_5={stage3_5_verdict.decision}"
            )
            logger.info(
                f"[disagreement_router] Routing to LLM — {reason} | "
                f"paper='{(paper.title or '')[:60]}'"
            )
            return (True, reason)

        # Condition 2: Blended confidence in uncertain zone [LOW, HIGH]
        if BLENDED_CONFIDENCE_LOW <= blended_confidence <= BLENDED_CONFIDENCE_HIGH:
            reason = f"borderline confidence: {blended_confidence:.4f}"
            logger.info(
                f"[disagreement_router] Routing to LLM — {reason} | "
                f"paper='{(paper.title or '')[:60]}'"
            )
            return (True, reason)

        # Neither condition met — accept Stage 3.5 verdict as final
        reason = "confident agreement — accepting Stage 3.5 verdict"
        logger.debug(
            f"[disagreement_router] No LLM needed — {reason} | "
            f"paper='{(paper.title or '')[:60]}'"
        )
        return (False, reason)

    # ── Embedding Store Growth ──────────────────────────────────────────────────

    def _embedding_store_growth(self, paper: PaperRecord, score: float) -> None:
        """
        Appends paper embedding to the Embedding Store after a final verdict.

        Growth rules:
          - score >= 0.80 → positive partition
          - score <= 0.20 → negative partition
          - 0.20 < score < 0.80 → skip (borderline, no feedback)
          - paper already in store (by DOI/PMID) → skip

        Handles encoding errors gracefully to avoid crashing the pipeline.
        """
        # Skip borderline scores — no feedback signal
        if GROWTH_REJECT_THRESHOLD < score < GROWTH_KEEP_THRESHOLD:
            logger.debug(
                f"[store_growth] Skip borderline score={score:.3f} | "
                f"paper='{(paper.title or '')[:50]}'"
            )
            return

        try:
            store = _get_embedding_store()
        except Exception as e:
            logger.warning(f"[store_growth] Could not initialize embedding store: {e}")
            return

        # Deduplication check
        doi = getattr(paper, "doi", None)
        pmid = getattr(paper, "pmid", None)

        if store.contains(doi=doi, pmid=pmid):
            logger.debug(
                f"[store_growth] Paper already in store (doi={doi}, pmid={pmid}) — skipping"
            )
            return

        # Determine target partition
        if score >= GROWTH_KEEP_THRESHOLD:
            partition = "positive"
        elif score <= GROWTH_REJECT_THRESHOLD:
            partition = "negative"
        else:
            # Shouldn't reach here due to early return above, but defensive
            return

        # Encode the paper
        try:
            model = _get_embedding_model()
            embedding = model.encode_paper(paper.title or "", paper.abstract)
        except Exception as e:
            logger.warning(
                f"[store_growth] Encoding failed for '{(paper.title or '')[:50]}': {e}"
            )
            return

        # Build metadata
        metadata = EmbeddingMetadata(
            doi=doi,
            pmid=pmid,
            title=paper.title or "",
            partition=partition,
            added_at=datetime.utcnow().isoformat(),
        )

        # Append to store
        try:
            store.append(vector=embedding, metadata=metadata)
            logger.debug(
                f"[store_growth] Appended to {partition} partition | "
                f"score={score:.3f} | paper='{(paper.title or '')[:50]}'"
            )
        except Exception as e:
            logger.warning(
                f"[store_growth] Append failed for '{(paper.title or '')[:50]}': {e}"
            )

    # Alias used in pipeline wiring (task 14.1 calls _store_growth)
    _store_growth = _embedding_store_growth

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

        # ── Calibrate Stage 2 thresholds from the same paper set ─────────────
        # Now that the ML model is trained, also update Stage 2 keep/review
        # thresholds so they reflect the actual score distribution of your data.
        logger.info("[filter] Running Stage 2 threshold calibration...")
        try:
            from collectors.stage2_calibrator import calibrate_stage2_thresholds
            new_keep, new_review = calibrate_stage2_thresholds(
                papers          = unique_papers,
                stage2_scorer   = self._stage2_rules,
                metadata_filter = self._metadata_filter,
            )
            # Hot-reload thresholds into this running instance immediately
            self.thresholds["keep"]   = new_keep
            self.thresholds["review"] = new_review
            logger.success(
                f"[filter] Stage 2 thresholds updated: "
                f"keep≥{new_keep:.3f}  review≥{new_review:.3f}"
            )
        except Exception as e:
            logger.warning(f"[filter] Stage 2 calibration failed: {e} — keeping current thresholds")

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

    def _export_rejected_csv(self, paper_verdicts: List[Tuple]) -> None:
        """
        Export a CSV of all rejected papers with stage and reason.

        Columns: title, doi, source, stage, score, reason
        Output: data/audit/rejected_report_YYYYMMDD_HHMMSS.csv
        """
        import csv
        from config import DATA_DIR

        rejected_rows = []
        for paper, verdict in paper_verdicts:
            if not verdict.keep and not verdict.review:
                rejected_rows.append({
                    "title": (paper.title or "")[:150],
                    "doi": getattr(paper, "doi", "") or "",
                    "source": getattr(paper, "source", "") or "",
                    "stage": self._humanize_stage(verdict.stage),
                    "score": round(verdict.score, 3),
                    "reason": self._humanize_reason(verdict.stage, verdict.reason, verdict.score),
                })

        if not rejected_rows:
            return

        audit_dir = DATA_DIR / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = audit_dir / f"rejected_report_{ts}.csv"

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["title", "doi", "source", "stage", "score", "reason"],
                )
                writer.writeheader()
                writer.writerows(rejected_rows)

            logger.info(
                f"[filter] Rejected papers CSV exported → {csv_path} "
                f"({len(rejected_rows)} papers)"
            )
        except Exception as e:
            logger.warning(f"[filter] Failed to export rejected CSV: {e}")

    @staticmethod
    def _humanize_stage(stage: str) -> str:
        """Convert internal stage name to human-readable label."""
        stage_map = {
            "stage1_mesh": "Stage 1: MeSH Metadata",
            "stage2_rules": "Stage 2: Keyword Rules",
            "stage3_ml": "Stage 3: ML Classifier",
            "stage3_5_embedding": "Stage 3.5: Embedding Similarity",
            "gate_metagenomics": "Metagenomics Gate",
            "stage4_llm": "Stage 4: LLM Verifier",
            "stage4_llm (cached)": "Stage 4: LLM (cached)",
            "stage4_llm (semantic_cache)": "Stage 4: LLM (semantic cache)",
            "stage4_llm_unavailable": "Stage 4: LLM Unavailable",
            "stage4_llm_failed": "Stage 4: LLM Failed",
        }
        for key, label in stage_map.items():
            if key in stage:
                return label
        return stage

    @staticmethod
    def _humanize_reason(stage: str, reason: str, score: float) -> str:
        """
        Convert technical reason strings to human-readable explanations.
        Keeps LLM reasons as-is since they're already natural language.
        """
        # Stage 1: already readable
        if "stage1" in stage:
            reason_map = {
                "animal_only_mesh": "Paper is about animal models only (no human subjects)",
                "no_microbiome_mesh": "Paper has no microbiome-related MeSH terms",
                "no_mesh_terms": "Paper has no MeSH terms assigned yet",
            }
            return reason_map.get(reason, reason)

        # Stage 2: extract matched terms
        if "stage2" in stage:
            if "neg:" in reason:
                # Extract negative terms that caused rejection
                try:
                    neg_part = reason.split("neg:[")[1].rstrip("]")
                    neg_terms = neg_part.split(",") if neg_part else []
                    if neg_terms and neg_terms[0]:
                        return f"Off-topic keywords detected: {', '.join(t.strip() for t in neg_terms[:5])}"
                except (IndexError, ValueError):
                    pass
            return f"Low relevance score ({score:.2f}) — insufficient positive keyword matches"

        # Metagenomics gate
        if "gate" in stage:
            return "No sequencing, bioinformatics, or data availability terms found in title/abstract"

        # Stage 3 ML: translate probability
        if "stage3_ml" in stage or "stage3" in stage and "stage3_5" not in stage:
            if score <= 0.15:
                return f"ML classifier confidence very low ({score:.0%}) — paper unlikely relevant to human microbiome"
            elif score <= 0.30:
                return f"ML classifier confidence low ({score:.0%}) — topic appears outside scope"
            else:
                return f"ML classifier scored below keep threshold ({score:.0%})"

        # Stage 3.5 Embedding: translate similarity scores
        if "stage3_5" in stage:
            if "neg_sim" in reason:
                try:
                    # Extract neg_sim value
                    parts = reason.split("neg_sim=")
                    if len(parts) > 1:
                        neg_val = parts[1][:6]
                        return f"Paper is highly similar to known-irrelevant papers (similarity: {neg_val}) and dissimilar to relevant ones"
                except (IndexError, ValueError):
                    pass
            return "Paper embedding is more similar to known-irrelevant papers than to relevant ones"

        # Stage 4 LLM: already natural language from the model
        if "stage4" in stage or "llm" in stage:
            # Strip the "llm:" prefix if present
            clean = reason.replace("llm:", "").replace("semantic_cache_hit:", "").strip()
            if clean:
                return clean
            return "LLM determined paper is not relevant to human microbiome research"

        # Fallback
        return reason

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
