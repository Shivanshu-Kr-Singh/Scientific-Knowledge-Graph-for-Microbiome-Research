"""
collectors/embedding_filter.py

Embedding-based classification stage (Stage 3.5).

Evaluates papers against positive and negative embedding partitions to make
high-confidence keep/reject decisions without invoking the LLM verifier.
Papers that don't meet confidence thresholds are routed as BORDERLINE to Stage 4.

Thresholds are configurable via config.py environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from collectors.embedding_model import EmbeddingModel
from collectors.embedding_store import EmbeddingStore
from config import (
    EMBEDDING_POS_KEEP_THRESHOLD,
    EMBEDDING_NEG_REJECT_THRESHOLD,
    EMBEDDING_CROSS_CEILING,
    EMBEDDING_MIN_PARTITION_SIZE,
)


@dataclass
class EmbeddingVerdict:
    """Result of embedding-based classification for a single paper."""

    decision: str  # "KEEP" | "REJECT" | "BORDERLINE" | "INSUFFICIENT_DATA"
    pos_similarity: float
    neg_similarity: float
    reason: str
    stage: str = "stage3_5_embedding"


class EmbeddingFilter:
    """
    Stage 3.5 embedding-based classifier.

    Decision logic:
      KEEP:              pos_sim >= POS_KEEP_THRESHOLD AND neg_sim < CROSS_CEILING
      REJECT:            neg_sim >= NEG_REJECT_THRESHOLD AND pos_sim < CROSS_CEILING
      BORDERLINE:        everything else → routes to Stage 4
      INSUFFICIENT_DATA: either partition has < MIN_PARTITION_SIZE embeddings

    Minimum store size: MIN_PARTITION_SIZE embeddings per partition before making decisions.
    """

    POS_KEEP_THRESHOLD = EMBEDDING_POS_KEEP_THRESHOLD
    NEG_REJECT_THRESHOLD = EMBEDDING_NEG_REJECT_THRESHOLD
    CROSS_CEILING = EMBEDDING_CROSS_CEILING
    MIN_PARTITION_SIZE = EMBEDDING_MIN_PARTITION_SIZE

    def __init__(self, embedding_model: EmbeddingModel, embedding_store: EmbeddingStore):
        self._model = embedding_model
        self._store = embedding_store

    def evaluate(self, paper) -> EmbeddingVerdict:
        """
        Evaluate a paper against both embedding partitions.

        Parameters
        ----------
        paper : object
            Paper object with at minimum `title` and `abstract` attributes.

        Returns
        -------
        EmbeddingVerdict
            Classification result with similarity scores and reasoning.
        """
        title = getattr(paper, "title", "") or ""
        abstract = getattr(paper, "abstract", "") or ""

        # Check minimum partition sizes
        pos_count = self._store.positive_count
        neg_count = self._store.negative_count

        if pos_count < self.MIN_PARTITION_SIZE or neg_count < self.MIN_PARTITION_SIZE:
            reason = (
                f"Insufficient data: positive={pos_count}, negative={neg_count}, "
                f"minimum required={self.MIN_PARTITION_SIZE} per partition"
            )
            logger.debug(
                f"[Stage 3.5] INSUFFICIENT_DATA for '{title[:60]}…' — {reason}"
            )
            return EmbeddingVerdict(
                decision="INSUFFICIENT_DATA",
                pos_similarity=0.0,
                neg_similarity=0.0,
                reason=reason,
            )

        # Encode paper
        embedding = self._model.encode_paper(title, abstract if abstract else None)

        # Query both partitions for top-1 similarity
        pos_results = self._store.query_similar(embedding, partition="positive", top_k=1)
        neg_results = self._store.query_similar(embedding, partition="negative", top_k=1)

        pos_sim = pos_results[0].score if pos_results else 0.0
        neg_sim = neg_results[0].score if neg_results else 0.0

        # Apply threshold logic
        if pos_sim >= self.POS_KEEP_THRESHOLD and neg_sim < self.CROSS_CEILING:
            decision = "KEEP"
            reason = (
                f"pos_sim={pos_sim:.4f} >= {self.POS_KEEP_THRESHOLD} "
                f"AND neg_sim={neg_sim:.4f} < {self.CROSS_CEILING}"
            )
        elif neg_sim >= self.NEG_REJECT_THRESHOLD and pos_sim < self.CROSS_CEILING:
            decision = "REJECT"
            reason = (
                f"neg_sim={neg_sim:.4f} >= {self.NEG_REJECT_THRESHOLD} "
                f"AND pos_sim={pos_sim:.4f} < {self.CROSS_CEILING}"
            )
        else:
            decision = "BORDERLINE"
            reason = (
                f"No confident threshold met: "
                f"pos_sim={pos_sim:.4f}, neg_sim={neg_sim:.4f}"
            )

        logger.debug(
            f"[Stage 3.5] {decision} for '{title[:60]}…' — "
            f"pos_sim={pos_sim:.4f}, neg_sim={neg_sim:.4f} | {reason}"
        )

        return EmbeddingVerdict(
            decision=decision,
            pos_similarity=pos_sim,
            neg_similarity=neg_sim,
            reason=reason,
        )
