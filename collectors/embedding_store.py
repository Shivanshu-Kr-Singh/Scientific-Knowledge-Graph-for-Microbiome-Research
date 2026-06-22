"""
collectors/embedding_store.py

Persistent embedding store with positive/negative partitions.
Uses NumPy brute-force cosine similarity for queries.
Storage layout:
  data/embeddings/positive.npy       — (N, dim) float32 matrix
  data/embeddings/positive_meta.json  — List[EmbeddingMetadata]
  data/embeddings/negative.npy       — (M, dim) float32 matrix
  data/embeddings/negative_meta.json  — List[EmbeddingMetadata]

Thread safety: Uses filelock for atomic append operations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np
from filelock import FileLock
from loguru import logger

from config import EMBEDDING_STORE_DIR, EMBEDDING_LATENCY_WARN_MS


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class EmbeddingMetadata:
    """Metadata stored alongside each embedding vector."""

    doi: Optional[str]
    pmid: Optional[str]
    title: str
    partition: str  # "positive" | "negative"
    added_at: str  # ISO timestamp
    source_stage: Optional[str] = None  # For future use
    confidence_score: Optional[float] = None  # For future use


class SimilarityResult:
    """Single result from a similarity query."""

    def __init__(self, score: float, metadata: EmbeddingMetadata):
        self.score = score  # cosine similarity [0, 1]
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"SimilarityResult(score={self.score:.4f}, "
            f"title='{self.metadata.title[:40]}...')"
        )


# ─── Protocol ─────────────────────────────────────────────────────────────────


class EmbeddingStoreInterface(Protocol):
    """
    Abstract interface for embedding storage.
    Permits future swap to FAISS without modifying calling code.
    """

    def query_similar(
        self, vector: np.ndarray, partition: str, top_k: int = 5
    ) -> List[SimilarityResult]: ...

    def append(self, vector: np.ndarray, metadata: EmbeddingMetadata) -> None: ...

    def contains(self, doi: Optional[str] = None, pmid: Optional[str] = None) -> bool: ...

    @property
    def positive_count(self) -> int: ...

    @property
    def negative_count(self) -> int: ...


# ─── Implementation ──────────────────────────────────────────────────────────


class EmbeddingStore:
    """
    NumPy-backed brute-force embedding store.

    Thread safety: Uses filelock for atomic append operations.
    """

    VALID_PARTITIONS = ("positive", "negative")

    def __init__(self, store_dir: Path | None = None):
        self._store_dir = Path(store_dir) if store_dir else EMBEDDING_STORE_DIR
        self._store_dir.mkdir(parents=True, exist_ok=True)

        # Internal latency tracking
        self._query_latencies: List[float] = []

        # Load partitions into memory
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, List[EmbeddingMetadata]] = {}

        for partition in self.VALID_PARTITIONS:
            self._load_partition(partition)

    # ─── Public API ───────────────────────────────────────────────────────

    def query_similar(
        self, vector: np.ndarray, partition: str, top_k: int = 5
    ) -> List[SimilarityResult]:
        """
        Compute brute-force cosine similarity against the requested partition.
        Returns top-k results ordered by descending similarity score.
        """
        if partition not in self.VALID_PARTITIONS:
            raise ValueError(
                f"Invalid partition '{partition}'. Must be one of {self.VALID_PARTITIONS}"
            )

        start_time = time.perf_counter()

        vectors = self._vectors[partition]
        metadata = self._metadata[partition]

        if vectors.size == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_latency(elapsed_ms)
            return []

        # Cosine similarity: dot(query, vectors) / (norm(query) * norms(vectors))
        query = vector.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_latency(elapsed_ms)
            return []

        vector_norms = np.linalg.norm(vectors, axis=1)
        # Avoid division by zero for any stored zero-norm vectors
        safe_norms = np.where(vector_norms == 0, 1.0, vector_norms)

        similarities = np.dot(vectors, query) / (query_norm * safe_norms)
        # Clamp to [0, 1] for cosine similarity (vectors should be non-negative in practice)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Get top-k indices
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = [
            SimilarityResult(
                score=float(similarities[idx]),
                metadata=metadata[idx],
            )
            for idx in top_indices
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._record_latency(elapsed_ms)

        return results

    def append(self, vector: np.ndarray, metadata: EmbeddingMetadata) -> None:
        """
        Append a vector and its metadata to the appropriate partition.
        Uses filelock for atomic writes to prevent corruption from concurrent access.
        """
        partition = metadata.partition
        if partition not in self.VALID_PARTITIONS:
            raise ValueError(
                f"Invalid partition '{partition}'. Must be one of {self.VALID_PARTITIONS}"
            )

        lock_path = self._store_dir / f"{partition}.lock"
        lock = FileLock(str(lock_path))

        with lock:
            # Reload from disk to pick up any changes from other processes
            self._load_partition(partition)

            vec = vector.astype(np.float32).reshape(1, -1)

            if self._vectors[partition].size == 0:
                self._vectors[partition] = vec
            else:
                self._vectors[partition] = np.vstack(
                    [self._vectors[partition], vec]
                )

            self._metadata[partition].append(metadata)

            # Persist to disk
            self._save_partition(partition)

        logger.debug(
            f"Appended embedding to {partition} partition "
            f"(doi={metadata.doi}, pmid={metadata.pmid})"
        )

    def contains(self, doi: Optional[str] = None, pmid: Optional[str] = None) -> bool:
        """
        Check if a paper is already in the store by DOI or PMID.
        Searches both positive and negative partitions.
        """
        if doi is None and pmid is None:
            return False

        for partition in self.VALID_PARTITIONS:
            for meta in self._metadata[partition]:
                if doi is not None and meta.doi is not None and meta.doi == doi:
                    return True
                if pmid is not None and meta.pmid is not None and meta.pmid == pmid:
                    return True

        return False

    @property
    def positive_count(self) -> int:
        """Number of embeddings in the positive partition."""
        return len(self._metadata["positive"])

    @property
    def negative_count(self) -> int:
        """Number of embeddings in the negative partition."""
        return len(self._metadata["negative"])

    def query_latency_stats(self) -> dict:
        """
        Return latency statistics for similarity queries.

        Returns:
            dict with keys: count, avg_ms, p95_ms
        """
        return {
            "count": len(self._query_latencies),
            "avg_ms": self._rolling_avg_latency_ms(),
            "p95_ms": self._p95_latency_ms(),
        }

    def _rolling_avg_latency_ms(self) -> float:
        """Rolling average of the last 100 recorded query latencies in milliseconds."""
        if not self._query_latencies:
            return 0.0
        window = self._query_latencies[-100:]
        return sum(window) / len(window)

    def _p95_latency_ms(self) -> float:
        """95th percentile of all recorded query latencies in milliseconds."""
        if not self._query_latencies:
            return 0.0
        return float(np.percentile(self._query_latencies, 95))

    # ─── Private Helpers ──────────────────────────────────────────────────

    def _load_partition(self, partition: str) -> None:
        """Load vectors and metadata for a partition from disk."""
        npy_path = self._store_dir / f"{partition}.npy"
        meta_path = self._store_dir / f"{partition}_meta.json"

        # Load vectors
        if npy_path.exists():
            try:
                vectors = np.load(str(npy_path), allow_pickle=False)
                if vectors.ndim != 2:
                    raise ValueError(f"Expected 2D array, got {vectors.ndim}D")
                self._vectors[partition] = vectors.astype(np.float32)
            except Exception as e:
                logger.error(
                    f"Corrupted .npy file for {partition} partition: {e}. "
                    f"Reinitializing empty partition."
                )
                self._vectors[partition] = np.empty((0, 0), dtype=np.float32)
                self._metadata[partition] = []
                self._save_partition(partition)
                return
        else:
            self._vectors[partition] = np.empty((0, 0), dtype=np.float32)

        # Load metadata
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                self._metadata[partition] = [
                    EmbeddingMetadata(
                        doi=m.get("doi"),
                        pmid=m.get("pmid"),
                        title=m.get("title", ""),
                        partition=m.get("partition", partition),
                        added_at=m.get("added_at", ""),
                        source_stage=m.get("source_stage"),
                        confidence_score=m.get("confidence_score"),
                    )
                    for m in raw_meta
                ]
            except Exception as e:
                logger.error(
                    f"Corrupted metadata file for {partition} partition: {e}. "
                    f"Reinitializing empty partition."
                )
                self._vectors[partition] = np.empty((0, 0), dtype=np.float32)
                self._metadata[partition] = []
                self._save_partition(partition)
                return
        else:
            self._metadata[partition] = []

    def _save_partition(self, partition: str) -> None:
        """Persist vectors and metadata for a partition to disk."""
        npy_path = self._store_dir / f"{partition}.npy"
        meta_path = self._store_dir / f"{partition}_meta.json"

        vectors = self._vectors[partition]
        if vectors.size > 0:
            np.save(str(npy_path), vectors)
        else:
            # Save an empty 2D array placeholder
            np.save(str(npy_path), np.empty((0, 0), dtype=np.float32))

        metadata_dicts = [asdict(m) for m in self._metadata[partition]]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dicts, f, ensure_ascii=False, indent=2)

    def _record_latency(self, elapsed_ms: float) -> None:
        """Record a query latency measurement and warn if threshold exceeded."""
        self._query_latencies.append(elapsed_ms)

        # Emit warning if rolling average exceeds configured threshold
        avg = self._rolling_avg_latency_ms()
        if avg > EMBEDDING_LATENCY_WARN_MS:
            logger.warning(
                f"[embedding_store] Latency warning: rolling avg={avg:.1f}ms > {EMBEDDING_LATENCY_WARN_MS}ms"
            )
