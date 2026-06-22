"""
collectors/embedding_model.py

Domain-tuned embedding model wrapper with graceful fallback logic.
Primary model: SPECTER2 (allenai/specter2) — optimized for scientific papers.
Fallback model: all-MiniLM-L6-v2 — general-purpose, already in project deps.

Implements a swappable Protocol interface so downstream consumers remain
model-agnostic and future replacements are non-breaking.
"""

from __future__ import annotations

from typing import Protocol, List

import numpy as np
from loguru import logger

from config import EMBEDDING_MODEL_NAME, EMBEDDING_FALLBACK_MODEL, EMBEDDING_BATCH_SIZE


class EmbeddingModelInterface(Protocol):
    """Swappable interface for embedding models."""

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of output embeddings."""
        ...

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into dense vectors. Returns shape (n, dimension)."""
        ...


class EmbeddingModel:
    """
    Domain-tuned embedding model with graceful fallback.

    Primary: SPECTER2 (allenai/specter2) — 768-dim, trained on scientific papers.
    Fallback: all-MiniLM-L6-v2 — 384-dim, general-purpose.

    Handles OOM during batch encoding by halving batch size and retrying.
    If OOM persists, falls back to single-text encoding.
    """

    PRIMARY_MODEL = EMBEDDING_MODEL_NAME
    FALLBACK_MODEL = EMBEDDING_FALLBACK_MODEL

    def __init__(self, model_name: str | None = None, batch_size: int = EMBEDDING_BATCH_SIZE):
        self._model = None
        self._model_name: str = ""
        self._batch_size = batch_size
        self._load(model_name)

    def _load(self, model_name: str | None = None) -> None:
        """Load the sentence-transformer model with fallback logic."""
        from sentence_transformers import SentenceTransformer

        primary = model_name or self.PRIMARY_MODEL
        try:
            logger.info(f"Loading embedding model: {primary}")
            self._model = SentenceTransformer(primary)
            self._model_name = primary
            logger.info(
                f"Loaded embedding model: {primary} "
                f"(dimension={self._model.get_sentence_embedding_dimension()})"
            )
        except Exception as e:
            logger.warning(
                f"Primary model '{primary}' failed to load: {e}. "
                f"Falling back to '{self.FALLBACK_MODEL}'."
            )
            try:
                self._model = SentenceTransformer(self.FALLBACK_MODEL)
                self._model_name = self.FALLBACK_MODEL
                logger.info(
                    f"Loaded fallback model: {self.FALLBACK_MODEL} "
                    f"(dimension={self._model.get_sentence_embedding_dimension()})"
                )
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Both primary model '{primary}' and fallback model "
                    f"'{self.FALLBACK_MODEL}' failed to load. "
                    f"Primary error: {e}. Fallback error: {fallback_err}"
                ) from fallback_err

    @property
    def dimension(self) -> int:
        """Returns embedding dimensionality (e.g. 768 for SPECTER2, 384 for MiniLM)."""
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Returns the name of the currently loaded model."""
        return self._model_name

    def encode(self, texts: List[str], batch_size: int | None = None) -> np.ndarray:
        """
        Encode texts into dense vectors.

        Returns shape (n, dimension) as float32 ndarray.
        Handles OOM by halving batch size and retrying.
        If OOM persists at batch_size=1, encodes one-at-a-time with explicit cleanup.

        Parameters
        ----------
        texts : List[str]
            Texts to encode.
        batch_size : int | None
            Override batch size. Defaults to instance batch_size.

        Returns
        -------
        np.ndarray
            Shape (len(texts), dimension). Values are finite floats.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        effective_batch_size = batch_size if batch_size is not None else self._batch_size
        embeddings = self._encode_with_oom_retry(texts, effective_batch_size)

        # Validate: discard rows with NaN or Inf, log warning
        valid_mask = np.isfinite(embeddings).all(axis=1)
        if not valid_mask.all():
            n_invalid = int((~valid_mask).sum())
            logger.warning(
                f"Discarding {n_invalid}/{len(texts)} embeddings containing NaN/Inf values."
            )
            embeddings[~valid_mask] = 0.0  # Zero out invalid rows

        return embeddings.astype(np.float32)

    def _encode_with_oom_retry(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Attempt encoding with OOM retry strategy:
        1. Try with given batch_size
        2. On OOM, halve batch_size and retry
        3. If batch_size reaches 1 and still OOM, encode one-at-a-time with cleanup
        """
        import gc

        current_batch_size = batch_size

        while current_batch_size >= 1:
            try:
                result = self._model.encode(
                    texts,
                    batch_size=current_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return np.asarray(result)
            except (RuntimeError, MemoryError) as e:
                error_str = str(e).lower()
                is_oom = (
                    isinstance(e, MemoryError)
                    or "out of memory" in error_str
                    or "cuda out of memory" in error_str
                    or "oom" in error_str
                )
                if not is_oom:
                    raise

                if current_batch_size == 1:
                    # Already at minimum — try one-at-a-time with GC
                    logger.warning(
                        "OOM at batch_size=1. Encoding texts one-at-a-time with GC."
                    )
                    break

                new_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    f"OOM during encoding (batch_size={current_batch_size}). "
                    f"Halving to {new_batch_size} and retrying."
                )
                gc.collect()
                current_batch_size = new_batch_size

        # Last resort: encode one text at a time
        gc.collect()
        results = []
        for i, text in enumerate(texts):
            try:
                vec = self._model.encode(
                    [text],
                    batch_size=1,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                results.append(np.asarray(vec).flatten())
            except (RuntimeError, MemoryError):
                logger.error(f"OOM encoding text {i} even individually. Returning zeros.")
                results.append(np.zeros(self.dimension, dtype=np.float32))
            gc.collect()

        return np.vstack(results)

    def encode_paper(self, title: str, abstract: str | None) -> np.ndarray:
        """
        Convenience: encode a single paper's title + abstract.

        Combines title and abstract with a separator for better semantic encoding.
        Returns shape (dimension,) — a single 1-D vector.

        Parameters
        ----------
        title : str
            Paper title.
        abstract : str | None
            Paper abstract. If None or empty, only the title is encoded.

        Returns
        -------
        np.ndarray
            Shape (dimension,). Dense embedding vector.
        """
        if abstract:
            text = f"{title} [SEP] {abstract}"
        else:
            text = title

        # encode returns (1, dimension), squeeze to (dimension,)
        embedding = self.encode([text])
        return embedding[0]
