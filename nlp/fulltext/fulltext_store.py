"""
nlp/fulltext/fulltext_store.py
--------------------------------
Stores and retrieves full-text content outside of enriched JSON files.

WHY THIS EXISTS:
  Full text can be 10–100KB per paper. Storing it inline in
  enriched_batch_NNN.json causes those files to balloon to several GB
  at 60K+ papers, making them slow to load and memory-intensive.

  This module writes full text to individual files:
    data/fulltext/{content_hash}.txt

  The enriched record stores only the path (fulltext_path field).
  When Layer 3 needs the text, it calls FullTextStore.load(path).

DESIGN:
  - Content-hash keyed — same paper never stored twice
  - Plain UTF-8 text files — readable, grep-able, no format overhead
  - load() returns empty string if file missing — graceful degradation
  - Thread-safe writes (each file is independent)
"""

import os
from pathlib import Path
from loguru import logger

from config import DATA_DIR

FULLTEXT_DIR = DATA_DIR / "fulltext"
FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)


class FullTextStore:
    """Saves and loads full-text content keyed by content_hash."""

    def __init__(self, base_dir: Path = FULLTEXT_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, content_hash: str, text: str) -> str:
        """
        Saves full text to disk. Returns the relative path string.
        Skips write if file already exists (idempotent).

        Returns:
            Relative path like "fulltext/abc123.txt"
        """
        if not content_hash or not text or not text.strip():
            return ""

        # Use first 2 chars as subdirectory to avoid huge flat directories
        # at 60K+ files: fulltext/ab/abcdef123.txt
        subdir = self.base_dir / content_hash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{content_hash}.txt"

        if not path.exists():
            try:
                path.write_text(text, encoding="utf-8")
            except Exception as e:
                logger.warning(f"[fulltext_store] Failed to save {content_hash}: {e}")
                return ""

        # Return path relative to DATA_DIR so it's portable
        try:
            return str(path.relative_to(DATA_DIR))
        except ValueError:
            return str(path)

    def load(self, relative_path: str) -> str:
        """
        Loads full text from disk. Returns empty string if missing.

        Args:
            relative_path: as stored in enriched record's fulltext_path field
        """
        if not relative_path:
            return ""

        # Try as relative to DATA_DIR first
        path = DATA_DIR / relative_path
        if not path.exists():
            # Try as absolute path (legacy)
            path = Path(relative_path)

        if not path.exists():
            return ""

        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"[fulltext_store] Failed to load {relative_path}: {e}")
            return ""

    def exists(self, content_hash: str) -> bool:
        """Returns True if full text is already stored for this hash."""
        if not content_hash:
            return False
        path = self.base_dir / content_hash[:2] / f"{content_hash}.txt"
        return path.exists()

    def stats(self) -> dict:
        """Returns storage statistics."""
        try:
            files = list(self.base_dir.rglob("*.txt"))
            total_bytes = sum(f.stat().st_size for f in files)
            return {
                "total_files": len(files),
                "total_mb": round(total_bytes / 1_048_576, 1),
                "base_dir": str(self.base_dir),
            }
        except Exception:
            return {"total_files": 0, "total_mb": 0}


# Module-level singleton — reused across pipeline runs
_store: FullTextStore = None


def get_store() -> FullTextStore:
    """Returns the module-level FullTextStore singleton."""
    global _store
    if _store is None:
        _store = FullTextStore()
    return _store
