"""
semantic/_cache.py — Shared JSON file cache helper.

Provides _JsonFileCache, used by both LLMExtractor and LLMGrounder to persist
results to disk with atomic write semantics (write-to-tmp then os.replace).
"""

import json
import os
from pathlib import Path


class _JsonFileCache:
    """
    A simple JSON file-backed dict cache with atomic write support.

    - load() returns {} if the file is missing or contains invalid JSON; never raises.
    - save() writes atomically: serializes to a .tmp file in the same directory,
      then calls os.replace() to move it into place, ensuring the cache file is
      never left in a partially-written or corrupted state.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        """
        Load JSON from self.path.

        Returns {} if the file does not exist or contains invalid JSON.
        Never raises.
        """
        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def save(self, data: dict) -> None:
        """
        Atomically write *data* to self.path.

        Serializes to a .tmp file in the same directory, then calls os.replace()
        to rename it to the target path. This guarantees the cache file is never
        partially written, even if the process is interrupted mid-write.
        """
        # Ensure the parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on POSIX and Windows (same filesystem)
