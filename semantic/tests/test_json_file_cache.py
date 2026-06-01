"""
Unit tests for semantic/_cache.py — _JsonFileCache.

Test cases:
1. Missing file returns {}
2. Invalid JSON returns {}
3. Save then load round-trip (data written can be read back correctly)
4. Tmp file is cleaned up after atomic write (no .tmp file left behind)

Requirements: 6.3, 6.4, 7.4
"""

import json
from pathlib import Path

import pytest

from semantic._cache import _JsonFileCache


# ---------------------------------------------------------------------------
# Test 1: Missing file returns {}
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_empty_dict(tmp_path):
    """load() on a non-existent path must return {} without raising."""
    cache = _JsonFileCache(tmp_path / "nonexistent.json")
    result = cache.load()
    assert result == {}


# ---------------------------------------------------------------------------
# Test 2: Invalid JSON returns {}
# ---------------------------------------------------------------------------

def test_load_invalid_json_returns_empty_dict(tmp_path):
    """load() on a file containing invalid JSON must return {} without raising."""
    cache_file = tmp_path / "bad.json"
    cache_file.write_text("this is not valid json {{{", encoding="utf-8")

    cache = _JsonFileCache(cache_file)
    result = cache.load()
    assert result == {}


def test_load_non_dict_json_returns_empty_dict(tmp_path):
    """load() on a file containing valid JSON that is not a dict (e.g. a list) returns {}."""
    cache_file = tmp_path / "list.json"
    cache_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    cache = _JsonFileCache(cache_file)
    result = cache.load()
    assert result == {}


# ---------------------------------------------------------------------------
# Test 3: Save then load round-trip
# ---------------------------------------------------------------------------

def test_save_then_load_round_trip(tmp_path):
    """Data written via save() can be read back correctly via load()."""
    cache_file = tmp_path / "cache.json"
    cache = _JsonFileCache(cache_file)

    data = {
        "abc123": {"canonical": "Lactobacillus reuteri", "ontology": "NCBI:1598"},
        "def456": {"canonical": "gut barrier", "ontology": "unknown"},
    }

    cache.save(data)
    loaded = cache.load()

    assert loaded == data


def test_save_overwrites_existing_data(tmp_path):
    """Calling save() twice replaces the previous content."""
    cache_file = tmp_path / "cache.json"
    cache = _JsonFileCache(cache_file)

    cache.save({"key1": "value1"})
    cache.save({"key2": "value2"})

    loaded = cache.load()
    assert loaded == {"key2": "value2"}


def test_save_empty_dict_round_trip(tmp_path):
    """Saving an empty dict and loading it back returns {}."""
    cache_file = tmp_path / "empty.json"
    cache = _JsonFileCache(cache_file)

    cache.save({})
    loaded = cache.load()

    assert loaded == {}


# ---------------------------------------------------------------------------
# Test 4: Tmp file is cleaned up after atomic write
# ---------------------------------------------------------------------------

def test_tmp_file_cleaned_up_after_save(tmp_path):
    """After save(), the .tmp file must not exist — only the target file remains."""
    cache_file = tmp_path / "cache.json"
    tmp_file = cache_file.with_suffix(".tmp")

    cache = _JsonFileCache(cache_file)
    cache.save({"hello": "world"})

    assert cache_file.exists(), "Target cache file should exist after save()"
    assert not tmp_file.exists(), ".tmp file should be removed after atomic rename"


def test_target_file_contains_valid_json_after_save(tmp_path):
    """The target file written by save() must contain valid, parseable JSON."""
    cache_file = tmp_path / "cache.json"
    cache = _JsonFileCache(cache_file)

    data = {"entity": "Bacteroides fragilis", "score": 0.95}
    cache.save(data)

    raw = cache_file.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed == data


# ---------------------------------------------------------------------------
# Test 5: Parent directory is created automatically
# ---------------------------------------------------------------------------

def test_save_creates_parent_directory(tmp_path):
    """save() must create missing parent directories rather than raising."""
    nested_file = tmp_path / "a" / "b" / "cache.json"
    cache = _JsonFileCache(nested_file)

    cache.save({"nested": True})

    assert nested_file.exists()
    assert cache.load() == {"nested": True}
