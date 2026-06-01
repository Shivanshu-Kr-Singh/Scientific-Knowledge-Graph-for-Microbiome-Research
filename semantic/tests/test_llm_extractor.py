"""
Tests for LLMExtractor in semantic/llm_extractor.py.

Property tests (Task 5.2):
  - Property 1: Extraction always returns tuple[list, list]
    Validates: Requirements 12.1, 4.1
  - Property 2: Every extracted entity has non-whitespace name and entity_type
    Validates: Requirements 12.2, 3.4
  - Property 3: Every extracted relation has confidence in [0.0, 1.0]
    Validates: Requirements 12.3, 3.5
  - Property 4: Extraction is idempotent (cache round-trip)
    Validates: Requirements 12.4, 7.5, 4.3
  - Property 5: Extraction schema JSON round-trip preserves all fields
    Validates: Requirements 12.5
  - Property 11: Whitespace-only and empty text returns empty lists without LLM call
    Validates: Requirements 4.2

Unit tests (Task 5.3):
  - empty text → ([], [])
  - cache hit skips Ollama call
  - Ollama error + fallback=True calls Gemini
  - Ollama error + fallback=False returns ([], [])
  - markdown fence stripped with WARNING
  - Gemini exception returns ([], [])
  Requirements: 4.1–4.8, 7.1–7.6, 8.1, 8.3, 8.5, 8.7, 8.8
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from semantic.candidate_store import CandidateEntity, CandidateRelation

# ─── Hypothesis strategies ────────────────────────────────────────────────────

# Non-empty text with at least one non-whitespace character
non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

# Whitespace-only text (including empty string)
whitespace_text = st.text(alphabet=" \t\n\r", min_size=0)


@composite
def extraction_schema(draw):
    """Generate a valid extraction schema dict."""
    entities = draw(st.lists(st.fixed_dictionaries({
        "name": st.text(min_size=1),
        "type": st.text(min_size=1),
        "confidence": st.floats(0.0, 1.0, allow_nan=False),
        "novel": st.booleans(),
    })))
    relations = draw(st.lists(st.fixed_dictionaries({
        "subject": st.text(),
        "predicate": st.text(),
        "object": st.text(),
        "confidence": st.floats(0.0, 1.0, allow_nan=False),
    })))
    return {"entities": entities, "relations": relations, "evidence": {}}


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _make_ollama_config(fallback: bool = False):
    """Return a BackendConfig pointing to Ollama with no real server needed."""
    from config import BackendConfig
    return BackendConfig(
        llm_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_extraction_model="llama3",
        ollama_grounding_model="llama3",
        ollama_timeout_seconds=5,
        ollama_max_retries=0,
        ollama_retry_backoff_base=1.0,
        ollama_fallback_to_gemini=fallback,
        gemini_extraction_model="gemini-2.0-flash",
        gemini_grounding_model="gemini-2.5-flash",
    )


def _make_gemini_config():
    """Return a BackendConfig pointing directly to Gemini."""
    from config import BackendConfig
    return BackendConfig(
        llm_backend="gemini",
        ollama_base_url="http://localhost:11434",
        ollama_extraction_model="llama3",
        ollama_grounding_model="llama3",
        ollama_timeout_seconds=5,
        ollama_max_retries=0,
        ollama_retry_backoff_base=1.0,
        ollama_fallback_to_gemini=False,
        gemini_extraction_model="gemini-2.0-flash",
        gemini_grounding_model="gemini-2.5-flash",
    )


def _valid_extraction_json(entities=None, relations=None) -> str:
    """Return a valid extraction JSON string."""
    if entities is None:
        entities = [{"name": "Lactobacillus", "type": "taxon", "confidence": 0.9, "novel": False}]
    if relations is None:
        relations = [{"subject": "Lactobacillus", "predicate": "modulates",
                      "object": "gut barrier", "confidence": 0.85}]
    return json.dumps({"entities": entities, "relations": relations, "evidence": {}})


# ─── Property 1 ──────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 1: Extraction always returns tuple[list, list]

@given(text=non_empty_text)
@settings(max_examples=100)
def test_property1_extract_always_returns_tuple_of_lists(text):
    """
    # Feature: ollama-llm-integration, Property 1: Extraction always returns tuple[list, list]
    Validates: Requirements 12.1, 4.1

    For any non-empty text, extract() returns a 2-tuple of lists even when the
    Ollama backend raises an error.
    """
    from semantic.ollama_client import OllamaUnavailableError

    config = _make_ollama_config(fallback=False)

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.llm_extractor._cache.load", return_value={}), \
         patch("semantic.llm_extractor._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate",
               side_effect=OllamaUnavailableError("down", 1)):
        from semantic.llm_extractor import LLMExtractor
        extractor = LLMExtractor()
        result = extractor.extract(text)

    assert isinstance(result, tuple), "extract() must return a tuple"
    assert len(result) == 2, "extract() must return exactly 2 elements"
    assert isinstance(result[0], list), "first element must be a list"
    assert isinstance(result[1], list), "second element must be a list"


# ─── Property 2 ──────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 2: Every extracted entity has non-whitespace name and entity_type

@given(schema=extraction_schema())
@settings(max_examples=100)
def test_property2_entities_have_non_whitespace_name_and_type(schema):
    """
    # Feature: ollama-llm-integration, Property 2: Every extracted entity has non-whitespace name and entity_type
    Validates: Requirements 12.2, 3.4

    For any mocked LLM response that produces a non-empty entity list, every
    CandidateEntity has name and entity_type with at least one non-whitespace char.
    """
    # Only test schemas that have at least one entity with non-empty name/type
    entities_with_content = [
        e for e in schema["entities"]
        if e["name"].strip() and e["type"].strip()
    ]
    if not entities_with_content:
        return  # nothing to assert

    schema_with_content = {
        "entities": entities_with_content,
        "relations": schema["relations"],
        "evidence": schema["evidence"],
    }
    raw_response = json.dumps(schema_with_content)
    config = _make_ollama_config(fallback=False)

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.llm_extractor._cache.load", return_value={}), \
         patch("semantic.llm_extractor._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate", return_value=raw_response):
        from semantic.llm_extractor import LLMExtractor
        extractor = LLMExtractor()
        entities, _ = extractor.extract("some biomedical text")

    for entity in entities:
        assert isinstance(entity, CandidateEntity)
        assert entity.name.strip(), f"Entity name {entity.name!r} must be non-whitespace"
        assert entity.entity_type.strip(), f"entity_type {entity.entity_type!r} must be non-whitespace"


# ─── Property 3 ──────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 3: Every extracted relation has confidence in [0.0, 1.0]

@given(schema=extraction_schema())
@settings(max_examples=100)
def test_property3_relations_confidence_in_range(schema):
    """
    # Feature: ollama-llm-integration, Property 3: Every extracted relation has confidence in [0.0, 1.0]
    Validates: Requirements 12.3, 3.5

    For any mocked LLM response, every CandidateRelation has confidence in [0.0, 1.0].
    """
    raw_response = json.dumps(schema)
    config = _make_ollama_config(fallback=False)

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.llm_extractor._cache.load", return_value={}), \
         patch("semantic.llm_extractor._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate", return_value=raw_response):
        from semantic.llm_extractor import LLMExtractor
        extractor = LLMExtractor()
        _, relations = extractor.extract("some biomedical text")

    for rel in relations:
        assert isinstance(rel, CandidateRelation)
        assert 0.0 <= rel.confidence <= 1.0, (
            f"confidence {rel.confidence} is outside [0.0, 1.0]"
        )


# ─── Property 4 ──────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 4: Extraction is idempotent (cache round-trip)

@given(text=non_empty_text)
@settings(max_examples=100)
def test_property4_extraction_is_idempotent(text):
    """
    # Feature: ollama-llm-integration, Property 4: Extraction is idempotent (cache round-trip)
    Validates: Requirements 12.4, 7.5, 4.3

    Calling extract(text) twice returns the same number of entities and the same
    set of entity names (order-insensitive).
    """
    from semantic._cache import _JsonFileCache

    valid_response = _valid_extraction_json()
    config = _make_ollama_config(fallback=False)

    call_count = {"n": 0}

    def fake_generate(model, prompt):
        call_count["n"] += 1
        return valid_response

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "llm_extract_cache.json"
        real_cache = _JsonFileCache(cache_file)

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate", side_effect=fake_generate):
            from semantic.llm_extractor import LLMExtractor
            extractor = LLMExtractor()
            r1 = extractor.extract(text)
            r2 = extractor.extract(text)

    # Second call must use cache — LLM called only once
    assert call_count["n"] == 1, "LLM should only be called once; second call should use cache"
    assert len(r1[0]) == len(r2[0]), "Entity count must be identical on second call"
    assert {e.name for e in r1[0]} == {e.name for e in r2[0]}, "Entity names must match"
    assert len(r1[1]) == len(r2[1]), "Relation count must be identical on second call"


# ─── Property 5 ──────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 5: Extraction schema JSON round-trip preserves all fields

@given(schema=extraction_schema())
@settings(max_examples=100)
def test_property5_extraction_schema_json_roundtrip(schema):
    """
    # Feature: ollama-llm-integration, Property 5: Extraction schema JSON round-trip preserves all fields
    Validates: Requirements 12.5

    Serializing a valid extraction schema dict to JSON and parsing it back
    produces a dict with field-by-field equal values for all keys.
    """
    serialized = json.dumps(schema)
    parsed = json.loads(serialized)

    assert set(parsed.keys()) == set(schema.keys()), "Top-level keys must be preserved"
    assert len(parsed["entities"]) == len(schema["entities"])
    assert len(parsed["relations"]) == len(schema["relations"])

    for orig, roundtripped in zip(schema["entities"], parsed["entities"]):
        for field in ("name", "type", "novel"):
            assert roundtripped[field] == orig[field], f"Entity field {field!r} changed"
        # confidence: floats may differ slightly due to JSON serialization of special values
        assert roundtripped["confidence"] == orig["confidence"] or (
            str(roundtripped["confidence"]) == str(orig["confidence"])
        )

    for orig, roundtripped in zip(schema["relations"], parsed["relations"]):
        for field in ("subject", "predicate", "object"):
            assert roundtripped[field] == orig[field], f"Relation field {field!r} changed"


# ─── Property 11 ─────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 11: Whitespace-only and empty text returns empty lists without LLM call

@given(text=whitespace_text)
@settings(max_examples=100)
def test_property11_whitespace_text_returns_empty_without_llm_call(text):
    """
    # Feature: ollama-llm-integration, Property 11: Whitespace-only and empty text returns empty lists without LLM call
    Validates: Requirements 4.2

    For any string composed entirely of whitespace (including empty string),
    extract() returns ([], []) without invoking the Ollama client or Gemini.
    """
    config = _make_ollama_config(fallback=False)

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.ollama_client.OllamaClient.generate") as mock_ollama, \
         patch("semantic.llm_extractor._get_gemini_client") as mock_gemini:
        from semantic.llm_extractor import LLMExtractor
        extractor = LLMExtractor()
        result = extractor.extract(text)

    assert result == ([], []), f"Expected ([], []) for whitespace text, got {result}"
    mock_ollama.assert_not_called()
    mock_gemini.assert_not_called()


def test_property11_none_returns_empty_without_llm_call():
    """
    # Feature: ollama-llm-integration, Property 11: Whitespace-only and empty text returns empty lists without LLM call
    Validates: Requirements 4.2

    None input returns ([], []) without invoking any LLM backend.
    """
    config = _make_ollama_config(fallback=False)

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.ollama_client.OllamaClient.generate") as mock_ollama, \
         patch("semantic.llm_extractor._get_gemini_client") as mock_gemini:
        from semantic.llm_extractor import LLMExtractor
        extractor = LLMExtractor()
        result = extractor.extract(None)

    assert result == ([], [])
    mock_ollama.assert_not_called()
    mock_gemini.assert_not_called()


# ─── Unit Tests (Task 5.3) ────────────────────────────────────────────────────

class TestLLMExtractorUnit:
    """Unit tests for LLMExtractor. Requirements: 4.1–4.8, 7.1–7.6, 8.1, 8.3, 8.5, 8.7, 8.8"""

    # ── Test: empty text → ([], []) ───────────────────────────────────────────

    def test_empty_string_returns_empty(self):
        """Req 4.2: empty string returns ([], []) without LLM call."""
        config = _make_ollama_config()
        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("")
        assert result == ([], [])
        mock_gen.assert_not_called()

    def test_whitespace_only_returns_empty(self):
        """Req 4.2: whitespace-only string returns ([], []) without LLM call."""
        config = _make_ollama_config()
        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("   \t\n  ")
        assert result == ([], [])
        mock_gen.assert_not_called()

    def test_none_returns_empty(self):
        """Req 4.2: None returns ([], []) without LLM call."""
        config = _make_ollama_config()
        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract(None)
        assert result == ([], [])
        mock_gen.assert_not_called()


    # ── Test: cache hit skips Ollama call ─────────────────────────────────────

    def test_cache_hit_skips_ollama_call(self, tmp_path):
        """Req 4.3, 7.3: cache hit returns cached result without calling Ollama."""
        from semantic._cache import _JsonFileCache

        text = "Lactobacillus modulates gut barrier"
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached_data = {
            "entities": [{"name": "Lactobacillus", "type": "taxon", "confidence": 0.9, "novel": False}],
            "relations": [],
            "evidence": {},
        }

        cache_file = tmp_path / "llm_extract_cache.json"
        real_cache = _JsonFileCache(cache_file)
        real_cache.save({key: cached_data})

        config = _make_ollama_config()
        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_extractor import LLMExtractor
            entities, relations = LLMExtractor().extract(text)

        mock_gen.assert_not_called()
        assert len(entities) == 1
        assert entities[0].name == "Lactobacillus"
        assert entities[0].entity_type == "taxon"


    # ── Test: Ollama error + fallback=True calls Gemini ───────────────────────

    def test_ollama_unavailable_with_fallback_calls_gemini(self):
        """Req 4.5, 8.1: OllamaUnavailableError + fallback=True → Gemini is called."""
        from semantic.ollama_client import OllamaUnavailableError

        config = _make_ollama_config(fallback=True)

        mock_gemini_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = _valid_extraction_json()
        mock_gemini_client.models.generate_content.return_value = mock_resp

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaUnavailableError("down", 1)), \
             patch("semantic.llm_extractor._get_gemini_client",
                   return_value=mock_gemini_client):
            from semantic.llm_extractor import LLMExtractor
            entities, relations = LLMExtractor().extract("some text")

        mock_gemini_client.models.generate_content.assert_called_once()
        assert len(entities) == 1
        assert entities[0].name == "Lactobacillus"

    def test_ollama_timeout_with_fallback_calls_gemini(self):
        """Req 4.5, 8.1: OllamaTimeoutError + fallback=True → Gemini is called."""
        from semantic.ollama_client import OllamaTimeoutError

        config = _make_ollama_config(fallback=True)

        mock_gemini_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = _valid_extraction_json()
        mock_gemini_client.models.generate_content.return_value = mock_resp

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaTimeoutError(5)), \
             patch("semantic.llm_extractor._get_gemini_client",
                   return_value=mock_gemini_client):
            from semantic.llm_extractor import LLMExtractor
            entities, relations = LLMExtractor().extract("some text")

        mock_gemini_client.models.generate_content.assert_called_once()
        assert len(entities) == 1


    # ── Test: Ollama error + fallback=False returns ([], []) ──────────────────

    def test_ollama_unavailable_no_fallback_returns_empty(self):
        """Req 4.6: OllamaUnavailableError + fallback=False → ([], []), no Gemini call."""
        from semantic.ollama_client import OllamaUnavailableError

        config = _make_ollama_config(fallback=False)

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaUnavailableError("down", 1)), \
             patch("semantic.llm_extractor._get_gemini_client") as mock_gemini:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])
        mock_gemini.assert_not_called()

    def test_ollama_timeout_no_fallback_returns_empty(self):
        """Req 4.6: OllamaTimeoutError + fallback=False → ([], []), no Gemini call."""
        from semantic.ollama_client import OllamaTimeoutError

        config = _make_ollama_config(fallback=False)

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaTimeoutError(5)), \
             patch("semantic.llm_extractor._get_gemini_client") as mock_gemini:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])
        mock_gemini.assert_not_called()


    # ── Test: markdown fence stripped with WARNING ─────────────────────────────

    def test_markdown_fence_stripped_with_warning(self, caplog):
        """Req 3.3: markdown code fence is stripped and a WARNING is logged."""
        import logging

        fenced_response = "```json\n" + _valid_extraction_json() + "\n```"
        config = _make_ollama_config(fallback=False)

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=fenced_response), \
             caplog.at_level(logging.WARNING, logger="semantic.llm_extractor"):
            from semantic.llm_extractor import LLMExtractor
            entities, relations = LLMExtractor().extract("some text")

        # Should still parse successfully
        assert len(entities) == 1
        assert entities[0].name == "Lactobacillus"

        # WARNING must have been logged
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("fence" in m.lower() or "markdown" in m.lower() for m in warning_messages), (
            f"Expected a WARNING about markdown fence, got: {warning_messages}"
        )

    def test_plain_backtick_fence_stripped(self, caplog):
        """Req 3.3: plain ``` fence (without json) is also stripped."""
        import logging

        fenced_response = "```\n" + _valid_extraction_json() + "\n```"
        config = _make_ollama_config(fallback=False)

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=fenced_response), \
             caplog.at_level(logging.WARNING, logger="semantic.llm_extractor"):
            from semantic.llm_extractor import LLMExtractor
            entities, _ = LLMExtractor().extract("some text")

        assert len(entities) == 1


    # ── Test: Gemini exception returns ([], []) ────────────────────────────────

    def test_gemini_exception_returns_empty(self):
        """Req 4.8, 8.5: Gemini exception during extraction → ([], []), log ERROR."""
        from semantic.ollama_client import OllamaUnavailableError

        config = _make_ollama_config(fallback=True)

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content.side_effect = RuntimeError("Gemini API error")

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaUnavailableError("down", 1)), \
             patch("semantic.llm_extractor._get_gemini_client",
                   return_value=mock_gemini_client):
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])

    def test_gemini_direct_exception_returns_empty(self):
        """Req 8.8: LLM_BACKEND=gemini, Gemini raises → ([], [])."""
        config = _make_gemini_config()

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.llm_extractor._get_gemini_client",
                   return_value=mock_gemini_client):
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])


    # ── Test: invalid JSON response returns ([], []) ───────────────────────────

    def test_invalid_json_response_returns_empty(self):
        """Req 3.1, 4.4: invalid JSON from Ollama → ([], []), do NOT write cache."""
        config = _make_ollama_config(fallback=False)

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save") as mock_save, \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value="this is not json at all"):
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])
        mock_save.assert_not_called()

    # ── Test: successful extraction writes to cache ────────────────────────────

    def test_successful_extraction_writes_to_cache(self, tmp_path):
        """Req 4.4, 7.4: successful extraction writes result to cache atomically."""
        from semantic._cache import _JsonFileCache

        text = "Lactobacillus modulates gut barrier"
        cache_file = tmp_path / "llm_extract_cache.json"
        real_cache = _JsonFileCache(cache_file)

        config = _make_ollama_config(fallback=False)

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=_valid_extraction_json()):
            from semantic.llm_extractor import LLMExtractor
            entities, _ = LLMExtractor().extract(text)

        assert len(entities) == 1
        # Cache file must exist and contain the result
        assert cache_file.exists()
        saved = json.loads(cache_file.read_text())
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert key in saved

    # ── Test: LLM_BACKEND=gemini uses Gemini directly ─────────────────────────

    def test_gemini_backend_calls_gemini_directly(self):
        """Req 8.7, 14.3: LLM_BACKEND=gemini → Gemini called directly, no Ollama."""
        config = _make_gemini_config()

        mock_gemini_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = _valid_extraction_json()
        mock_gemini_client.models.generate_content.return_value = mock_resp

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.llm_extractor._get_gemini_client",
                   return_value=mock_gemini_client), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_ollama:
            from semantic.llm_extractor import LLMExtractor
            entities, _ = LLMExtractor().extract("some text")

        mock_ollama.assert_not_called()
        mock_gemini_client.models.generate_content.assert_called_once()
        assert len(entities) == 1

    # ── Test: fallback WARNING log contains "activating Gemini fallback" ──────

    def test_fallback_warning_contains_expected_phrase(self, caplog):
        """Req 4.5: WARNING log must include 'activating Gemini fallback'."""
        import logging
        from semantic.ollama_client import OllamaUnavailableError

        config = _make_ollama_config(fallback=True)

        mock_gemini_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = _valid_extraction_json()
        mock_gemini_client.models.generate_content.return_value = mock_resp

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaUnavailableError("down", 1)), \
             patch("semantic.llm_extractor._get_gemini_client",
                   return_value=mock_gemini_client), \
             caplog.at_level(logging.WARNING, logger="semantic.llm_extractor"):
            from semantic.llm_extractor import LLMExtractor
            LLMExtractor().extract("some text")

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("activating Gemini fallback" in m for m in warning_messages), (
            f"Expected 'activating Gemini fallback' in WARNING, got: {warning_messages}"
        )

    # ── Test: text truncated to 12,000 characters ─────────────────────────────

    def test_text_truncated_to_12000_chars(self):
        """Req 4.7, 11.5: input text is truncated to 12,000 chars before prompt construction."""
        config = _make_ollama_config(fallback=False)
        long_text = "x" * 20000

        captured_prompts = []

        def capture_generate(model, prompt):
            captured_prompts.append(prompt)
            return _valid_extraction_json()

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=capture_generate):
            from semantic.llm_extractor import LLMExtractor
            LLMExtractor().extract(long_text)

        assert len(captured_prompts) == 1
        # The prompt must contain at most 12,000 x's (the truncated text)
        assert "x" * 12001 not in captured_prompts[0], "Text was not truncated to 12,000 chars"
        assert "x" * 12000 in captured_prompts[0], "Truncated text should be 12,000 chars"
