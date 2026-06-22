"""
Tests for LLMExtractor in semantic/llm_extractor.py.

Property tests:
  - Property 1: Extraction always returns tuple[list, list]
  - Property 2: Every extracted entity has non-whitespace name and entity_type
  - Property 3: Every extracted relation has confidence in [0.0, 1.0]
  - Property 4: Extraction is idempotent (cache round-trip)
  - Property 5: Extraction schema JSON round-trip preserves all fields
  - Property 11: Whitespace-only and empty text returns empty lists without LLM call

Unit tests:
  - empty text → ([], [])
  - cache hit skips Ollama call
  - Ollama error → ([], [])
  - markdown fence stripped with WARNING
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from semantic.candidate_store import CandidateEntity, CandidateRelation


# ─── Hypothesis strategies ────────────────────────────────────────────────────

non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())
whitespace_text = st.text(alphabet=" \t\n\r", min_size=0)


@composite
def extraction_schema(draw):
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_ollama_config():
    from config import BackendConfig
    return BackendConfig(
        llm_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_extraction_model="llama3",
        ollama_grounding_model="llama3",
        ollama_timeout_seconds=5,
        ollama_max_retries=0,
        ollama_retry_backoff_base=1.0,
    )


def _valid_extraction_json(entities=None, relations=None) -> str:
    if entities is None:
        entities = [{"name": "Lactobacillus", "type": "taxon", "confidence": 0.9, "novel": False}]
    if relations is None:
        relations = [{"subject": "Lactobacillus", "predicate": "modulates",
                      "object": "gut barrier", "confidence": 0.85}]
    return json.dumps({"entities": entities, "relations": relations, "evidence": {}})


# ─── Property 1 ──────────────────────────────────────────────────────────────

@given(text=non_empty_text)
@settings(max_examples=100)
def test_property1_extract_always_returns_tuple_of_lists(text):
    """extract() always returns a 2-tuple of lists, even on Ollama error."""
    from semantic.ollama_client import OllamaUnavailableError

    config = _make_ollama_config()

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.llm_extractor._cache.load", return_value={}), \
         patch("semantic.llm_extractor._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate",
               side_effect=OllamaUnavailableError("down", 1)):
        from semantic.llm_extractor import LLMExtractor
        result = LLMExtractor().extract(text)

    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], list) and isinstance(result[1], list)


# ─── Property 2 ──────────────────────────────────────────────────────────────

@given(schema=extraction_schema())
@settings(max_examples=100)
def test_property2_entities_have_non_whitespace_name_and_type(schema):
    """Every CandidateEntity returned has non-whitespace name and entity_type."""
    entities_with_content = [
        e for e in schema["entities"]
        if e["name"].strip() and e["type"].strip()
    ]
    if not entities_with_content:
        return

    schema_with_content = {
        "entities": entities_with_content,
        "relations": schema["relations"],
        "evidence": schema["evidence"],
    }
    config = _make_ollama_config()

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.llm_extractor._cache.load", return_value={}), \
         patch("semantic.llm_extractor._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate",
               return_value=json.dumps(schema_with_content)):
        from semantic.llm_extractor import LLMExtractor
        entities, _ = LLMExtractor().extract("some biomedical text")

    for entity in entities:
        assert isinstance(entity, CandidateEntity)
        assert entity.name.strip()
        assert entity.entity_type.strip()


# ─── Property 3 ──────────────────────────────────────────────────────────────

@given(schema=extraction_schema())
@settings(max_examples=100)
def test_property3_relations_confidence_in_range(schema):
    """Every CandidateRelation has confidence in [0.0, 1.0]."""
    config = _make_ollama_config()

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.llm_extractor._cache.load", return_value={}), \
         patch("semantic.llm_extractor._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate",
               return_value=json.dumps(schema)):
        from semantic.llm_extractor import LLMExtractor
        _, relations = LLMExtractor().extract("some biomedical text")

    for rel in relations:
        assert isinstance(rel, CandidateRelation)
        assert 0.0 <= rel.confidence <= 1.0


# ─── Property 4 ──────────────────────────────────────────────────────────────

@given(text=non_empty_text)
@settings(max_examples=100)
def test_property4_extraction_is_idempotent(text):
    """Calling extract(text) twice uses cache on the second call."""
    from semantic._cache import _JsonFileCache

    call_count = {"n": 0}

    def fake_generate(model, prompt):
        call_count["n"] += 1
        return _valid_extraction_json()

    config = _make_ollama_config()

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

    assert call_count["n"] == 1, "LLM should be called only once; second call uses cache"
    assert len(r1[0]) == len(r2[0])
    assert {e.name for e in r1[0]} == {e.name for e in r2[0]}


# ─── Property 5 ──────────────────────────────────────────────────────────────

@given(schema=extraction_schema())
@settings(max_examples=100)
def test_property5_extraction_schema_json_roundtrip(schema):
    """JSON round-trip of extraction schema preserves all fields."""
    parsed = json.loads(json.dumps(schema))

    assert set(parsed.keys()) == set(schema.keys())
    assert len(parsed["entities"]) == len(schema["entities"])
    assert len(parsed["relations"]) == len(schema["relations"])

    for orig, rt in zip(schema["entities"], parsed["entities"]):
        for field in ("name", "type", "novel"):
            assert rt[field] == orig[field]

    for orig, rt in zip(schema["relations"], parsed["relations"]):
        for field in ("subject", "predicate", "object"):
            assert rt[field] == orig[field]


# ─── Property 11 ─────────────────────────────────────────────────────────────

@given(text=whitespace_text)
@settings(max_examples=100)
def test_property11_whitespace_text_returns_empty_without_llm_call(text):
    """Whitespace-only/empty text returns ([], []) without any LLM call."""
    config = _make_ollama_config()

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.ollama_client.OllamaClient.generate") as mock_ollama:
        from semantic.llm_extractor import LLMExtractor
        result = LLMExtractor().extract(text)

    assert result == ([], [])
    mock_ollama.assert_not_called()


def test_property11_none_returns_empty_without_llm_call():
    """None input returns ([], []) without any LLM call."""
    config = _make_ollama_config()

    with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
         patch("semantic.ollama_client.OllamaClient.generate") as mock_ollama:
        from semantic.llm_extractor import LLMExtractor
        result = LLMExtractor().extract(None)

    assert result == ([], [])
    mock_ollama.assert_not_called()


# ─── Unit Tests ──────────────────────────────────────────────────────────────

class TestLLMExtractorUnit:

    def test_empty_string_returns_empty(self):
        config = _make_ollama_config()
        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("")
        assert result == ([], [])
        mock_gen.assert_not_called()

    def test_whitespace_only_returns_empty(self):
        config = _make_ollama_config()
        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("   \t\n  ")
        assert result == ([], [])
        mock_gen.assert_not_called()

    def test_none_returns_empty(self):
        config = _make_ollama_config()
        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract(None)
        assert result == ([], [])
        mock_gen.assert_not_called()

    def test_cache_hit_skips_ollama_call(self, tmp_path):
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

    def test_ollama_unavailable_returns_empty(self):
        """OllamaUnavailableError → ([], [])."""
        from semantic.ollama_client import OllamaUnavailableError

        config = _make_ollama_config()

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaUnavailableError("down", 1)):
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])

    def test_ollama_timeout_returns_empty(self):
        """OllamaTimeoutError → ([], [])."""
        from semantic.ollama_client import OllamaTimeoutError

        config = _make_ollama_config()

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaTimeoutError(5)):
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])

    def test_markdown_fence_stripped_with_warning(self, caplog):
        """Markdown code fence is stripped and a WARNING is logged."""
        import logging

        fenced_response = "```json\n" + _valid_extraction_json() + "\n```"
        config = _make_ollama_config()

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=fenced_response), \
             caplog.at_level(logging.WARNING, logger="semantic.llm_extractor"):
            from semantic.llm_extractor import LLMExtractor
            entities, _ = LLMExtractor().extract("some text")

        assert len(entities) == 1
        assert entities[0].name == "Lactobacillus"
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("fence" in m.lower() or "markdown" in m.lower() for m in warning_messages)

    def test_plain_backtick_fence_stripped(self, caplog):
        """Plain ``` fence (without json specifier) is also stripped."""
        import logging

        fenced_response = "```\n" + _valid_extraction_json() + "\n```"
        config = _make_ollama_config()

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=fenced_response), \
             caplog.at_level(logging.WARNING, logger="semantic.llm_extractor"):
            from semantic.llm_extractor import LLMExtractor
            entities, _ = LLMExtractor().extract("some text")

        assert len(entities) == 1

    def test_invalid_json_response_returns_empty_and_no_cache_write(self):
        """Invalid JSON from Ollama → ([], []) and cache is NOT written."""
        config = _make_ollama_config()

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache.load", return_value={}), \
             patch("semantic.llm_extractor._cache.save") as mock_save, \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value="this is not json at all"):
            from semantic.llm_extractor import LLMExtractor
            result = LLMExtractor().extract("some text")

        assert result == ([], [])
        mock_save.assert_not_called()

    def test_successful_extraction_writes_to_cache(self, tmp_path):
        """Successful extraction writes result to cache."""
        from semantic._cache import _JsonFileCache

        text = "Lactobacillus modulates gut barrier"
        cache_file = tmp_path / "llm_extract_cache.json"
        real_cache = _JsonFileCache(cache_file)

        config = _make_ollama_config()

        with patch("semantic.llm_extractor.BACKEND_CONFIG", config), \
             patch("semantic.llm_extractor._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=_valid_extraction_json()):
            from semantic.llm_extractor import LLMExtractor
            entities, _ = LLMExtractor().extract(text)

        assert len(entities) == 1
        assert cache_file.exists()
        saved = json.loads(cache_file.read_text())
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert key in saved
