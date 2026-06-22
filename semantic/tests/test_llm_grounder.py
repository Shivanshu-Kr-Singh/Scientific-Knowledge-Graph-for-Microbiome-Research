"""
Tests for LLMGrounder in semantic/llm_grounder.py.

Property tests (Task 6.2):
  - Property 6: Grounding always returns dict with exactly "canonical" and "ontology"
    Validates: Requirements 13.1, 5.1
  - Property 7: Grounding is idempotent (cache round-trip)
    Validates: Requirements 13.3, 6.6, 5.2
  - Property 8: Grounding canonical is always a non-empty string
    Validates: Requirements 13.2, 3.6
  - Property 9: Grounding schema JSON round-trip preserves all fields
    Validates: Requirements 13.4
  - Property 10: Invalid LLM grounding response falls back to entity.name
    Validates: Requirements 13.5, 3.2
  - Property 15: Grounding cache key is MD5 of name concatenated with entity_type
    Validates: Requirements 6.2

Unit tests (Task 6.3):
  - Cache hit skips Ollama call
  - Empty canonical substituted with entity.name
  - Ollama error returns fallback dict; logs ERROR
  - Invalid JSON returns fallback dict; logs ERROR
  - Successful Ollama response is cached (keyed by MD5 of name + entity_type)
  - Missing "ontology" key returns fallback dict
  Requirements: 5.1-5.6, 6.1-6.7
"""

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from semantic.candidate_store import CandidateEntity

# --- Hypothesis strategies ---------------------------------------------------

# Non-empty text with at least one non-whitespace character
non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())


@composite
def candidate_entity(draw):
    """Generate an arbitrary CandidateEntity with non-empty name and entity_type."""
    name = draw(st.text(min_size=1).filter(lambda s: s.strip()))
    entity_type = draw(st.text(min_size=1).filter(lambda s: s.strip()))
    return CandidateEntity(name=name, entity_type=entity_type)


@composite
def grounding_schema(draw):
    """Generate a valid grounding schema dict."""
    canonical = draw(st.text(min_size=1).filter(lambda s: s.strip()))
    ontology = draw(st.text())
    return {"canonical": canonical, "ontology": ontology}


def _is_valid_grounding_json(s: str) -> bool:
    """Return True if s is valid JSON with string 'canonical' and 'ontology' keys."""
    try:
        d = json.loads(s)
        return (
            isinstance(d, dict)
            and isinstance(d.get("canonical"), str)
            and isinstance(d.get("ontology"), str)
            and d["canonical"].strip() != ""
        )
    except Exception:
        return False


# --- Shared helpers ----------------------------------------------------------


def _make_ollama_config():
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
    )


def _make_entity(name: str = "Lactobacillus reuteri", entity_type: str = "taxon") -> CandidateEntity:
    """Create a simple CandidateEntity for testing."""
    return CandidateEntity(name=name, entity_type=entity_type)


def _valid_grounding_json(canonical: str = "Lactobacillus reuteri DSM 17938",
                          ontology: str = "NCBI:1598") -> str:
    """Return a valid grounding JSON string."""
    return json.dumps({"canonical": canonical, "ontology": ontology})


def _cache_key(entity: CandidateEntity) -> str:
    """Compute the expected MD5 cache key for an entity."""
    return hashlib.md5((entity.name + entity.entity_type).encode("utf-8")).hexdigest()


# === Property 6 ==============================================================
# Feature: ollama-llm-integration, Property 6: Grounding always returns dict with exactly "canonical" and "ontology"

@given(entity=candidate_entity())
@settings(max_examples=100, deadline=None)
def test_property6_resolve_always_returns_canonical_and_ontology(entity):
    """
    # Feature: ollama-llm-integration, Property 6: Grounding always returns dict with exactly "canonical" and "ontology"
    Validates: Requirements 13.1, 5.1

    For any CandidateEntity, resolve() returns a dict with exactly the keys
    "canonical" and "ontology" (both strings) even when Ollama raises an error.
    """
    from semantic.ollama_client import OllamaUnavailableError

    config = _make_ollama_config()

    with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
         patch("semantic.llm_grounder._cache.load", return_value={}), \
         patch("semantic.llm_grounder._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate",
               side_effect=OllamaUnavailableError("down", 1)):
        from semantic.llm_grounder import LLMGrounder
        result = LLMGrounder().resolve(entity)

    assert isinstance(result, dict), "resolve() must return a dict"
    assert set(result.keys()) == {"canonical", "ontology"}, (
        f"resolve() must return exactly keys 'canonical' and 'ontology', got {set(result.keys())}"
    )
    assert isinstance(result["canonical"], str), "'canonical' must be a string"
    assert isinstance(result["ontology"], str), "'ontology' must be a string"


# === Property 7 ==============================================================
# Feature: ollama-llm-integration, Property 7: Grounding is idempotent (cache round-trip)

@given(entity=candidate_entity())
@settings(max_examples=100)
def test_property7_grounding_is_idempotent(entity):
    """
    # Feature: ollama-llm-integration, Property 7: Grounding is idempotent (cache round-trip)
    Validates: Requirements 13.3, 6.6, 5.2

    Calling resolve(entity) twice returns identical "canonical" and "ontology"
    values. The LLM is only called once; the second call uses the cache.
    """
    from semantic._cache import _JsonFileCache

    valid_response = _valid_grounding_json()
    config = _make_ollama_config()

    call_count = {"n": 0}

    def fake_generate(model, prompt):
        call_count["n"] += 1
        return valid_response

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "llm_ground_cache.json"
        real_cache = _JsonFileCache(cache_file)

        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate", side_effect=fake_generate):
            from semantic.llm_grounder import LLMGrounder
            grounder = LLMGrounder()
            r1 = grounder.resolve(entity)
            r2 = grounder.resolve(entity)

    assert call_count["n"] == 1, "LLM should only be called once; second call should use cache"
    assert r1["canonical"] == r2["canonical"], "canonical must be identical on second call"
    assert r1["ontology"] == r2["ontology"], "ontology must be identical on second call"


# === Property 8 ==============================================================
# Feature: ollama-llm-integration, Property 8: Grounding canonical is always a non-empty string

@given(entity=candidate_entity())
@settings(max_examples=100)
def test_property8_canonical_is_always_non_empty(entity):
    """
    # Feature: ollama-llm-integration, Property 8: Grounding canonical is always a non-empty string
    Validates: Requirements 13.2, 3.6

    For any CandidateEntity with a non-empty name, the "canonical" value in the
    returned dict is always a non-empty string. When Ollama raises an error, the
    system substitutes entity.name.
    """
    from semantic.ollama_client import OllamaUnavailableError

    config = _make_ollama_config()

    with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
         patch("semantic.llm_grounder._cache.load", return_value={}), \
         patch("semantic.llm_grounder._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate",
               side_effect=OllamaUnavailableError("down", 1)):
        from semantic.llm_grounder import LLMGrounder
        result = LLMGrounder().resolve(entity)

    assert result["canonical"].strip(), (
        f"'canonical' must be non-empty, got {result['canonical']!r} for entity {entity.name!r}"
    )


@given(entity=candidate_entity())
@settings(max_examples=100)
def test_property8_empty_canonical_substituted_with_entity_name(entity):
    """
    # Feature: ollama-llm-integration, Property 8: Grounding canonical is always a non-empty string
    Validates: Requirements 13.2, 3.6

    When the LLM returns an empty canonical string, entity.name is substituted.
    """
    empty_canonical_response = json.dumps({"canonical": "", "ontology": "NCBI:1"})
    config = _make_ollama_config()

    with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
         patch("semantic.llm_grounder._cache.load", return_value={}), \
         patch("semantic.llm_grounder._cache.save"), \
         patch("semantic.ollama_client.OllamaClient.generate",
               return_value=empty_canonical_response):
        from semantic.llm_grounder import LLMGrounder
        result = LLMGrounder().resolve(entity)

    assert result["canonical"] == entity.name, (
        f"Empty canonical should be substituted with entity.name={entity.name!r}, "
        f"got {result['canonical']!r}"
    )
    assert result["canonical"].strip(), "'canonical' must be non-empty after substitution"


# === Property 9 ==============================================================
# Feature: ollama-llm-integration, Property 9: Grounding schema JSON round-trip preserves all fields

@given(schema=grounding_schema())
@settings(max_examples=100)
def test_property9_grounding_schema_json_roundtrip(schema):
    """
    # Feature: ollama-llm-integration, Property 9: Grounding schema JSON round-trip preserves all fields
    Validates: Requirements 13.4

    Serializing a valid grounding schema dict to JSON and parsing it back
    produces a dict with field-by-field equal values for both keys.
    """
    serialized = json.dumps(schema)
    parsed = json.loads(serialized)

    assert set(parsed.keys()) == set(schema.keys()), (
        f"Keys must be preserved after round-trip: expected {set(schema.keys())}, got {set(parsed.keys())}"
    )
    assert parsed["canonical"] == schema["canonical"], (
        f"'canonical' changed after round-trip: {schema['canonical']!r} -> {parsed['canonical']!r}"
    )
    assert parsed["ontology"] == schema["ontology"], (
        f"'ontology' changed after round-trip: {schema['ontology']!r} -> {parsed['ontology']!r}"
    )


# === Property 10 =============================================================
# Feature: ollama-llm-integration, Property 10: Invalid LLM grounding response falls back to entity.name

@given(entity=candidate_entity())
@settings(max_examples=100)
def test_property10_invalid_json_falls_back_to_entity_name(entity):
    """
    # Feature: ollama-llm-integration, Property 10: Invalid LLM grounding response falls back to entity.name
    Validates: Requirements 13.5, 3.2

    When the LLM returns a response that is not valid JSON, resolve() returns
    {"canonical": entity.name, "ontology": "unknown"}.
    """
    config = _make_ollama_config()

    with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
         patch("semantic.llm_grounder._cache.load", return_value={}), \
         patch("semantic.llm_grounder._cache.save") as mock_save, \
         patch("semantic.ollama_client.OllamaClient.generate",
               return_value="this is not valid json"):
        from semantic.llm_grounder import LLMGrounder
        result = LLMGrounder().resolve(entity)

    assert result == {"canonical": entity.name, "ontology": "unknown"}, (
        f"Expected fallback dict, got {result!r}"
    )
    mock_save.assert_not_called()


@given(entity=candidate_entity())
@settings(max_examples=100)
def test_property10_missing_canonical_key_falls_back_to_entity_name(entity):
    """
    # Feature: ollama-llm-integration, Property 10: Invalid LLM grounding response falls back to entity.name
    Validates: Requirements 13.5, 3.2

    When the LLM returns valid JSON but missing the "canonical" key, resolve()
    returns {"canonical": entity.name, "ontology": "unknown"}.
    """
    config = _make_ollama_config()
    missing_canonical = json.dumps({"ontology": "NCBI:1598"})

    with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
         patch("semantic.llm_grounder._cache.load", return_value={}), \
         patch("semantic.llm_grounder._cache.save") as mock_save, \
         patch("semantic.ollama_client.OllamaClient.generate",
               return_value=missing_canonical):
        from semantic.llm_grounder import LLMGrounder
        result = LLMGrounder().resolve(entity)

    assert result == {"canonical": entity.name, "ontology": "unknown"}, (
        f"Expected fallback dict, got {result!r}"
    )
    mock_save.assert_not_called()


@given(entity=candidate_entity())
@settings(max_examples=100)
def test_property10_non_string_canonical_falls_back_to_entity_name(entity):
    """
    # Feature: ollama-llm-integration, Property 10: Invalid LLM grounding response falls back to entity.name
    Validates: Requirements 13.5, 3.2

    When the LLM returns valid JSON but "canonical" is not a string, resolve()
    returns {"canonical": entity.name, "ontology": "unknown"}.
    """
    config = _make_ollama_config()
    non_string_canonical = json.dumps({"canonical": 42, "ontology": "NCBI:1598"})

    with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
         patch("semantic.llm_grounder._cache.load", return_value={}), \
         patch("semantic.llm_grounder._cache.save") as mock_save, \
         patch("semantic.ollama_client.OllamaClient.generate",
               return_value=non_string_canonical):
        from semantic.llm_grounder import LLMGrounder
        result = LLMGrounder().resolve(entity)

    assert result == {"canonical": entity.name, "ontology": "unknown"}, (
        f"Expected fallback dict, got {result!r}"
    )
    mock_save.assert_not_called()


# === Property 15 =============================================================
# Feature: ollama-llm-integration, Property 15: Grounding cache key is MD5 of name concatenated with entity_type

@given(entity=candidate_entity())
@settings(max_examples=100)
def test_property15_cache_key_is_md5_of_name_plus_entity_type(entity):
    """
    # Feature: ollama-llm-integration, Property 15: Grounding cache key is MD5 of name concatenated with entity_type
    Validates: Requirements 6.2

    For any CandidateEntity, the cache key used by LLMGrounder equals
    hashlib.md5((name + entity_type).encode("utf-8")).hexdigest().
    """
    from semantic._cache import _JsonFileCache

    valid_response = _valid_grounding_json()
    config = _make_ollama_config()
    expected_key = hashlib.md5(
        (entity.name + entity.entity_type).encode("utf-8")
    ).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "llm_ground_cache.json"
        real_cache = _JsonFileCache(cache_file)

        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=valid_response):
            from semantic.llm_grounder import LLMGrounder
            LLMGrounder().resolve(entity)

        saved = real_cache.load()

    assert expected_key in saved, (
        f"Expected cache key {expected_key!r} (MD5 of {entity.name!r} + {entity.entity_type!r}) "
        f"not found in cache. Keys present: {list(saved.keys())}"
    )


# =============================================================================
# Unit Tests (Task 6.3)
# Requirements: 5.1-5.7, 6.1-6.7, 8.2, 8.4, 8.6, 8.7, 8.8
# =============================================================================


class TestLLMGrounderUnit:
    """Unit tests for LLMGrounder."""

    # --- Test 1: Cache hit skips Ollama call ---------------------------------

    def test_cache_hit_skips_ollama_call(self, tmp_path):
        """
        Req 5.2, 6.6: when the cache already has a valid entry for the entity,
        resolve() returns the cached dict without calling OllamaClient.generate().
        """
        from semantic._cache import _JsonFileCache

        entity = _make_entity()
        key = _cache_key(entity)
        cached_result = {"canonical": "Lactobacillus reuteri DSM 17938", "ontology": "NCBI:1598"}

        cache_file = tmp_path / "llm_ground_cache.json"
        real_cache = _JsonFileCache(cache_file)
        real_cache.save({key: cached_result})

        config = _make_ollama_config()
        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate") as mock_gen:
            from semantic.llm_grounder import LLMGrounder
            result = LLMGrounder().resolve(entity)

        mock_gen.assert_not_called()
        assert result == {"canonical": "Lactobacillus reuteri DSM 17938", "ontology": "NCBI:1598"}

    # --- Test 2: Empty canonical substituted with entity.name ----------------

    def test_empty_canonical_substituted_with_entity_name(self):
        """
        Req 3.6: when Ollama returns {"canonical": "", "ontology": "NCBI:1234"},
        the returned dict has canonical == entity.name.
        """
        entity = _make_entity(name="Lactobacillus reuteri")
        response_with_empty_canonical = json.dumps({"canonical": "", "ontology": "NCBI:1234"})

        config = _make_ollama_config()
        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache.load", return_value={}), \
             patch("semantic.llm_grounder._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=response_with_empty_canonical):
            from semantic.llm_grounder import LLMGrounder
            result = LLMGrounder().resolve(entity)

        assert result["canonical"] == entity.name, (
            f"Expected canonical to be entity.name={entity.name!r}, got {result['canonical']!r}"
        )
        assert result["ontology"] == "NCBI:1234"


    # --- Test 3: Ollama error returns fallback dict --------------------------

    def test_ollama_unavailable_returns_fallback_dict(self, caplog):
        """
        Req 5.5, 5.6: OllamaUnavailableError returns
        {"canonical": entity.name, "ontology": "unknown"} and logs ERROR.
        """
        from semantic.ollama_client import OllamaUnavailableError

        entity = _make_entity(name="Bifidobacterium longum")
        config = _make_ollama_config()

        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache.load", return_value={}), \
             patch("semantic.llm_grounder._cache.save"), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   side_effect=OllamaUnavailableError("server down", 1)), \
             caplog.at_level(logging.ERROR, logger="semantic.llm_grounder"):
            from semantic.llm_grounder import LLMGrounder
            result = LLMGrounder().resolve(entity)

        assert result == {"canonical": entity.name, "ontology": "unknown"}

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_messages) > 0, "Expected at least one ERROR log message"


    # --- Test 5: Invalid JSON returns fallback dict --------------------------

    def test_invalid_json_returns_fallback_dict(self, caplog):
        """
        Req 3.2, 5.4: when Ollama returns a non-JSON string, resolve() returns
        {"canonical": entity.name, "ontology": "unknown"} and logs ERROR.
        Cache must NOT be written.
        """
        entity = _make_entity(name="Faecalibacterium prausnitzii")
        config = _make_ollama_config()

        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache.load", return_value={}), \
             patch("semantic.llm_grounder._cache.save") as mock_save, \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value="this is definitely not valid JSON!!!"), \
             caplog.at_level(logging.ERROR, logger="semantic.llm_grounder"):
            from semantic.llm_grounder import LLMGrounder
            result = LLMGrounder().resolve(entity)

        assert result == {"canonical": entity.name, "ontology": "unknown"}
        mock_save.assert_not_called()

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_messages) > 0, "Expected at least one ERROR log message"

    # --- Test 6: Successful Ollama response is cached ------------------------

    def test_successful_response_is_cached(self, tmp_path):
        """
        Req 5.3, 6.2, 6.3, 6.5: after a successful resolve(), the cache file
        contains the result keyed by MD5 of name + entity_type.
        """
        from semantic._cache import _JsonFileCache

        entity = _make_entity(name="Roseburia intestinalis", entity_type="taxon")
        cache_file = tmp_path / "llm_ground_cache.json"
        real_cache = _JsonFileCache(cache_file)

        config = _make_ollama_config()
        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache", real_cache), \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=_valid_grounding_json(
                       canonical="Roseburia intestinalis L1-82",
                       ontology="NCBI:166486"
                   )):
            from semantic.llm_grounder import LLMGrounder
            result = LLMGrounder().resolve(entity)

        assert result["canonical"] == "Roseburia intestinalis L1-82"
        assert result["ontology"] == "NCBI:166486"

        assert cache_file.exists(), "Cache file must be created after successful resolve()"
        saved = json.loads(cache_file.read_text())
        expected_key = _cache_key(entity)
        assert expected_key in saved, (
            f"Cache key {expected_key!r} not found in {list(saved.keys())}"
        )
        assert saved[expected_key]["canonical"] == "Roseburia intestinalis L1-82"
        assert saved[expected_key]["ontology"] == "NCBI:166486"


    # --- Test 8: Missing "ontology" key returns fallback dict ----------------

    def test_missing_ontology_key_returns_fallback_dict(self, caplog):
        """
        Req 3.2, 5.4: when Ollama returns {"canonical": "SomeName"} (missing ontology),
        resolve() returns {"canonical": entity.name, "ontology": "unknown"}.
        Cache must NOT be written.
        """
        entity = _make_entity(name="Clostridium difficile")
        config = _make_ollama_config()

        response_missing_ontology = json.dumps({"canonical": "Clostridioides difficile"})

        with patch("semantic.llm_grounder.BACKEND_CONFIG", config), \
             patch("semantic.llm_grounder._cache.load", return_value={}), \
             patch("semantic.llm_grounder._cache.save") as mock_save, \
             patch("semantic.ollama_client.OllamaClient.generate",
                   return_value=response_missing_ontology), \
             caplog.at_level(logging.ERROR, logger="semantic.llm_grounder"):
            from semantic.llm_grounder import LLMGrounder
            result = LLMGrounder().resolve(entity)

        assert result == {"canonical": entity.name, "ontology": "unknown"}
        mock_save.assert_not_called()

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_messages) > 0, "Expected at least one ERROR log message"
