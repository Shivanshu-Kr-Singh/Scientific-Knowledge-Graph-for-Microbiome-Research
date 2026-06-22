"""
Shared Hypothesis strategies and pytest fixtures for the ollama-llm-integration
test suite.

All strategies defined here are importable by any test file in this package:

    from semantic.tests.conftest import non_empty_text, candidate_entity, ...

They are also available as pytest fixtures (for tests that prefer fixture injection).

Requirements: 12.1–12.5, 13.1–13.5
"""

import pytest
from hypothesis import strategies as st
from hypothesis.strategies import composite

from semantic.candidate_store import CandidateEntity


# ---------------------------------------------------------------------------
# Strategy: arbitrary non-empty text
# At least one non-whitespace character.
# ---------------------------------------------------------------------------
non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())


# ---------------------------------------------------------------------------
# Strategy: whitespace-only text (for Property 11)
# Includes the empty string.
# ---------------------------------------------------------------------------
whitespace_text = st.text(alphabet=" \t\n\r", min_size=0)


# ---------------------------------------------------------------------------
# Strategy: arbitrary CandidateEntity
# Both name and entity_type contain at least one non-whitespace character.
# ---------------------------------------------------------------------------
@composite
def candidate_entity(draw):
    """Generate an arbitrary CandidateEntity with non-empty name and entity_type."""
    name = draw(st.text(min_size=1).filter(lambda s: s.strip()))
    entity_type = draw(st.text(min_size=1).filter(lambda s: s.strip()))
    return CandidateEntity(name=name, entity_type=entity_type)


# ---------------------------------------------------------------------------
# Strategy: valid extraction schema dict
# Contains "entities" (list), "relations" (list), and "evidence" (dict).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Strategy: valid grounding schema dict
# Contains "canonical" (non-empty string) and "ontology" (string).
# ---------------------------------------------------------------------------
@composite
def grounding_schema(draw):
    """Generate a valid grounding schema dict."""
    canonical = draw(st.text(min_size=1).filter(lambda s: s.strip()))
    ontology = draw(st.text())
    return {"canonical": canonical, "ontology": ontology}


# ---------------------------------------------------------------------------
# Strategy: invalid LLM response strings (for error condition properties)
# Covers: arbitrary text, empty string, valid JSON missing keys, wrong type,
# and markdown-fenced JSON.
# ---------------------------------------------------------------------------
invalid_llm_response = st.one_of(
    st.text(),                          # arbitrary text (may or may not be valid JSON)
    st.just(""),                        # empty string
    st.just("{}"),                      # valid JSON but missing required keys
    st.just('{"canonical": 42}'),       # wrong type for canonical
    st.just("```json\n{}\n```"),        # markdown fence
)


# ---------------------------------------------------------------------------
# Strategy: non-numeric strings for config validation (Property 12)
# Strings that cannot be parsed as int or float.
# Null bytes are excluded because os.environ cannot hold them.
# ---------------------------------------------------------------------------
non_numeric_string = st.text(min_size=1).filter(
    lambda s: "\x00" not in s
    and not s.strip().lstrip("-").replace(".", "", 1).isdigit()
)


# ---------------------------------------------------------------------------
# Strategy: invalid backend strings (Property 13)
# Any string that is not "ollama".
# Null bytes are excluded because os.environ cannot hold them.
# ---------------------------------------------------------------------------
invalid_backend = st.text().filter(
    lambda s: s != "ollama" and "\x00" not in s
)


# ---------------------------------------------------------------------------
# Pytest fixtures — thin wrappers so tests can also receive strategies via
# fixture injection if they prefer that style.
# ---------------------------------------------------------------------------

@pytest.fixture
def non_empty_text_strategy():
    """Pytest fixture returning the non_empty_text Hypothesis strategy."""
    return non_empty_text


@pytest.fixture
def whitespace_text_strategy():
    """Pytest fixture returning the whitespace_text Hypothesis strategy."""
    return whitespace_text


@pytest.fixture
def candidate_entity_strategy():
    """Pytest fixture returning the candidate_entity Hypothesis strategy."""
    return candidate_entity


@pytest.fixture
def extraction_schema_strategy():
    """Pytest fixture returning the extraction_schema Hypothesis strategy."""
    return extraction_schema


@pytest.fixture
def grounding_schema_strategy():
    """Pytest fixture returning the grounding_schema Hypothesis strategy."""
    return grounding_schema


@pytest.fixture
def invalid_llm_response_strategy():
    """Pytest fixture returning the invalid_llm_response Hypothesis strategy."""
    return invalid_llm_response


@pytest.fixture
def non_numeric_string_strategy():
    """Pytest fixture returning the non_numeric_string Hypothesis strategy."""
    return non_numeric_string


@pytest.fixture
def invalid_backend_strategy():
    """Pytest fixture returning the invalid_backend Hypothesis strategy."""
    return invalid_backend
