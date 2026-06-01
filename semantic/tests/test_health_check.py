"""
Unit tests for check_ollama_health() in semantic/ollama_client.py.

Covers requirements 9.1–9.7:
  9.1  GET {OLLAMA_BASE_URL}/api/tags with 10-second timeout
  9.2  Verify both extraction and grounding models appear in models[].name
  9.3  Both models found → True, log INFO with model names and base URL
  9.4  Server unreachable or non-200 → False, log ERROR with base URL / status
  9.5  One or more models missing → False, log ERROR with ollama pull hint
  9.6  Callable as check_ollama_health() -> bool
  9.7  LLM_BACKEND=gemini → skip probe, return True without HTTP call

All tests pass a BackendConfig directly to check_ollama_health(config=...)
to avoid touching environment variables.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from config import BackendConfig
from semantic.ollama_client import check_ollama_health


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_ollama_config(
    extraction_model: str = "llama3",
    grounding_model: str = "mistral",
    base_url: str = "http://localhost:11434",
) -> BackendConfig:
    """Build a BackendConfig pointing at the Ollama backend."""
    return BackendConfig(
        llm_backend="ollama",
        ollama_base_url=base_url,
        ollama_extraction_model=extraction_model,
        ollama_grounding_model=grounding_model,
        ollama_timeout_seconds=30,
        ollama_max_retries=3,
        ollama_retry_backoff_base=2.0,
        ollama_fallback_to_gemini=False,
        gemini_extraction_model="gemini-2.0-flash",
        gemini_grounding_model="gemini-2.5-flash",
    )


def _make_gemini_config() -> BackendConfig:
    """Build a BackendConfig pointing at the Gemini backend."""
    return BackendConfig(
        llm_backend="gemini",
        ollama_base_url="http://localhost:11434",
        ollama_extraction_model="llama3",
        ollama_grounding_model="mistral",
        ollama_timeout_seconds=30,
        ollama_max_retries=3,
        ollama_retry_backoff_base=2.0,
        ollama_fallback_to_gemini=False,
        gemini_extraction_model="gemini-2.0-flash",
        gemini_grounding_model="gemini-2.5-flash",
    )


def _make_tags_response(model_names: list[str], status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response for GET /api/tags."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "models": [{"name": name} for name in model_names]
    }
    return mock_resp


# ─── Test 1: Both models present → True, logs INFO ────────────────────────────

def test_both_models_present_returns_true(caplog):
    """
    Req 9.2, 9.3: When both extraction and grounding models appear in the
    /api/tags response, check_ollama_health() returns True and logs INFO
    containing the model names and base URL.
    """
    config = _make_ollama_config(extraction_model="llama3", grounding_model="mistral")
    response = _make_tags_response(["llama3", "mistral", "phi3"])

    with patch("requests.get", return_value=response):
        with caplog.at_level(logging.INFO, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is True
    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("llama3" in msg for msg in info_messages)
    assert any("mistral" in msg for msg in info_messages)
    assert any("http://localhost:11434" in msg for msg in info_messages)


# ─── Test 2: One model missing → False, logs ERROR with pull hint ─────────────

def test_one_model_missing_returns_false_with_pull_hint(caplog):
    """
    Req 9.5: When one required model is absent from the /api/tags response,
    check_ollama_health() returns False and logs an ERROR containing
    'ollama pull <model_name>' for the missing model.
    """
    config = _make_ollama_config(extraction_model="llama3", grounding_model="mistral")
    # Only llama3 is available; mistral is missing
    response = _make_tags_response(["llama3", "phi3"])

    with patch("requests.get", return_value=response):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("mistral" in msg for msg in error_messages)
    assert any("ollama pull mistral" in msg for msg in error_messages)


# ─── Test 3: Both models missing → False, logs ERROR with pull hints for both ─

def test_both_models_missing_returns_false_with_pull_hints(caplog):
    """
    Req 9.5: When both required models are absent, check_ollama_health() returns
    False and logs ERROR with 'ollama pull' hints for each missing model.
    """
    config = _make_ollama_config(extraction_model="llama3", grounding_model="mistral")
    # Neither model is available
    response = _make_tags_response(["phi3", "gemma2"])

    with patch("requests.get", return_value=response):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("ollama pull llama3" in msg for msg in error_messages)
    assert any("ollama pull mistral" in msg for msg in error_messages)


# ─── Test 4: Connection error → False, logs ERROR ─────────────────────────────

def test_connection_error_returns_false(caplog):
    """
    Req 9.4: When the Ollama server is unreachable (ConnectionError),
    check_ollama_health() returns False and logs an ERROR containing the base URL.
    """
    config = _make_ollama_config()

    with patch("requests.get", side_effect=requests.ConnectionError("Connection refused")):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("http://localhost:11434" in msg for msg in error_messages)


# ─── Test 5: Non-200 HTTP status → False, logs ERROR with status code ─────────

def test_non_200_status_returns_false(caplog):
    """
    Req 9.4: When the Ollama server returns a non-200 status (e.g. 503),
    check_ollama_health() returns False and logs an ERROR containing the status code.
    """
    config = _make_ollama_config()
    response = _make_tags_response([], status_code=503)

    with patch("requests.get", return_value=response):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("503" in msg for msg in error_messages)


# ─── Test 6: LLM_BACKEND=gemini → True without HTTP call ──────────────────────

def test_gemini_backend_returns_true_without_http_call():
    """
    Req 9.7: When LLM_BACKEND is 'gemini', check_ollama_health() skips the
    Ollama probe and returns True without making any HTTP request.
    """
    config = _make_gemini_config()

    with patch("requests.get") as mock_get:
        result = check_ollama_health(config=config)

    assert result is True
    mock_get.assert_not_called()


# ─── Test 7: Extraction and grounding models are the same (deduplication) ─────

def test_same_extraction_and_grounding_model_returns_true():
    """
    Edge case: When extraction and grounding models are the same string,
    the deduplication in required_models (a set) means only one model needs
    to be present. check_ollama_health() should return True.
    """
    config = _make_ollama_config(extraction_model="llama3", grounding_model="llama3")
    response = _make_tags_response(["llama3"])

    with patch("requests.get", return_value=response):
        result = check_ollama_health(config=config)

    assert result is True


def test_same_model_missing_returns_false(caplog):
    """
    Edge case: When extraction and grounding models are the same and that model
    is missing, check_ollama_health() returns False with exactly one pull hint.
    """
    config = _make_ollama_config(extraction_model="llama3", grounding_model="llama3")
    response = _make_tags_response(["mistral"])

    with patch("requests.get", return_value=response):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    pull_hints = [msg for msg in error_messages if "ollama pull llama3" in msg]
    # Only one pull hint for the single missing model (set deduplication)
    assert len(pull_hints) == 1


# ─── Test 8: Invalid/unparseable JSON response → False, logs ERROR ────────────

def test_invalid_json_response_returns_false(caplog):
    """
    Req 9.4: When the /api/tags response body cannot be parsed as JSON,
    check_ollama_health() returns False and logs an ERROR.
    """
    config = _make_ollama_config()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("No JSON object could be decoded")

    with patch("requests.get", return_value=mock_resp):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_messages) >= 1


# ─── Test 9: GET request uses correct URL and 10-second timeout ───────────────

def test_get_request_uses_correct_url_and_timeout():
    """
    Req 9.1: GET request must target {base_url}/api/tags with timeout=10.
    """
    config = _make_ollama_config(base_url="http://localhost:11434")
    response = _make_tags_response(["llama3", "mistral"])

    with patch("requests.get", return_value=response) as mock_get:
        check_ollama_health(config=config)

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    # First positional arg is the URL
    assert call_args[0][0] == "http://localhost:11434/api/tags"
    # timeout keyword arg must be 10
    assert call_args[1]["timeout"] == 10


# ─── Test 10: Callable as standalone function returning bool ──────────────────

def test_callable_as_standalone_function():
    """
    Req 9.6: check_ollama_health() is importable and callable as a standalone
    function that returns a bool.
    """
    config = _make_ollama_config()
    response = _make_tags_response(["llama3", "mistral"])

    with patch("requests.get", return_value=response):
        result = check_ollama_health(config=config)

    assert isinstance(result, bool)


# ─── Test 11: Non-200 status code 404 → False ─────────────────────────────────

def test_404_status_returns_false(caplog):
    """
    Req 9.4: HTTP 404 (or any non-200) causes the health check to return False.
    """
    config = _make_ollama_config()
    response = _make_tags_response([], status_code=404)

    with patch("requests.get", return_value=response):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("404" in msg for msg in error_messages)


# ─── Test 12: Generic exception (e.g. OSError) → False, logs ERROR ────────────

def test_generic_exception_returns_false(caplog):
    """
    Req 9.4: Any exception during the GET request (not just ConnectionError)
    causes check_ollama_health() to return False and log an ERROR.
    """
    config = _make_ollama_config()

    with patch("requests.get", side_effect=OSError("Network unreachable")):
        with caplog.at_level(logging.ERROR, logger="semantic.ollama_client"):
            result = check_ollama_health(config=config)

    assert result is False
    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_messages) >= 1
