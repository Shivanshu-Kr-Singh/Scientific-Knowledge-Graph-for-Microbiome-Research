"""
Tests for OllamaClient in semantic/ollama_client.py.

Covers:
  - Property 14: OllamaClient makes at most MAX_RETRIES + 1 total attempts
    Validates: Requirements 10.2, 2.3
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from hypothesis import given, settings, strategies as st

from config import BackendConfig
from semantic.ollama_client import OllamaClient, OllamaTimeoutError, OllamaUnavailableError


# ─── Helper ───────────────────────────────────────────────────────────────────

def _make_config(max_retries: int) -> BackendConfig:
    """Build a BackendConfig with the given max_retries and fast settings."""
    return BackendConfig(
        llm_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_extraction_model="llama3",
        ollama_grounding_model="llama3",
        ollama_timeout_seconds=30,
        ollama_max_retries=max_retries,
        ollama_retry_backoff_base=1.0,  # base=1.0 → sleep(1.0^n) = sleep(1s), patched anyway
        ollama_fallback_to_gemini=False,
        gemini_extraction_model="gemini-2.0-flash",
        gemini_grounding_model="gemini-2.5-flash",
    )


# ─── Property 14 ─────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 14: OllamaClient makes at most MAX_RETRIES + 1 total attempts

@given(max_retries=st.integers(min_value=0, max_value=5))
@settings(max_examples=100)
def test_property14_retry_attempt_count_on_http_error(max_retries):
    """
    # Feature: ollama-llm-integration, Property 14: OllamaClient makes at most MAX_RETRIES + 1 total attempts
    Validates: Requirements 10.2, 2.3

    For any OLLAMA_MAX_RETRIES = N (N ≥ 0), when the Ollama server consistently
    returns HTTP 500, OllamaClient.generate() SHALL make exactly N + 1 total
    HTTP attempts before raising OllamaUnavailableError.
    """
    config = _make_config(max_retries)
    client = OllamaClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("requests.post", return_value=mock_response) as mock_post, \
         patch("time.sleep"):  # suppress backoff delays
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    assert mock_post.call_count == max_retries + 1


@given(max_retries=st.integers(min_value=0, max_value=5))
@settings(max_examples=100)
def test_property14_retry_attempt_count_on_connection_error(max_retries):
    """
    # Feature: ollama-llm-integration, Property 14: OllamaClient makes at most MAX_RETRIES + 1 total attempts
    Validates: Requirements 10.2, 2.3

    For any OLLAMA_MAX_RETRIES = N (N ≥ 0), when the Ollama server consistently
    raises a ConnectionError, OllamaClient.generate() SHALL make exactly N + 1
    total HTTP attempts before raising OllamaUnavailableError.
    """
    config = _make_config(max_retries)
    client = OllamaClient(config)

    with patch("requests.post", side_effect=requests.ConnectionError("refused")) as mock_post, \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    assert mock_post.call_count == max_retries + 1


@given(max_retries=st.integers(min_value=0, max_value=5))
@settings(max_examples=100)
def test_property14_retry_attempt_count_on_timeout(max_retries):
    """
    # Feature: ollama-llm-integration, Property 14: OllamaClient makes at most MAX_RETRIES + 1 total attempts
    Validates: Requirements 10.2, 2.3

    For any OLLAMA_MAX_RETRIES = N (N ≥ 0), when every request times out,
    OllamaClient.generate() SHALL make exactly N + 1 total HTTP attempts
    before raising OllamaTimeoutError.
    """
    config = _make_config(max_retries)
    client = OllamaClient(config)

    with patch("requests.post", side_effect=requests.Timeout("timed out")) as mock_post, \
         patch("time.sleep"):
        with pytest.raises(OllamaTimeoutError):
            client.generate("llama3", "test prompt")

    assert mock_post.call_count == max_retries + 1


# ─── Unit Tests ───────────────────────────────────────────────────────────────
# Task 3.3: Unit tests for OllamaClient
# Requirements: 2.1–2.7, 10.1–10.6

def _make_fast_config(max_retries: int = 2, timeout: int = 5, backoff_base: float = 2.0) -> BackendConfig:
    """Build a BackendConfig with small values for fast unit tests."""
    return BackendConfig(
        llm_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_extraction_model="llama3",
        ollama_grounding_model="llama3",
        ollama_timeout_seconds=timeout,
        ollama_max_retries=max_retries,
        ollama_retry_backoff_base=backoff_base,
        ollama_fallback_to_gemini=False,
        gemini_extraction_model="gemini-2.0-flash",
        gemini_grounding_model="gemini-2.5-flash",
    )


def _make_200_response(response_text: str) -> MagicMock:
    """Return a mock requests.Response with status 200 and a valid JSON body."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": response_text}
    return mock_resp


def _make_200_response_no_field() -> MagicMock:
    """Return a mock requests.Response with status 200 but missing 'response' field."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"other_field": "value"}
    return mock_resp


def _make_200_response_invalid_json() -> MagicMock:
    """Return a mock requests.Response with status 200 but json() raises ValueError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
    return mock_resp


def _make_500_response() -> MagicMock:
    """Return a mock requests.Response with status 500."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    return mock_resp


# ── Test 1: HTTP 200 with valid response field returns the response string ────

def test_http_200_valid_response_returns_string():
    """
    Req 2.1, 2.2: HTTP 200 with valid 'response' field → return that string.
    """
    config = _make_fast_config()
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_200_response('{"result": "ok"}')), \
         patch("time.sleep"):
        result = client.generate("llama3", "test prompt")

    assert result == '{"result": "ok"}'


# ── Test 2: HTTP 200 but missing 'response' field → retry, raise OllamaUnavailableError ──

def test_http_200_missing_response_field_triggers_retry():
    """
    Req 2.2: Missing 'response' field treated as failed attempt → retry policy applied.
    After all retries exhausted, raises OllamaUnavailableError.
    """
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_200_response_no_field()) as mock_post, \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    # 1 initial + 2 retries = 3 total attempts
    assert mock_post.call_count == 3


# ── Test 3: HTTP 200 but invalid JSON body → retry, raise OllamaUnavailableError ──

def test_http_200_invalid_json_triggers_retry():
    """
    Req 2.2: Invalid JSON body treated as failed attempt → retry policy applied.
    After all retries exhausted, raises OllamaUnavailableError.
    """
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_200_response_invalid_json()) as mock_post, \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    assert mock_post.call_count == 3


# ── Test 4: HTTP 500 → retry up to MAX_RETRIES times, then raise OllamaUnavailableError ──

def test_http_500_triggers_retry_and_raises_unavailable():
    """
    Req 2.3, 2.4, 10.2, 10.5: HTTP 500 triggers retry up to MAX_RETRIES times,
    then raises OllamaUnavailableError.
    """
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_500_response()) as mock_post, \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    # 1 initial + 2 retries = 3 total attempts
    assert mock_post.call_count == 3


# ── Test 5: Network error (ConnectionError) → retry, raise OllamaUnavailableError ──

def test_connection_error_triggers_retry_and_raises_unavailable():
    """
    Req 2.3, 2.4, 10.5: ConnectionError triggers retry, raises OllamaUnavailableError
    after all attempts exhausted.
    """
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    with patch("requests.post", side_effect=requests.ConnectionError("Connection refused")) as mock_post, \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    assert mock_post.call_count == 3


# ── Test 6: Timeout → retry, raise OllamaTimeoutError when all timeout ──

def test_timeout_triggers_retry_and_raises_timeout_error():
    """
    Req 2.5, 2.6, 10.3, 10.6: requests.Timeout treated as failed attempt;
    when all retries time out, raises OllamaTimeoutError (not OllamaUnavailableError).
    """
    config = _make_fast_config(max_retries=2, timeout=5)
    client = OllamaClient(config)

    with patch("requests.post", side_effect=requests.Timeout("timed out")) as mock_post, \
         patch("time.sleep"):
        with pytest.raises(OllamaTimeoutError) as exc_info:
            client.generate("llama3", "test prompt")

    assert mock_post.call_count == 3
    # OllamaTimeoutError must NOT be OllamaUnavailableError
    assert not isinstance(exc_info.value, OllamaUnavailableError)


# ── Test 7: Mixed — some timeouts then non-timeout error → OllamaUnavailableError ──

def test_mixed_timeout_then_http_error_raises_unavailable():
    """
    Req 2.5, 2.6, 10.6: When the final attempt fails with a non-timeout error
    (even after earlier timeouts), raises OllamaUnavailableError, not OllamaTimeoutError.
    """
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    # First two attempts timeout, final attempt gets HTTP 500
    side_effects = [
        requests.Timeout("timed out"),
        requests.Timeout("timed out"),
        _make_500_response(),
    ]

    with patch("requests.post", side_effect=side_effects), \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")


# ── Test 8: Backoff formula correctness ──────────────────────────────────────

def test_backoff_formula_correctness():
    """
    Req 10.1, 10.3: Verify time.sleep is called with min(base^attempt, base^max_retries).
    With base=2, max_retries=3:
      attempt 1 → sleep(min(2^1, 2^3)) = sleep(2)
      attempt 2 → sleep(min(2^2, 2^3)) = sleep(4)
      attempt 3 → sleep(min(2^3, 2^3)) = sleep(8)
      attempt 4 (final) → no sleep
    """
    config = _make_fast_config(max_retries=3, backoff_base=2.0)
    client = OllamaClient(config)

    sleep_calls = []

    def capture_sleep(seconds):
        sleep_calls.append(seconds)

    with patch("requests.post", return_value=_make_500_response()), \
         patch("time.sleep", side_effect=capture_sleep):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    # 3 retries → 3 sleep calls (no sleep after final attempt)
    assert len(sleep_calls) == 3
    assert sleep_calls[0] == min(2.0 ** 1, 2.0 ** 3)  # 2.0
    assert sleep_calls[1] == min(2.0 ** 2, 2.0 ** 3)  # 4.0
    assert sleep_calls[2] == min(2.0 ** 3, 2.0 ** 3)  # 8.0 (capped)


def test_backoff_cap_applied():
    """
    Req 10.1: Backoff is capped at base^max_retries.
    With base=2, max_retries=2:
      attempt 1 → sleep(min(2^1, 2^2)) = sleep(2)
      attempt 2 → sleep(min(2^2, 2^2)) = sleep(4)  ← cap = 4
      attempt 3 (final) → no sleep
    """
    config = _make_fast_config(max_retries=2, backoff_base=2.0)
    client = OllamaClient(config)

    sleep_calls = []

    with patch("requests.post", return_value=_make_500_response()), \
         patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == 2.0
    assert sleep_calls[1] == 4.0  # capped at base^max_retries = 2^2 = 4


# ── Test 9: Request body contains correct fields ──────────────────────────────

def test_request_body_contains_correct_fields():
    """
    Req 2.1: POST body must contain model, prompt, stream=False, format="json".
    """
    config = _make_fast_config()
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_200_response("hello")) as mock_post, \
         patch("time.sleep"):
        client.generate("llama3", "my test prompt")

    assert mock_post.call_count == 1
    _, kwargs = mock_post.call_args
    body = kwargs["json"]
    assert body["model"] == "llama3"
    assert body["prompt"] == "my test prompt"
    assert body["stream"] is False
    assert body["format"] == "json"


def test_request_url_is_correct():
    """
    Req 2.1: POST must go to {base_url}/api/generate.
    """
    config = _make_fast_config()
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_200_response("hello")) as mock_post, \
         patch("time.sleep"):
        client.generate("llama3", "prompt")

    call_args = mock_post.call_args
    url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url") or call_args[0][0]
    # First positional arg is the URL
    assert call_args[0][0] == "http://localhost:11434/api/generate"


# ── Test 10: WARNING log emitted on each retry ────────────────────────────────

def test_warning_logged_on_each_retry(caplog):
    """
    Req 2.7, 10.4: WARNING log emitted on each failed attempt with attempt number,
    total allowed attempts, model name, and error.
    """
    import logging
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_500_response()), \
         patch("time.sleep"), \
         caplog.at_level(logging.WARNING, logger="semantic.ollama_client"):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    # 3 total attempts → 3 WARNING logs (2 with backoff info, 1 final "No more retries")
    assert len(warning_records) == 3

    # First two logs mention attempt number and total
    assert "1/3" in warning_records[0].message
    assert "2/3" in warning_records[1].message
    assert "3/3" in warning_records[2].message

    # All logs mention the model name
    for record in warning_records:
        assert "llama3" in record.message


# ── Test 11: OllamaUnavailableError contains attempt count = MAX_RETRIES + 1 ──

def test_unavailable_error_contains_attempt_count():
    """
    Req 2.4: OllamaUnavailableError must contain the number of attempts made,
    which equals MAX_RETRIES + 1.
    """
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_500_response()), \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError) as exc_info:
            client.generate("llama3", "test prompt")

    assert exc_info.value.attempts == 3  # max_retries=2 → 2+1=3 total attempts


def test_unavailable_error_attempt_count_with_zero_retries():
    """
    Req 2.4, 10.2: With max_retries=0, exactly 1 attempt is made.
    OllamaUnavailableError.attempts == 1.
    """
    config = _make_fast_config(max_retries=0)
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_500_response()), \
         patch("time.sleep"):
        with pytest.raises(OllamaUnavailableError) as exc_info:
            client.generate("llama3", "test prompt")

    assert exc_info.value.attempts == 1


# ── Test 12: OllamaTimeoutError contains the configured timeout_seconds value ──

def test_timeout_error_contains_configured_timeout():
    """
    Req 2.6: OllamaTimeoutError must contain the configured timeout_seconds value.
    """
    config = _make_fast_config(max_retries=1, timeout=7)
    client = OllamaClient(config)

    with patch("requests.post", side_effect=requests.Timeout("timed out")), \
         patch("time.sleep"):
        with pytest.raises(OllamaTimeoutError) as exc_info:
            client.generate("llama3", "test prompt")

    assert exc_info.value.timeout_seconds == 7


# ── Test: No backoff sleep after final attempt ────────────────────────────────

def test_no_sleep_after_final_attempt():
    """
    Req 10.3: No backoff is applied after the final attempt.
    With max_retries=2, there are 3 attempts and exactly 2 sleep calls.
    """
    config = _make_fast_config(max_retries=2)
    client = OllamaClient(config)

    sleep_calls = []

    with patch("requests.post", return_value=_make_500_response()), \
         patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        with pytest.raises(OllamaUnavailableError):
            client.generate("llama3", "test prompt")

    # 3 attempts → 2 sleeps (between attempt 1→2 and 2→3; none after attempt 3)
    assert len(sleep_calls) == 2


# ── Test: Timeout also applies backoff before retry ──────────────────────────

def test_timeout_applies_backoff_before_retry():
    """
    Req 10.3: Timeout counts as a failed attempt and applies the same backoff before retrying.
    With max_retries=2, base=2: sleep(2), sleep(4) before attempts 2 and 3.
    """
    config = _make_fast_config(max_retries=2, backoff_base=2.0)
    client = OllamaClient(config)

    sleep_calls = []

    with patch("requests.post", side_effect=requests.Timeout("timed out")), \
         patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        with pytest.raises(OllamaTimeoutError):
            client.generate("llama3", "test prompt")

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == 2.0
    assert sleep_calls[1] == 4.0


# ── Test: Timeout uses configured timeout_seconds in requests.post call ───────

def test_request_uses_configured_timeout():
    """
    Req 2.5: requests.post must be called with the configured timeout value.
    """
    config = _make_fast_config(timeout=7)
    client = OllamaClient(config)

    with patch("requests.post", return_value=_make_200_response("ok")) as mock_post, \
         patch("time.sleep"):
        client.generate("llama3", "prompt")

    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 7
