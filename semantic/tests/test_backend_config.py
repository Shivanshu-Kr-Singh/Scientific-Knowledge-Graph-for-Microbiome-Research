"""
Tests for BackendConfig validation in config.py.

Covers:
  - Property 12: BackendConfig raises ConfigurationError for non-numeric env vars
  - Property 13: BackendConfig raises ConfigurationError for invalid LLM_BACKEND values
  - Unit tests: valid env vars produce correct types; each invalid env var raises
    ConfigurationError with correct message; missing GEMINI_API_KEY when required.
"""

import os
import unittest
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

from config import ConfigurationError, _load_backend_config


# ─── Hypothesis strategies ────────────────────────────────────────────────────

# Strings that cannot be parsed as int or float (for Property 12).
# The filter mirrors the spec: strip whitespace, strip leading minus, remove at
# most one decimal point, then check isdigit().
# Also exclude null bytes since os.environ cannot hold them.
non_numeric_string = st.text(min_size=1).filter(
    lambda s: "\x00" not in s
    and not s.strip().lstrip("-").replace(".", "", 1).isdigit()
)

# Strings that are not valid LLM_BACKEND values (for Property 13).
# Exclude null bytes since os.environ cannot hold them.
invalid_backend = st.text().filter(
    lambda s: s not in {"ollama", "gemini"} and "\x00" not in s
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _call_with_env(**overrides):
    """Call _load_backend_config() with a clean env that only has the given vars."""
    base = {
        "LLM_BACKEND": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EXTRACTION_MODEL": "llama3",
        "OLLAMA_GROUNDING_MODEL": "llama3",
        "OLLAMA_TIMEOUT_SECONDS": "30",
        "OLLAMA_MAX_RETRIES": "3",
        "OLLAMA_RETRY_BACKOFF_BASE": "2.0",
        "OLLAMA_FALLBACK_TO_GEMINI": "false",
    }
    base.update(overrides)
    # Remove keys whose value is None (simulates absent env var)
    env = {k: v for k, v in base.items() if v is not None}
    with patch.dict(os.environ, env, clear=True):
        return _load_backend_config()


# ─── Property 12 ─────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 12: BackendConfig raises ConfigurationError for non-numeric env vars

@given(bad_value=non_numeric_string)
@settings(max_examples=100)
def test_property12_non_numeric_timeout_raises(bad_value):
    """
    # Feature: ollama-llm-integration, Property 12: BackendConfig raises ConfigurationError for non-numeric env vars
    Validates: Requirements 1.2
    For any non-numeric string passed as OLLAMA_TIMEOUT_SECONDS, _load_backend_config()
    SHALL raise ConfigurationError that includes the variable name and the invalid value.
    """
    with pytest.raises(ConfigurationError) as exc_info:
        _call_with_env(OLLAMA_TIMEOUT_SECONDS=bad_value)
    msg = str(exc_info.value)
    assert "OLLAMA_TIMEOUT_SECONDS" in msg
    assert bad_value in msg or repr(bad_value) in msg


@given(bad_value=non_numeric_string)
@settings(max_examples=100)
def test_property12_non_numeric_max_retries_raises(bad_value):
    """
    # Feature: ollama-llm-integration, Property 12: BackendConfig raises ConfigurationError for non-numeric env vars
    Validates: Requirements 1.2
    For any non-numeric string passed as OLLAMA_MAX_RETRIES, _load_backend_config()
    SHALL raise ConfigurationError that includes the variable name and the invalid value.
    """
    with pytest.raises(ConfigurationError) as exc_info:
        _call_with_env(OLLAMA_MAX_RETRIES=bad_value)
    msg = str(exc_info.value)
    assert "OLLAMA_MAX_RETRIES" in msg
    assert bad_value in msg or repr(bad_value) in msg


@given(bad_value=non_numeric_string)
@settings(max_examples=100)
def test_property12_non_numeric_backoff_raises(bad_value):
    """
    # Feature: ollama-llm-integration, Property 12: BackendConfig raises ConfigurationError for non-numeric env vars
    Validates: Requirements 1.2
    For any non-numeric string passed as OLLAMA_RETRY_BACKOFF_BASE, _load_backend_config()
    SHALL raise ConfigurationError that includes the variable name and the invalid value.
    """
    with pytest.raises(ConfigurationError) as exc_info:
        _call_with_env(OLLAMA_RETRY_BACKOFF_BASE=bad_value)
    msg = str(exc_info.value)
    assert "OLLAMA_RETRY_BACKOFF_BASE" in msg
    assert bad_value in msg or repr(bad_value) in msg


# ─── Property 13 ─────────────────────────────────────────────────────────────
# Feature: ollama-llm-integration, Property 13: BackendConfig raises ConfigurationError for invalid LLM_BACKEND values

@given(bad_backend=invalid_backend)
@settings(max_examples=100)
def test_property13_invalid_backend_raises(bad_backend):
    """
    # Feature: ollama-llm-integration, Property 13: BackendConfig raises ConfigurationError for invalid LLM_BACKEND values
    Validates: Requirements 1.3
    For any string not in {"ollama", "gemini"} passed as LLM_BACKEND,
    _load_backend_config() SHALL raise ConfigurationError that lists the accepted values.
    """
    with pytest.raises(ConfigurationError) as exc_info:
        _call_with_env(LLM_BACKEND=bad_backend)
    msg = str(exc_info.value)
    # The error message must mention both accepted values
    assert "ollama" in msg
    assert "gemini" in msg


# ─── Unit tests: valid env vars produce correct types ─────────────────────────

class TestBackendConfigValidEnvVars(unittest.TestCase):

    def test_defaults_produce_correct_types(self):
        """Valid defaults produce a BackendConfig with correct Python types."""
        cfg = _call_with_env()
        self.assertIsInstance(cfg.llm_backend, str)
        self.assertIsInstance(cfg.ollama_base_url, str)
        self.assertIsInstance(cfg.ollama_extraction_model, str)
        self.assertIsInstance(cfg.ollama_grounding_model, str)
        self.assertIsInstance(cfg.ollama_timeout_seconds, int)
        self.assertIsInstance(cfg.ollama_max_retries, int)
        self.assertIsInstance(cfg.ollama_retry_backoff_base, float)
        self.assertIsInstance(cfg.ollama_fallback_to_gemini, bool)
        self.assertIsInstance(cfg.gemini_extraction_model, str)
        self.assertIsInstance(cfg.gemini_grounding_model, str)

    def test_default_values(self):
        """Default env vars produce the expected default values."""
        cfg = _call_with_env()
        self.assertEqual(cfg.llm_backend, "ollama")
        self.assertEqual(cfg.ollama_base_url, "http://localhost:11434")
        self.assertEqual(cfg.ollama_extraction_model, "llama3")
        self.assertEqual(cfg.ollama_grounding_model, "llama3")
        self.assertEqual(cfg.ollama_timeout_seconds, 30)
        self.assertEqual(cfg.ollama_max_retries, 3)
        self.assertAlmostEqual(cfg.ollama_retry_backoff_base, 2.0)
        self.assertFalse(cfg.ollama_fallback_to_gemini)
        self.assertEqual(cfg.gemini_extraction_model, "gemini-2.0-flash")
        self.assertEqual(cfg.gemini_grounding_model, "gemini-2.5-flash")

    def test_ollama_backend_accepted(self):
        cfg = _call_with_env(LLM_BACKEND="ollama")
        self.assertEqual(cfg.llm_backend, "ollama")

    def test_gemini_backend_accepted_with_api_key(self):
        cfg = _call_with_env(LLM_BACKEND="gemini", GEMINI_API_KEY="test-key")
        self.assertEqual(cfg.llm_backend, "gemini")

    def test_fallback_true_string_becomes_bool_true(self):
        cfg = _call_with_env(
            OLLAMA_FALLBACK_TO_GEMINI="true",
            GEMINI_API_KEY="test-key",
        )
        self.assertTrue(cfg.ollama_fallback_to_gemini)

    def test_fallback_false_string_becomes_bool_false(self):
        cfg = _call_with_env(OLLAMA_FALLBACK_TO_GEMINI="false")
        self.assertFalse(cfg.ollama_fallback_to_gemini)

    def test_fallback_arbitrary_string_becomes_bool_false(self):
        """Any value other than 'true' maps to False."""
        cfg = _call_with_env(OLLAMA_FALLBACK_TO_GEMINI="yes")
        self.assertFalse(cfg.ollama_fallback_to_gemini)

    def test_custom_numeric_values_parsed_correctly(self):
        cfg = _call_with_env(
            OLLAMA_TIMEOUT_SECONDS="60",
            OLLAMA_MAX_RETRIES="5",
            OLLAMA_RETRY_BACKOFF_BASE="3.0",
        )
        self.assertEqual(cfg.ollama_timeout_seconds, 60)
        self.assertEqual(cfg.ollama_max_retries, 5)
        self.assertAlmostEqual(cfg.ollama_retry_backoff_base, 3.0)

    def test_max_retries_zero_is_valid(self):
        cfg = _call_with_env(OLLAMA_MAX_RETRIES="0")
        self.assertEqual(cfg.ollama_max_retries, 0)

    def test_timeout_one_is_valid(self):
        cfg = _call_with_env(OLLAMA_TIMEOUT_SECONDS="1")
        self.assertEqual(cfg.ollama_timeout_seconds, 1)

    def test_backoff_one_is_valid(self):
        cfg = _call_with_env(OLLAMA_RETRY_BACKOFF_BASE="1.0")
        self.assertAlmostEqual(cfg.ollama_retry_backoff_base, 1.0)


# ─── Unit tests: each invalid env var raises ConfigurationError ───────────────

class TestBackendConfigInvalidEnvVars(unittest.TestCase):

    def test_invalid_backend_raises_with_accepted_values(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(LLM_BACKEND="openai")
        msg = str(ctx.exception)
        self.assertIn("ollama", msg)
        self.assertIn("gemini", msg)

    def test_non_int_timeout_raises_with_var_name_and_value(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_TIMEOUT_SECONDS="abc")
        msg = str(ctx.exception)
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", msg)
        self.assertIn("abc", msg)

    def test_non_int_max_retries_raises_with_var_name_and_value(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_MAX_RETRIES="xyz")
        msg = str(ctx.exception)
        self.assertIn("OLLAMA_MAX_RETRIES", msg)
        self.assertIn("xyz", msg)

    def test_non_float_backoff_raises_with_var_name_and_value(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_RETRY_BACKOFF_BASE="not-a-float")
        msg = str(ctx.exception)
        self.assertIn("OLLAMA_RETRY_BACKOFF_BASE", msg)
        self.assertIn("not-a-float", msg)

    def test_timeout_zero_raises(self):
        """OLLAMA_TIMEOUT_SECONDS=0 is below the minimum of 1."""
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_TIMEOUT_SECONDS="0")
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", str(ctx.exception))

    def test_negative_timeout_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_TIMEOUT_SECONDS="-5")
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", str(ctx.exception))

    def test_negative_max_retries_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_MAX_RETRIES="-1")
        self.assertIn("OLLAMA_MAX_RETRIES", str(ctx.exception))

    def test_backoff_below_one_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_RETRY_BACKOFF_BASE="0.5")
        self.assertIn("OLLAMA_RETRY_BACKOFF_BASE", str(ctx.exception))

    def test_float_string_for_int_field_raises(self):
        """'3.5' is a valid float but not a valid int."""
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_TIMEOUT_SECONDS="3.5")
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", str(ctx.exception))


# ─── Unit tests: missing GEMINI_API_KEY when required ─────────────────────────

class TestBackendConfigGeminiApiKey(unittest.TestCase):

    def test_gemini_backend_without_api_key_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(LLM_BACKEND="gemini")  # no GEMINI_API_KEY
        msg = str(ctx.exception)
        self.assertIn("GEMINI_API_KEY", msg)

    def test_fallback_true_without_api_key_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_FALLBACK_TO_GEMINI="true")  # no GEMINI_API_KEY
        msg = str(ctx.exception)
        self.assertIn("GEMINI_API_KEY", msg)

    def test_gemini_backend_with_api_key_does_not_raise(self):
        cfg = _call_with_env(LLM_BACKEND="gemini", GEMINI_API_KEY="my-key")
        self.assertEqual(cfg.llm_backend, "gemini")

    def test_fallback_true_with_api_key_does_not_raise(self):
        cfg = _call_with_env(OLLAMA_FALLBACK_TO_GEMINI="true", GEMINI_API_KEY="my-key")
        self.assertTrue(cfg.ollama_fallback_to_gemini)

    def test_ollama_backend_without_api_key_does_not_raise(self):
        """Ollama backend with no fallback doesn't need GEMINI_API_KEY."""
        cfg = _call_with_env(LLM_BACKEND="ollama", OLLAMA_FALLBACK_TO_GEMINI="false")
        self.assertEqual(cfg.llm_backend, "ollama")
