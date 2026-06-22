"""
Tests for BackendConfig validation in config.py.

Covers:
  - Property 12: BackendConfig raises ConfigurationError for non-numeric env vars
  - Property 13: BackendConfig raises ConfigurationError for invalid LLM_BACKEND values
  - Unit tests: valid env vars produce correct types; each invalid env var raises
    ConfigurationError with correct message.
"""

import os
import unittest
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

from config import ConfigurationError, _load_backend_config


# ─── Hypothesis strategies ────────────────────────────────────────────────────

non_numeric_string = st.text(min_size=1).filter(
    lambda s: "\x00" not in s
    and not s.strip().lstrip("-").replace(".", "", 1).isdigit()
)

# Only "ollama" is valid — any other string should raise
invalid_backend = st.text().filter(
    lambda s: s != "ollama" and "\x00" not in s
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
    }
    base.update(overrides)
    env = {k: v for k, v in base.items() if v is not None}
    with patch.dict(os.environ, env, clear=True):
        return _load_backend_config()


# ─── Property 12 ─────────────────────────────────────────────────────────────

@given(bad_value=non_numeric_string)
@settings(max_examples=100)
def test_property12_non_numeric_timeout_raises(bad_value):
    """
    For any non-numeric string passed as OLLAMA_TIMEOUT_SECONDS,
    _load_backend_config() SHALL raise ConfigurationError with the variable
    name and invalid value in the message.
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
    For any non-numeric string passed as OLLAMA_MAX_RETRIES,
    _load_backend_config() SHALL raise ConfigurationError.
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
    For any non-numeric string passed as OLLAMA_RETRY_BACKOFF_BASE,
    _load_backend_config() SHALL raise ConfigurationError.
    """
    with pytest.raises(ConfigurationError) as exc_info:
        _call_with_env(OLLAMA_RETRY_BACKOFF_BASE=bad_value)
    msg = str(exc_info.value)
    assert "OLLAMA_RETRY_BACKOFF_BASE" in msg
    assert bad_value in msg or repr(bad_value) in msg


# ─── Property 13 ─────────────────────────────────────────────────────────────

@given(bad_backend=invalid_backend)
@settings(max_examples=100)
def test_property13_invalid_backend_raises(bad_backend):
    """
    For any string other than 'ollama' passed as LLM_BACKEND,
    _load_backend_config() SHALL raise ConfigurationError.
    """
    with pytest.raises(ConfigurationError) as exc_info:
        _call_with_env(LLM_BACKEND=bad_backend)
    msg = str(exc_info.value)
    assert "ollama" in msg


# ─── Unit tests: valid env vars produce correct types ─────────────────────────

class TestBackendConfigValidEnvVars(unittest.TestCase):

    def test_defaults_produce_correct_types(self):
        cfg = _call_with_env()
        self.assertIsInstance(cfg.llm_backend, str)
        self.assertIsInstance(cfg.ollama_base_url, str)
        self.assertIsInstance(cfg.ollama_extraction_model, str)
        self.assertIsInstance(cfg.ollama_grounding_model, str)
        self.assertIsInstance(cfg.ollama_timeout_seconds, int)
        self.assertIsInstance(cfg.ollama_max_retries, int)
        self.assertIsInstance(cfg.ollama_retry_backoff_base, float)

    def test_default_values(self):
        cfg = _call_with_env()
        self.assertEqual(cfg.llm_backend, "ollama")
        self.assertEqual(cfg.ollama_base_url, "http://localhost:11434")
        self.assertEqual(cfg.ollama_extraction_model, "llama3")
        self.assertEqual(cfg.ollama_grounding_model, "llama3")
        self.assertEqual(cfg.ollama_timeout_seconds, 30)
        self.assertEqual(cfg.ollama_max_retries, 3)
        self.assertAlmostEqual(cfg.ollama_retry_backoff_base, 2.0)

    def test_ollama_backend_accepted(self):
        cfg = _call_with_env(LLM_BACKEND="ollama")
        self.assertEqual(cfg.llm_backend, "ollama")

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

    def test_openai_backend_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(LLM_BACKEND="openai")
        self.assertIn("ollama", str(ctx.exception))

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
        with self.assertRaises(ConfigurationError) as ctx:
            _call_with_env(OLLAMA_TIMEOUT_SECONDS="3.5")
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", str(ctx.exception))
