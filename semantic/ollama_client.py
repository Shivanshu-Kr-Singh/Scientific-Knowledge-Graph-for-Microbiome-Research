"""
semantic/ollama_client.py — Ollama HTTP client with retry/backoff logic.

Provides:
  - OllamaUnavailableError: raised when all retry attempts fail with network/HTTP errors
  - OllamaTimeoutError: raised when all retry attempts time out
  - OllamaClient: sends prompts to the Ollama /api/generate endpoint
  - check_ollama_health(): probes /api/tags to verify the server and models are available
"""

import logging
import time

import requests

from config import BACKEND_CONFIG, BackendConfig

log = logging.getLogger(__name__)


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class OllamaUnavailableError(Exception):
    """Raised when all retry attempts fail with network or HTTP errors."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class OllamaTimeoutError(Exception):
    """Raised when all retry attempts time out."""

    def __init__(self, timeout_seconds: int) -> None:
        super().__init__(
            f"All Ollama requests timed out after {timeout_seconds}s"
        )
        self.timeout_seconds = timeout_seconds


# ─── OllamaClient ─────────────────────────────────────────────────────────────

class OllamaClient:
    """
    HTTP client for the Ollama /api/generate endpoint.

    Sends POST requests with format:"json" and stream:false to enforce
    structured JSON output. Implements exponential-backoff retry logic
    as specified in the design document.
    """

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

    def generate(self, model: str, prompt: str) -> str:
        """
        POST {base_url}/api/generate with format:"json", stream:false.

        Returns the ``response`` field from the Ollama reply on success.
        Raises OllamaUnavailableError or OllamaTimeoutError after all retries
        are exhausted.

        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 10.1–10.6
        """
        cfg = self._config
        url = f"{cfg.ollama_base_url}/api/generate"
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},  # deterministic output
        }

        total_attempts = cfg.ollama_max_retries + 1
        last_error: Exception | None = None
        last_was_timeout = False

        for attempt in range(total_attempts):
            try:
                response = requests.post(
                    url,
                    json=body,
                    timeout=cfg.ollama_timeout_seconds,
                )
                if response.status_code == 200:
                    data = response.json()
                    if "response" in data:
                        return data["response"]
                    raise ValueError("Missing 'response' field in Ollama reply")
                raise requests.HTTPError(
                    f"HTTP {response.status_code}", response=response
                )

            except requests.Timeout as exc:
                last_error = exc
                last_was_timeout = True

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                last_was_timeout = False

            # ── Backoff / logging ─────────────────────────────────────────────
            if attempt < total_attempts - 1:
                backoff = min(
                    cfg.ollama_retry_backoff_base ** (attempt + 1),
                    cfg.ollama_retry_backoff_base ** cfg.ollama_max_retries,
                )
                log.warning(
                    "Attempt %d/%d failed for model %r: %s. "
                    "Retrying in %.1fs",
                    attempt + 1,
                    total_attempts,
                    model,
                    last_error,
                    backoff,
                )
                time.sleep(backoff)
            else:
                log.warning(
                    "Attempt %d/%d failed for model %r: %s. No more retries.",
                    attempt + 1,
                    total_attempts,
                    model,
                    last_error,
                )

        # ── All attempts exhausted ────────────────────────────────────────────
        if last_was_timeout:
            raise OllamaTimeoutError(cfg.ollama_timeout_seconds)
        raise OllamaUnavailableError(str(last_error), total_attempts)


# ─── Health Check ─────────────────────────────────────────────────────────────

def check_ollama_health(config: BackendConfig | None = None) -> bool:
    """
    Probe the Ollama server to verify it is reachable and both configured
    models are available.

    GET {base_url}/api/tags with a 10-second timeout; verify that both
    ``ollama_extraction_model`` and ``ollama_grounding_model`` appear in the
    ``models`` array of the response.

    Returns True on success, False on any failure.
    If LLM_BACKEND is "gemini", skips the probe and returns True immediately.

    Requirements: 9.1–9.7
    """
    cfg = config if config is not None else BACKEND_CONFIG

    # Req 9.7: skip probe when using Gemini backend
    if cfg.llm_backend == "gemini":
        log.info("LLM_BACKEND=gemini — skipping Ollama health probe")
        return True

    url = f"{cfg.ollama_base_url}/api/tags"
    try:
        response = requests.get(url, timeout=10)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Ollama health check failed: could not reach %s — %s",
            cfg.ollama_base_url,
            exc,
        )
        return False

    if response.status_code != 200:
        log.error(
            "Ollama health check failed: %s returned HTTP %d",
            url,
            response.status_code,
        )
        return False

    try:
        data = response.json()
        available_names = {m["name"] for m in data.get("models", [])}
    except Exception as exc:  # noqa: BLE001
        log.error("Ollama health check: failed to parse /api/tags response — %s", exc)
        return False

    required_models = {
        cfg.ollama_extraction_model,
        cfg.ollama_grounding_model,
    }
    missing = required_models - available_names

    if missing:
        for model_name in sorted(missing):
            log.error(
                "Ollama health check: model %r not found. "
                "Run: ollama pull %s",
                model_name,
                model_name,
            )
        return False

    log.info(
        "Ollama health check passed: models %s available at %s",
        sorted(required_models),
        cfg.ollama_base_url,
    )
    return True
