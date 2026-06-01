# Implementation Plan: Ollama LLM Integration

## Overview

Replace the Google Gemini cloud API backend with local Ollama inference. The implementation proceeds in layers: shared infrastructure first (`BackendConfig`, `_JsonFileCache`), then the Ollama client, then the updated extractor and grounder, and finally the test suite. All existing interfaces (`extract()` and `resolve()`) are preserved so Layer 3 callers require no changes.

## Tasks

- [x] 1. Add `BackendConfig` and `ConfigurationError` to `config.py`
  - [x] 1.1 Implement `ConfigurationError` exception class and `BackendConfig` frozen dataclass
    - Add `ConfigurationError(Exception)` to `config.py`
    - Add `BackendConfig` dataclass with all ten typed fields as specified in the design
    - Implement `_load_backend_config()` with full validation: enum check for `LLM_BACKEND`, int/float parsing for numeric fields, `GEMINI_API_KEY` presence checks
    - Assign `BACKEND_CONFIG: BackendConfig = _load_backend_config()` at module level
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 1.2 Write property tests for `BackendConfig` validation
    - **Property 12: BackendConfig raises ConfigurationError for non-numeric env vars**
    - **Validates: Requirements 1.2**
    - **Property 13: BackendConfig raises ConfigurationError for invalid LLM_BACKEND values**
    - **Validates: Requirements 1.3**
    - Place tests in `semantic/tests/test_backend_config.py`

- [x] 2. Implement `semantic/_cache.py` — `_JsonFileCache` shared helper
  - [x] 2.1 Implement `_JsonFileCache` with atomic write support
    - Create `semantic/_cache.py`
    - Implement `__init__(self, path: Path)`, `load() -> dict`, and `save(data: dict) -> None`
    - `load()` must return `{}` on missing file or invalid JSON without raising
    - `save()` must use the `tmp → os.replace()` atomic write pattern
    - _Requirements: 6.3, 6.4, 7.4_

  - [x] 2.2 Write unit tests for `_JsonFileCache`
    - Test: missing file returns `{}`; invalid JSON returns `{}`; save then load round-trip; tmp file is cleaned up after atomic write
    - Place tests in `semantic/tests/test_json_file_cache.py`
    - _Requirements: 6.3, 6.4, 7.4_

- [x] 3. Implement `semantic/ollama_client.py`
  - [x] 3.1 Implement `OllamaUnavailableError`, `OllamaTimeoutError`, and `OllamaClient`
    - Create `semantic/ollama_client.py`
    - Implement `OllamaUnavailableError(message, attempts)` and `OllamaTimeoutError(timeout_seconds)`
    - Implement `OllamaClient.__init__(config: BackendConfig)` and `OllamaClient.generate(model, prompt) -> str`
    - POST to `{base_url}/api/generate` with `stream: false`, `format: "json"`
    - Implement retry loop with exponential backoff: `min(base ** attempt, base ** max_retries)`; log each failure at WARNING level
    - Raise `OllamaTimeoutError` when all attempts time out; raise `OllamaUnavailableError` otherwise
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 3.2 Write property test for retry attempt count
    - **Property 14: OllamaClient makes at most MAX_RETRIES + 1 total attempts**
    - **Validates: Requirements 10.2, 2.3**
    - Place tests in `semantic/tests/test_ollama_client.py`

  - [x] 3.3 Write unit tests for `OllamaClient`
    - Test: HTTP 200 with valid body returns response string; HTTP 500 triggers retry; timeout triggers retry; correct error types raised; backoff formula correctness
    - Place tests in `semantic/tests/test_ollama_client.py`
    - _Requirements: 2.1–2.7, 10.1–10.6_

  - [x] 3.4 Implement `check_ollama_health()` function
    - Add `check_ollama_health(config: BackendConfig | None = None) -> bool` to `semantic/ollama_client.py`
    - GET `{base_url}/api/tags` with 10-second timeout; verify both extraction and grounding model names appear in the `models` array
    - Return `True` and log INFO on success; return `False` and log ERROR on failure
    - If `LLM_BACKEND == "gemini"`, skip probe and return `True` immediately
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 3.5 Write unit tests for `check_ollama_health()`
    - Test: both models present → True; one model missing → False with pull hint; connection error → False; `LLM_BACKEND=gemini` → True without HTTP call
    - Place tests in `semantic/tests/test_health_check.py`
    - _Requirements: 9.1–9.7_

- [x] 4. Checkpoint — core infrastructure complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Update `semantic/llm_extractor.py` to route to Ollama or Gemini
  - [x] 5.1 Refactor `LLMExtractor` to use `BackendConfig`, `_JsonFileCache`, and `OllamaClient`
    - Replace module-level Gemini client with lazy `_get_gemini_client()` helper
    - Replace `load_cache()` / `save_cache()` with `_JsonFileCache` instance at `semantic/cache/llm_extract_cache.json`
    - In `extract()`: guard empty/whitespace/None input → return `([], [])`; check cache by MD5 key; route to Ollama or Gemini based on `BACKEND_CONFIG.llm_backend`
    - On Ollama error + `fallback_to_gemini=True`: log WARNING with "activating Gemini fallback", invoke Gemini
    - On Ollama error + `fallback_to_gemini=False`: log ERROR, return `([], [])`
    - On JSON parse failure: log ERROR with raw response, return `([], [])`; do not write to cache
    - On success: write to cache atomically, return `(entities, relations)`
    - Truncate input text to 12,000 characters before prompt construction
    - Update extraction prompt to match design (explicit JSON-only instruction + populated example)
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 7.1, 7.2, 7.3, 7.4, 8.1, 8.3, 8.5, 8.7, 8.8, 11.1, 11.3, 11.5, 14.1, 14.3, 14.5_

  - [x] 5.2 Write property tests for `LLMExtractor`
    - **Property 1: Extraction always returns tuple[list, list]**
    - **Validates: Requirements 12.1, 4.1**
    - **Property 2: Every extracted entity has non-whitespace name and entity_type**
    - **Validates: Requirements 12.2, 3.4**
    - **Property 3: Every extracted relation has confidence in [0.0, 1.0]**
    - **Validates: Requirements 12.3, 3.5**
    - **Property 4: Extraction is idempotent (cache round-trip)**
    - **Validates: Requirements 12.4, 7.5, 4.3**
    - **Property 5: Extraction schema JSON round-trip preserves all fields**
    - **Validates: Requirements 12.5**
    - **Property 11: Whitespace-only and empty text returns empty lists without LLM call**
    - **Validates: Requirements 4.2**
    - Place tests in `semantic/tests/test_llm_extractor.py`

  - [x] 5.3 Write unit tests for `LLMExtractor`
    - Test: empty text → `([], [])`; cache hit skips Ollama call; Ollama error + fallback=True calls Gemini; Ollama error + fallback=False returns `([], [])`; markdown fence stripped with WARNING; Gemini exception returns `([], [])`
    - Place tests in `semantic/tests/test_llm_extractor.py`
    - _Requirements: 4.1–4.8, 7.1–7.6, 8.1, 8.3, 8.5, 8.7, 8.8_

- [x] 6. Update `semantic/llm_grounder.py` to route to Ollama or Gemini
  - [x] 6.1 Refactor `LLMGrounder` to use `BackendConfig`, `_JsonFileCache`, and `OllamaClient`
    - Replace module-level Gemini client with lazy `_get_gemini_client()` helper
    - Add `_JsonFileCache` instance at `semantic/cache/llm_ground_cache.json`
    - In `resolve()`: check grounding cache by MD5 of `name + entity_type`; route to Ollama or Gemini based on `BACKEND_CONFIG.llm_backend`
    - On Ollama error + `fallback_to_gemini=True`: log WARNING with "activating Gemini fallback", invoke Gemini
    - On Ollama error + `fallback_to_gemini=False`: log ERROR, return `{"canonical": entity.name, "ontology": "unknown"}`
    - On JSON parse failure or missing/invalid keys: log ERROR, return fallback dict; do not write to cache
    - Substitute `entity.name` when `canonical` is empty string
    - On success: write to cache atomically, return result
    - Update grounding prompt to match design (explicit JSON-only instruction + populated example)
    - _Requirements: 3.2, 3.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 8.2, 8.4, 8.6, 8.7, 8.8, 11.2, 11.4, 14.2, 14.4_

  - [x] 6.2 Write property tests for `LLMGrounder`
    - **Property 6: Grounding always returns dict with exactly "canonical" and "ontology"**
    - **Validates: Requirements 13.1, 5.1**
    - **Property 7: Grounding is idempotent (cache round-trip)**
    - **Validates: Requirements 13.3, 6.6, 5.2**
    - **Property 8: Grounding canonical is always a non-empty string**
    - **Validates: Requirements 13.2, 3.6**
    - **Property 9: Grounding schema JSON round-trip preserves all fields**
    - **Validates: Requirements 13.4**
    - **Property 10: Invalid LLM grounding response falls back to entity.name**
    - **Validates: Requirements 13.5, 3.2**
    - **Property 15: Grounding cache key is MD5 of name concatenated with entity_type**
    - **Validates: Requirements 6.2**
    - Place tests in `semantic/tests/test_llm_grounder.py`

  - [x] 6.3 Write unit tests for `LLMGrounder`
    - Test: cache hit skips Ollama call; empty canonical substituted with entity.name; Ollama error + fallback=True calls Gemini; invalid JSON returns fallback dict; Gemini exception returns fallback dict
    - Place tests in `semantic/tests/test_llm_grounder.py`
    - _Requirements: 5.1–5.7, 6.1–6.7, 8.2, 8.4, 8.6, 8.7, 8.8_

- [x] 7. Create test infrastructure and `semantic/tests/` package
  - [x] 7.1 Create `semantic/tests/__init__.py` and shared test fixtures
    - Create `semantic/tests/` directory with `__init__.py`
    - Add shared Hypothesis strategies (`non_empty_text`, `whitespace_text`, `candidate_entity`, `extraction_schema`, `grounding_schema`, `invalid_llm_response`, `non_numeric_string`, `invalid_backend`) as a `conftest.py` or shared module
    - _Requirements: 12.1–12.5, 13.1–13.5_

- [x] 8. Final checkpoint — all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The design uses Python throughout — no language selection needed
- `_JsonFileCache` is a shared helper; implement it before the extractor and grounder
- `BackendConfig` is imported by all other modules; implement it first
- Hypothesis strategies are shared across multiple test files — define them in `conftest.py` to avoid duplication
- Property tests use `@settings(max_examples=100)` and the tag format: `# Feature: ollama-llm-integration, Property N: <text>`
- The `requests` library (already in `requirements.txt`) is used for all Ollama HTTP calls — no additional SDK needed
- Lazy Gemini import (`_get_gemini_client()`) ensures `google-genai` is not required when `LLM_BACKEND=ollama`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "7.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4"] },
    { "id": 4, "tasks": ["3.5", "5.1", "6.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "6.2", "6.3"] }
  ]
}
```-=