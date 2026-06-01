# Requirements Document

## Introduction

This document specifies the requirements for the Ollama LLM Integration, the third component of the Scientific Knowledge Graph system for microbiome research. This feature replaces the Google Gemini cloud API backend in `semantic/llm_extractor.py` and `semantic/llm_grounder.py` with a local Ollama inference backend, eliminating cloud API dependencies and API key requirements.

The two components being replaced are:
- **LLM Extractor** (`semantic/llm_extractor.py`): Extracts entities, relations, and evidence metadata from biomedical text, returning structured JSON. Currently uses `gemini-2.0-flash` with a file-based MD5 cache.
- **LLM Grounder** (`semantic/llm_grounder.py`): Resolves entity names to canonical forms and ontology IDs. Currently uses `gemini-2.5-flash` with no caching.

The integration must preserve the existing Python interfaces (`LLMExtractor.extract()` and `LLMGrounder.resolve()`) so that all upstream callers in Layer 3 are unaffected. It must also improve reliability through retry logic, timeout handling, structured JSON output enforcement, and grounding-side caching. An optional Gemini fallback allows graceful degradation when Ollama is unavailable.

## Glossary

- **Ollama_Client**: The component responsible for communicating with the local Ollama HTTP API to submit prompts and receive completions
- **LLM_Extractor**: The component that accepts biomedical text and returns a list of `CandidateEntity` objects and a list of `CandidateRelation` objects by invoking an LLM
- **LLM_Grounder**: The component that accepts a `CandidateEntity` and returns a dict containing `canonical` (str) and `ontology` (str) fields by invoking an LLM
- **Ollama_Model**: A named model available on the local Ollama server (e.g., `llama3`, `mistral`, `phi3`, `gemma2`)
- **Extraction_Cache**: The persistent file-based JSON cache keyed by MD5 hash of input text, storing LLM extraction results to avoid redundant inference calls
- **Grounding_Cache**: The persistent file-based JSON cache keyed by MD5 hash of entity name and type, storing LLM grounding results to avoid redundant inference calls
- **Structured_JSON_Output**: LLM output constrained to valid JSON by using Ollama's `format: "json"` parameter, eliminating the need for markdown fence stripping
- **Retry_Policy**: The configuration governing how many times a failed LLM call is retried and the backoff delay between attempts
- **Timeout**: The maximum number of seconds the Ollama_Client will wait for a response before treating the call as failed
- **Gemini_Fallback**: The optional secondary backend using the `google-genai` SDK that is invoked when Ollama is unavailable and fallback is enabled
- **Backend_Config**: The configuration object (sourced from environment variables or `config.py`) that controls which LLM backend is active, the Ollama base URL, the model name, retry settings, timeout, and fallback behavior
- **Health_Check**: A lightweight probe that verifies the Ollama server is reachable and the configured model is available before processing begins
- **CandidateEntity**: The Pydantic model defined in `semantic/candidate_store.py` with fields `name`, `entity_type`, `canonical`, `ontology`, `ontology_id`, and `grounded`
- **CandidateRelation**: The Pydantic model defined in `semantic/candidate_store.py` with fields `subject`, `predicate`, `object`, and `confidence`
- **Extraction_Schema**: The JSON schema for LLM extraction output containing `entities` (list), `relations` (list), and `evidence` (dict) fields
- **Grounding_Schema**: The JSON schema for LLM grounding output containing `canonical` (str) and `ontology` (str) fields

## Requirements

### Requirement 1: Ollama Backend Configuration

**User Story:** As a system operator, I want all Ollama connection settings to be controlled through environment variables and `config.py`, so that I can switch models, adjust timeouts, and toggle fallback behavior without modifying source code.

#### Acceptance Criteria

1. THE Backend_Config SHALL read the following environment variables at import time: `LLM_BACKEND` (accepted values: `"ollama"` or `"gemini"`, default `"ollama"`), `OLLAMA_BASE_URL` (default `"http://localhost:11434"`), `OLLAMA_EXTRACTION_MODEL` (default `"llama3"`), `OLLAMA_GROUNDING_MODEL` (default `"llama3"`), `OLLAMA_TIMEOUT_SECONDS` (integer ≥ 1, default `30`), `OLLAMA_MAX_RETRIES` (integer ≥ 0, default `3`), `OLLAMA_RETRY_BACKOFF_BASE` (numeric ≥ 1, default `2`), and `OLLAMA_FALLBACK_TO_GEMINI` (accepted values: `"true"` or `"false"`, default `"false"`)
2. THE Backend_Config SHALL expose all settings as typed attributes (str, int, float, bool) so that callers do not perform string parsing themselves; IF any numeric env var contains a non-numeric value (e.g., `OLLAMA_TIMEOUT_SECONDS="abc"`), THE Backend_Config SHALL raise a `ConfigurationError` at import time identifying the variable name and the invalid value
3. WHEN `LLM_BACKEND` is set to a value outside the accepted set `{"ollama", "gemini"}`, THE Backend_Config SHALL raise a `ConfigurationError` at import time with a message listing the accepted values
4. WHEN `LLM_BACKEND` is set to `"gemini"` and `GEMINI_API_KEY` is absent from the environment, THE Backend_Config SHALL raise a `ConfigurationError` at import time with a message identifying the missing variable
5. WHEN `OLLAMA_FALLBACK_TO_GEMINI` is `"true"` and `GEMINI_API_KEY` is absent from the environment, THE Backend_Config SHALL raise a `ConfigurationError` at import time with a message identifying the missing variable
6. THE Backend_Config SHALL be importable from `config.py` so that all other modules obtain settings from a single source

### Requirement 2: Ollama Client

**User Story:** As a developer, I want a dedicated Ollama client component that handles HTTP communication, JSON mode enforcement, retries, and timeouts, so that LLM_Extractor and LLM_Grounder do not contain duplicated networking logic.

#### Acceptance Criteria

1. THE Ollama_Client SHALL send POST requests to `{OLLAMA_BASE_URL}/api/generate` with a JSON body containing `model`, `prompt`, `stream: false`, and `format: "json"` to enforce Structured_JSON_Output
2. WHEN the Ollama server returns HTTP 200, THE Ollama_Client SHALL parse the `response` field from the response body and return it as a string; IF the response body is missing the `response` field or is not valid JSON, THE Ollama_Client SHALL treat the call as a failed attempt and apply the Retry_Policy
3. WHEN the Ollama server returns a non-200 HTTP status code or a network error occurs, THE Ollama_Client SHALL retry the request up to `OLLAMA_MAX_RETRIES` additional times after the initial attempt, with exponential backoff of `min(OLLAMA_RETRY_BACKOFF_BASE ^ attempt_number, OLLAMA_RETRY_BACKOFF_BASE ^ OLLAMA_MAX_RETRIES)` seconds between attempts
4. WHEN all retry attempts are exhausted without a successful response, THE Ollama_Client SHALL raise an `OllamaUnavailableError` containing the last error message and the number of attempts made
5. WHEN a request to the Ollama server exceeds `OLLAMA_TIMEOUT_SECONDS`, THE Ollama_Client SHALL treat the request as a failed attempt and apply the Retry_Policy; THE Ollama_Client SHALL NOT treat a timed-out request as successful under any circumstance
6. WHEN all retry attempts time out, THE Ollama_Client SHALL raise an `OllamaTimeoutError` containing the configured timeout value; no backoff is applied after the final attempt
7. THE Ollama_Client SHALL log each retry attempt at WARNING level with: attempt number, total allowed attempts, model name, and the error that triggered the retry

### Requirement 3: Structured JSON Output Parsing

**User Story:** As a developer, I want LLM responses to be parsed into validated Python objects, so that downstream components always receive well-formed data regardless of model output variations.

#### Acceptance Criteria

1. WHEN the Ollama_Client returns a response string, THE LLM_Extractor SHALL attempt to parse it as JSON; IF parsing fails (invalid JSON or missing required top-level keys `entities`, `relations`, `evidence`), THE LLM_Extractor SHALL return `([], [])` and log the raw response string at ERROR level
2. WHEN the Ollama_Client returns a response string, THE LLM_Grounder SHALL attempt to parse it as JSON; IF parsing fails (invalid JSON or missing required keys `canonical` or `ontology`, or either value is not a string), THE LLM_Grounder SHALL return `{"canonical": entity.name, "ontology": "unknown"}` and log the raw response string at ERROR level
3. IF a response string begins with a markdown code fence (e.g., ` ```json ` or ` ``` `), THE component SHALL strip the opening and closing fences, log a WARNING, and then pass the stripped string to the JSON parser; IF the JSON parser fails after stripping, THE component SHALL apply the appropriate fallback (`([], [])` for LLM_Extractor, `{"canonical": entity.name, "ontology": "unknown"}` for LLM_Grounder)
4. WHEN the parsed response contains an `entities` field whose value is a list, THE LLM_Extractor SHALL construct one `CandidateEntity` per list element using the `name` field (defaulting to `""` if absent or not a string) and the `type` field (defaulting to `"unknown"` if absent or not a string); IF the `entities` field is absent or not a list, THE LLM_Extractor SHALL treat it as an empty list
5. WHEN the parsed response contains a `relations` field whose value is a list, THE LLM_Extractor SHALL construct one `CandidateRelation` per list element using `subject`, `predicate`, `object` (each defaulting to `""` if absent or not a string) and `confidence` (defaulting to `0.8` if absent or not a number); IF the `relations` field is absent or not a list, THE LLM_Extractor SHALL treat it as an empty list
6. WHEN the parsed Grounding_Schema response contains a `canonical` field that is an empty string, THE LLM_Grounder SHALL substitute `entity.name` for the `canonical` value before returning the result

### Requirement 4: LLM Extractor Interface Preservation

**User Story:** As a Layer 3 developer, I want `LLMExtractor.extract(text)` to return the same types as before the migration, so that no calling code needs to change.

#### Acceptance Criteria

1. THE LLM_Extractor SHALL expose a method `extract(text: str) -> tuple[list[CandidateEntity], list[CandidateRelation]]` with the identical signature as the current implementation
2. WHEN `text` is empty (zero-length or whitespace-only) or `None`, THE LLM_Extractor SHALL return `([], [])` without making any LLM call
3. WHEN the Extraction_Cache contains a non-null entry whose key equals the MD5 hash of `text`, THE LLM_Extractor SHALL return the cached result without invoking the Ollama_Client or Gemini_Fallback
4. WHEN the Extraction_Cache does not contain a valid entry for `text`, THE LLM_Extractor SHALL invoke the Ollama_Client; IF the Ollama_Client returns a parseable response, THE LLM_Extractor SHALL store the parsed result in the Extraction_Cache and return it; IF parsing fails, THE LLM_Extractor SHALL return `([], [])` and log an ERROR without storing anything in the cache
5. WHEN `LLM_BACKEND` is `"ollama"` and the Ollama_Client raises `OllamaUnavailableError` or `OllamaTimeoutError` and `OLLAMA_FALLBACK_TO_GEMINI` is `"true"`, THE LLM_Extractor SHALL invoke the Gemini_Fallback and log a WARNING that includes the error type and the phrase "activating Gemini fallback"
6. WHEN `LLM_BACKEND` is `"ollama"` and the Ollama_Client raises `OllamaUnavailableError` or `OllamaTimeoutError` and `OLLAMA_FALLBACK_TO_GEMINI` is `"false"`, THE LLM_Extractor SHALL return `([], [])` and log an ERROR that includes the error type and the error message
7. THE LLM_Extractor SHALL truncate input text to 12,000 characters before constructing the prompt
8. WHEN the Gemini_Fallback raises an exception during extraction, THE LLM_Extractor SHALL return `([], [])` and log an ERROR that includes the exception type and message

### Requirement 5: LLM Grounder Interface Preservation

**User Story:** As a Layer 3 developer, I want `LLMGrounder.resolve(entity)` to return the same dict structure as before the migration, so that no calling code needs to change.

#### Acceptance Criteria

1. THE LLM_Grounder SHALL expose a method `resolve(entity: CandidateEntity) -> dict` that returns a dict containing exactly the keys `"canonical"` (str) and `"ontology"` (str) with no additional keys
2. WHEN the Grounding_Cache contains an entry whose value contains both `"canonical"` (str) and `"ontology"` (str) keys and whose cache key equals the MD5 hash of `entity.name + entity.entity_type`, THE LLM_Grounder SHALL return that cached dict without invoking the Ollama_Client or Gemini_Fallback
3. WHEN the Grounding_Cache does not contain a valid entry, THE LLM_Grounder SHALL invoke the Ollama_Client; IF the Ollama_Client returns a parseable response, THE LLM_Grounder SHALL store the parsed result in the Grounding_Cache and return it
4. IF the Ollama_Client returns a response that fails JSON parsing or schema validation, THE LLM_Grounder SHALL return `{"canonical": entity.name, "ontology": "unknown"}` and log an ERROR without storing anything in the cache
5. IF the Ollama_Client raises `OllamaUnavailableError` or `OllamaTimeoutError` and `OLLAMA_FALLBACK_TO_GEMINI` is `"true"`, THE LLM_Grounder SHALL invoke the Gemini_Fallback and log a WARNING that includes the error type and the phrase "activating Gemini fallback"
6. IF the Ollama_Client raises `OllamaUnavailableError` or `OllamaTimeoutError` and `OLLAMA_FALLBACK_TO_GEMINI` is `"false"`, THE LLM_Grounder SHALL return `{"canonical": entity.name, "ontology": "unknown"}` and log an ERROR that includes the error type and the error message
7. WHEN the Gemini_Fallback raises an exception during grounding, THE LLM_Grounder SHALL return `{"canonical": entity.name, "ontology": "unknown"}` and log an ERROR that includes the exception type and message

### Requirement 6: Grounding Cache

**User Story:** As a system operator, I want the LLM Grounder to cache results the same way the LLM Extractor does, so that repeated grounding calls for the same entity do not incur redundant inference overhead.

#### Acceptance Criteria

1. THE Grounding_Cache SHALL be stored as a JSON file at `semantic/cache/llm_ground_cache.json`, mirroring the location convention of the existing `llm_extract_cache.json`
2. THE Grounding_Cache SHALL use the MD5 hash of the UTF-8 encoded concatenation of `entity.name` and `entity.entity_type` (in that order, with no separator) as the cache key
3. WHEN a grounding result is stored in the Grounding_Cache, THE LLM_Grounder SHALL write the updated cache to disk such that the cache file is never left in a partially-written or corrupted state on process interruption; this applies both to updates and to initial file creation
4. WHEN the Grounding_Cache file does not exist or contains invalid JSON, THE LLM_Grounder SHALL treat the cache as empty and proceed without raising an exception
5. THE Grounding_Cache SHALL store the parsed dict (with `"canonical"` and `"ontology"` keys) as the cache value, not the raw LLM response string
6. WHEN the Grounding_Cache contains a valid entry for an entity, THE LLM_Grounder SHALL return that cached dict on all subsequent calls for the same entity without invoking any LLM backend
7. WHEN the Grounding_Cache contains an entry whose value is missing `"canonical"` or `"ontology"` keys or whose values are not strings, THE LLM_Grounder SHALL treat that entry as a cache miss and invoke the LLM backend

### Requirement 7: Extraction Cache Preservation

**User Story:** As a system operator, I want the existing extraction cache to continue working after the migration, so that previously cached Gemini results are not invalidated and inference is not repeated for already-processed texts.

#### Acceptance Criteria

1. THE Extraction_Cache SHALL continue to use `semantic/cache/llm_extract_cache.json` as the storage path, unchanged from the current implementation
2. THE Extraction_Cache SHALL continue to use the MD5 hash of the input text as the cache key, unchanged from the current implementation
3. WHEN the Extraction_Cache contains an entry produced by the Gemini backend, THE LLM_Extractor SHALL return that cached entry without re-invoking any LLM backend, because the cache stores parsed output (entity/relation lists), not backend-specific metadata
4. WHEN a new extraction result is stored in the Extraction_Cache, THE LLM_Extractor SHALL write the updated cache to disk such that the cache file is never left in a partially-written or corrupted state on process interruption
5. WHEN `extract(text)` is called with a non-empty text (containing at least one non-whitespace character) and the cache already contains a valid entry for that text, THE LLM_Extractor SHALL return a list with the same elements in the same order as the first call
6. WHEN the Extraction_Cache contains an entry whose value is malformed (not a dict with the expected structure), THE LLM_Extractor SHALL treat that entry as a cache miss and invoke the LLM backend

### Requirement 8: Gemini Fallback Backend

**User Story:** As a system operator, I want an optional Gemini fallback so that the pipeline continues to produce results during Ollama outages, without requiring a full backend switch.

#### Acceptance Criteria

1. WHEN `OLLAMA_FALLBACK_TO_GEMINI` is `"true"` and the Ollama_Client raises any exception, THE LLM_Extractor SHALL invoke the Gemini backend using the existing `google-genai` SDK; THE LLM_Extractor SHALL NOT invoke the Gemini backend before an Ollama failure occurs
2. WHEN `OLLAMA_FALLBACK_TO_GEMINI` is `"true"` and the Ollama_Client raises any exception, THE LLM_Grounder SHALL invoke the Gemini backend using the existing `google-genai` SDK; THE LLM_Grounder SHALL NOT invoke the Gemini backend before an Ollama failure occurs
3. WHEN the Gemini_Fallback is invoked by THE LLM_Extractor, it SHALL use the model name from `GEMINI_EXTRACTION_MODEL` environment variable (default `"gemini-2.0-flash"`)
4. WHEN the Gemini_Fallback is invoked by THE LLM_Grounder, it SHALL use the model name from `GEMINI_GROUNDING_MODEL` environment variable (default `"gemini-2.5-flash"`)
5. WHEN the Gemini_Fallback raises an exception during extraction, THE LLM_Extractor SHALL return `([], [])` and log an ERROR that includes the exception type and message
6. WHEN the Gemini_Fallback raises an exception during grounding, THE LLM_Grounder SHALL return `{"canonical": entity.name, "ontology": "unknown"}` and log an ERROR that includes the exception type and message
7. IF `LLM_BACKEND` is set to `"gemini"`, THE LLM_Extractor and THE LLM_Grounder SHALL use the Gemini backend directly without attempting an Ollama call, regardless of the `OLLAMA_FALLBACK_TO_GEMINI` setting
8. WHEN `LLM_BACKEND` is `"gemini"` and the Gemini backend raises an exception during extraction, THE LLM_Extractor SHALL return `([], [])` and log an ERROR; WHEN it raises during grounding, THE LLM_Grounder SHALL return `{"canonical": entity.name, "ontology": "unknown"}` and log an ERROR

### Requirement 9: Ollama Server Health Check

**User Story:** As a system operator, I want a health check that verifies Ollama is reachable and the configured model is available before the pipeline processes any papers, so that misconfiguration is detected early with a clear error message.

#### Acceptance Criteria

1. WHEN `check_ollama_health()` is called, THE Health_Check SHALL send a GET request to `{OLLAMA_BASE_URL}/api/tags` with a connection timeout of 10 seconds
2. WHEN the GET request succeeds with HTTP 200, THE Health_Check SHALL verify that both `OLLAMA_EXTRACTION_MODEL` and `OLLAMA_GROUNDING_MODEL` appear as exact string matches against the `name` field of each entry in the `models` array of the response body
3. WHEN the Health_Check succeeds (both models found), THE Health_Check SHALL return `True` and log an INFO message containing the model names and the base URL
4. WHEN the Health_Check fails because the Ollama server is unreachable or returns a non-200 status, THE Health_Check SHALL return `False` and log an ERROR message containing the base URL and the connection error or HTTP status code
5. WHEN the Health_Check fails because one or more required models are not in the `models` list, THE Health_Check SHALL return `False` and log an ERROR message listing all missing model names and a hint to run `ollama pull {model_name}` for each missing model
6. THE Health_Check SHALL be callable as a standalone function `check_ollama_health() -> bool` importable from `semantic/ollama_client.py`
7. IF `LLM_BACKEND` is `"gemini"`, THE Health_Check SHALL skip the Ollama probe and return `True` without making any network call

### Requirement 10: Retry and Timeout Behavior

**User Story:** As a system operator, I want failed LLM calls to be retried with exponential backoff, so that transient Ollama errors (model loading, temporary overload) do not cause permanent extraction or grounding failures.

#### Acceptance Criteria

1. WHEN the Ollama_Client receives a network error or non-200 HTTP response, THE Ollama_Client SHALL wait `min(OLLAMA_RETRY_BACKOFF_BASE ^ attempt_number, OLLAMA_RETRY_BACKOFF_BASE ^ OLLAMA_MAX_RETRIES)` seconds before the next attempt, where `attempt_number` starts at 1; this caps the maximum backoff delay at `OLLAMA_RETRY_BACKOFF_BASE ^ OLLAMA_MAX_RETRIES` seconds
2. THE Ollama_Client SHALL make at most `OLLAMA_MAX_RETRIES + 1` total attempts (1 initial + up to `OLLAMA_MAX_RETRIES` retries)
3. WHEN a request times out after `OLLAMA_TIMEOUT_SECONDS`, THE Ollama_Client SHALL count it as a failed attempt and apply the same backoff before retrying; no backoff is applied after the final attempt
4. THE Ollama_Client SHALL log each failed attempt at WARNING level including: the attempt number, the total allowed attempts, the Python exception class name, and the backoff delay in seconds (0 for the final attempt)
5. WHEN the final attempt fails with a network error or non-200 HTTP response, THE Ollama_Client SHALL raise `OllamaUnavailableError` without further retrying
6. WHEN the final attempt fails with a timeout, THE Ollama_Client SHALL raise `OllamaTimeoutError` without further retrying; the responsibility for invoking the Gemini_Fallback lies with LLM_Extractor and LLM_Grounder, not with Ollama_Client

### Requirement 11: Prompt Design for Structured Output

**User Story:** As a developer, I want the prompts sent to Ollama to be optimized for structured JSON output, so that smaller local models produce well-formed extraction and grounding results reliably.

#### Acceptance Criteria

1. THE LLM_Extractor prompt SHALL instruct the model to return ONLY a JSON object with the keys `entities`, `relations`, and `evidence`, with no preamble, prose sentences, or code fences; the instruction SHALL explicitly state that no text outside the JSON object is permitted
2. THE LLM_Grounder prompt SHALL instruct the model to return ONLY a JSON object with the keys `canonical` and `ontology`, with no preamble, prose sentences, or code fences; the instruction SHALL explicitly state that no text outside the JSON object is permitted
3. THE LLM_Extractor prompt SHALL include a populated example JSON object with representative placeholder values for all required fields (`entities`, `relations`, `evidence`) so that the model can infer the required format from the example
4. THE LLM_Grounder prompt SHALL include a populated example JSON object with representative placeholder values for both required fields (`canonical`, `ontology`) so that the model can infer the required format from the example
5. THE LLM_Extractor SHALL truncate input text to 12,000 characters before embedding it in the prompt

### Requirement 12: Extraction Output Correctness Properties

**User Story:** As a developer, I want property-based tests that verify the structural correctness of extraction output across arbitrary biomedical text inputs, so that regressions in JSON parsing or object construction are caught automatically.

#### Acceptance Criteria

1. FOR ALL non-empty text inputs (containing at least one non-whitespace character), THE LLM_Extractor SHALL return a tuple of exactly two elements where the first element is a list and the second element is a list; this holds even when the LLM backend returns an error or unparseable response
2. FOR ALL non-empty text inputs, every element in the first returned list SHALL be an instance of `CandidateEntity` with a `name` field containing at least one non-whitespace character and an `entity_type` field containing at least one non-whitespace character
3. FOR ALL non-empty text inputs, every element in the second returned list SHALL be an instance of `CandidateRelation` with a `confidence` value in the range [0.0, 1.0] inclusive
4. FOR ALL non-empty text inputs, calling `extract(text)` a second time SHALL return lists with the same number of elements and the same set of entity names (order-insensitive) as the first call
5. FOR ALL JSON strings that are valid Extraction_Schema objects (containing `entities` as a list, `relations` as a list, and `evidence` as a dict), parsing the string to a Python dict, serializing it back to JSON, and parsing again SHALL produce a dict with field-by-field equal values for all keys present in the original

### Requirement 13: Grounding Output Correctness Properties

**User Story:** As a developer, I want property-based tests that verify the structural correctness of grounding output across arbitrary entity inputs, so that regressions in JSON parsing or fallback logic are caught automatically.

#### Acceptance Criteria

1. FOR ALL `CandidateEntity` inputs, THE LLM_Grounder SHALL return a dict containing exactly the keys `"canonical"` and `"ontology"` with string values and no additional keys
2. FOR ALL `CandidateEntity` inputs, the `"canonical"` value in the returned dict SHALL be a non-empty string (containing at least one non-whitespace character)
3. FOR ALL `CandidateEntity` inputs, calling `resolve(entity)` a second time within the same process (where the cache file persists between calls) SHALL return a dict with identical `"canonical"` and `"ontology"` values as the first call
4. FOR ALL JSON strings that are valid Grounding_Schema objects (containing `"canonical"` as a non-empty string and `"ontology"` as a string), parsing the string to a Python dict, serializing it back to JSON, and parsing again SHALL produce a dict with field-by-field equal values for both keys
5. WHEN the LLM returns a response that is not valid JSON or is missing the `"canonical"` key or has a non-string `"canonical"` value for any `CandidateEntity` input, THE LLM_Grounder SHALL return a dict where `"canonical"` equals `entity.name` and `"ontology"` equals `"unknown"`

### Requirement 14: Backward Compatibility and Migration

**User Story:** As a system maintainer, I want the migration from Gemini to Ollama to require zero changes to any code outside `semantic/llm_extractor.py`, `semantic/llm_grounder.py`, and `config.py`, so that Layer 3 graph construction and all other callers are unaffected.

#### Acceptance Criteria

1. THE LLM_Extractor SHALL preserve the `extract(text: str) -> tuple[list[CandidateEntity], list[CandidateRelation]]` method signature without adding required parameters
2. THE LLM_Grounder SHALL preserve the `resolve(entity: CandidateEntity) -> dict` method signature without adding required parameters
3. WHEN `LLM_BACKEND` is set to `"gemini"`, THE LLM_Extractor SHALL return the same types and structures as the pre-migration implementation (a tuple of two lists), use the same Gemini model identifiers (`gemini-2.0-flash` for extraction), and read from and write to the same cache file path (`semantic/cache/llm_extract_cache.json`)
4. WHEN `LLM_BACKEND` is set to `"gemini"`, THE LLM_Grounder SHALL return the same dict structure as the pre-migration implementation (keys `"canonical"` and `"ontology"`), use the same Gemini model identifier (`gemini-2.5-flash`), and read from and write to the same cache file path (`semantic/cache/llm_ground_cache.json`)
5. WHEN the `ollama` Python package is not installed and `LLM_BACKEND` is `"gemini"`, THE System SHALL not raise an `ImportError` for the missing `ollama` package at module import time
6. WHEN `LLM_BACKEND` is `"ollama"` and the `ollama` Python package is not installed, THE System SHALL raise an `ImportError` at module import time with a message instructing the operator to run `pip install ollama`
7. WHEN `LLM_BACKEND` is set to a value outside `{"ollama", "gemini"}`, THE System SHALL raise a `ValueError` at module import time with a message listing the accepted values and the invalid value that was provided
