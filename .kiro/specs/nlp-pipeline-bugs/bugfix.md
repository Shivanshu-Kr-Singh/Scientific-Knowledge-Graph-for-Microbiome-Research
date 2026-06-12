# Bugfix Requirements Document

## Introduction

Two bugs are causing Layer 2 (NLP Processing Pipeline) to fail or run significantly slower than expected when processing `PaperRecord` objects.

**Bug 1** is a hard crash: `EnrichedPaperRecord(**paper.model_dump(), ..., full_text=full_text, ...)` in `nlp/pipeline.py` passes `full_text` twice — once through the unpacked `**paper.model_dump()` dict (which includes `PaperRecord.full_text`) and once as an explicit keyword argument. Python raises `TypeError: got multiple values for keyword argument 'full_text'`, aborting processing for every paper.

**Bug 2** is a performance degradation: when the Ollama LLM is slow (CPU inference on large contexts), each paper blocks for up to `timeout × (max_retries + 1)` seconds (e.g. 120 s × 2 = 240 s) before returning an empty result. The error is handled gracefully, but the timeout value of 120 s (read from `OLLAMA_TIMEOUT_SECONDS` with a default of `30 s`, currently set to `120 s` in `.env`) is too short for the model being used, causing repeated blocking delays across the whole pipeline batch.

---

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `pipeline.process_one()` constructs `EnrichedPaperRecord(**paper.model_dump(), ..., full_text=full_text, ...)` AND the base `PaperRecord` object has a `full_text` field (declared at `models.py` line 65) THEN the system raises `TypeError: EnrichedPaperRecord() got multiple values for keyword argument 'full_text'`, crashing the per-paper processing step.

1.2 WHEN `EnrichedPaperRecord` re-declares `full_text: str | None = None` as its own field AND `EnrichedPaperRecord` already inherits `full_text` from `PaperRecord` THEN the system contains a redundant field declaration that contributes to ambiguity in field resolution during model construction.

1.3 WHEN the Ollama LLM does not respond within `OLLAMA_TIMEOUT_SECONDS` (currently 120 s) THEN the system retries up to `OLLAMA_MAX_RETRIES` times, blocking the pipeline for `120 s × (retries + 1)` seconds per paper before returning an empty extraction result.

1.4 WHEN the Ollama timeout fires and all retries are exhausted THEN the system silently returns an empty NER result (no entities extracted from the LLM tier) and logs a warning, but the pipeline continues — the empty result is swallowed and the paper proceeds with only Tier 1/2 entities.

### Expected Behavior (Correct)

2.1 WHEN `pipeline.process_one()` constructs `EnrichedPaperRecord` with `**paper.model_dump()` AND passes `full_text` as an override keyword argument THEN the system SHALL construct the record without raising a `TypeError`, with `full_text` taking the value fetched by the full-text orchestrator (not the value from the base `PaperRecord.model_dump()`).

2.2 WHEN `EnrichedPaperRecord` is defined THEN the system SHALL NOT re-declare `full_text` as its own field, since it is already inherited from `PaperRecord` and the override happens at the call site.

2.3 WHEN the Ollama LLM is invoked for a paper AND the model is running slowly on CPU THEN the system SHALL wait long enough for a valid response to arrive before timing out, avoiding unnecessary retries and empty results.

2.4 WHEN `OLLAMA_TIMEOUT_SECONDS` is set to a sufficiently high value in the environment (e.g. `600` s) THEN the system SHALL respect that value and use it as the per-request timeout, reducing the number of spurious timeouts under slow CPU inference.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `pipeline.process_one()` is called with a `PaperRecord` that has `full_text = None` THEN the system SHALL CONTINUE TO construct a valid `EnrichedPaperRecord` with `full_text` resolved to the string fetched by `FullTextOrchestrator` (or an empty string if unavailable).

3.2 WHEN `pipeline.process_one()` is called with a `PaperRecord` that has a non-`None` `full_text` (pre-populated from PMC XML enrichment) THEN the system SHALL CONTINUE TO construct a valid `EnrichedPaperRecord` — the `full_text` passed explicitly SHALL take precedence over the base record's value.

3.3 WHEN all other fields of `PaperRecord` (doi, title, abstract, authors, etc.) are passed through `**paper.model_dump()` THEN the system SHALL CONTINUE TO carry those fields forward unchanged into `EnrichedPaperRecord`.

3.4 WHEN Ollama responds successfully within the timeout window THEN the system SHALL CONTINUE TO return extracted entities from the LLM tier and cache them as before.

3.5 WHEN `OLLAMA_MAX_RETRIES` and `OLLAMA_RETRY_BACKOFF_BASE` are set in the environment THEN the system SHALL CONTINUE TO respect those values for retry count and backoff delay.

3.6 WHEN `OLLAMA_FALLBACK_TO_GEMINI=true` is configured THEN the system SHALL CONTINUE TO fall back to the Gemini backend on timeout or unavailability, unchanged by the timeout value increase.
