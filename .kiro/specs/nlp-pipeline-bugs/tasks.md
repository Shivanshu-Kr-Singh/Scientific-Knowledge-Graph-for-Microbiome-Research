# Implementation Plan

- [ ] 1. Write bug condition exploration tests
  - **Property 1: Bug Condition** — Duplicate `full_text` Keyword Argument Crash
  - **CRITICAL**: These tests MUST FAIL on unfixed code — failure confirms the bugs exist
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: These tests encode the expected behavior — they will validate the fix when they pass after implementation
  - **GOAL**: Surface counterexamples that demonstrate both bugs exist
  - **Test B1a — Null full_text crash**: Construct `PaperRecord(title="t", full_text=None)`, then call `EnrichedPaperRecord(**paper.model_dump(), full_text="fetched")` — assert this raises `TypeError: got multiple values for keyword argument 'full_text'` on unfixed code (from Bug Condition in design: `"full_text" IN paper.model_dump().keys() AND full_text IS PASSED AS explicit_kwarg`)
  - **Test B1b — Non-null full_text crash**: Same as above but with `PaperRecord(full_text="original text")` — assert `TypeError` is raised
  - **Test B2 — Timeout fires**: Use `unittest.mock.patch` to make `requests.post` sleep for `timeout + 1` seconds; assert `OllamaTimeoutError` is raised after all retries and `LLMExtractor.extract()` returns `([], [])`
  - Run all tests on UNFIXED code
  - **EXPECTED OUTCOME**: B1a and B1b raise `TypeError` (confirms Bug 1 exists); B2 confirms timeout behavior
  - Document counterexamples found to confirm root cause
  - Mark task complete when tests are written, run, and failures are documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** — All PaperRecord Fields Forwarded Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe on unfixed code: `PaperRecord(doi="10.1/x", title="T", abstract="A").model_dump(exclude={"full_text"})` returns all fields except `full_text` — record these field names and values
  - Write property-based test using Hypothesis: for any `PaperRecord` generated with arbitrary `doi`, `title`, `abstract`, `authors`, `keywords`, `issn`, `publication_date`, `article_types`, `is_open_access`, `citation_count`, and `mesh_terms` values, assert that `paper.model_dump(exclude={"full_text"}).keys()` contains all expected keys and that each value round-trips correctly
  - This test does NOT call `EnrichedPaperRecord(...)` (which crashes on unfixed code) — it only verifies the `model_dump(exclude={"full_text"})` API works correctly and returns all non-`full_text` fields
  - Verify test passes on UNFIXED code (the preservation of non-`full_text` fields via `model_dump` is already correct; only the call site is broken)
  - Also write: mock-based test that a successful Ollama response (mock returns valid JSON) still results in correct entity extraction — verify on unfixed code to record baseline behavior
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 3. Fix for duplicate `full_text` kwarg crash and Ollama timeout

  - [ ] 3.1 Remove redundant `full_text` re-declaration from `EnrichedPaperRecord`
    - Open `nlp/enriched_record.py`
    - Delete the line `full_text: str | None = None` from `EnrichedPaperRecord` (near the `fetch_source` / `fetch_status` block)
    - `full_text` is already inherited from `PaperRecord` — the re-declaration is redundant and contributes to confusion
    - Verify `EnrichedPaperRecord` still has `full_text` accessible (inherited from `PaperRecord`)
    - _Bug_Condition: isBugCondition_B1 — `"full_text"` re-declared in subclass_
    - _Expected_Behavior: `EnrichedPaperRecord.model_fields` contains `full_text` exactly once, from `PaperRecord`_
    - _Preservation: all other `EnrichedPaperRecord` fields (`fetch_source`, `fetch_status`, `study_design`, etc.) unchanged_
    - _Requirements: 2.2, 3.3_

  - [ ] 3.2 Exclude `full_text` from `model_dump()` spread in `process_one()`
    - Open `nlp/pipeline.py`, locate the `EnrichedPaperRecord(...)` construction in `process_one()`
    - Change `**paper.model_dump()` to `**paper.model_dump(exclude={"full_text"})`
    - The explicit `full_text=full_text` kwarg (already present at the call site) will now correctly set the field to the value fetched by `FullTextOrchestrator`
    - No other changes to the call site are needed
    - _Bug_Condition: isBugCondition_B1 — `"full_text" IN paper.model_dump().keys()` AND passed again as explicit kwarg_
    - _Expected_Behavior: `EnrichedPaperRecord` is constructed without TypeError; `result.full_text == fetched_full_text`_
    - _Preservation: all other fields from `paper.model_dump()` are still forwarded unchanged_
    - _Requirements: 2.1, 3.1, 3.2, 3.3_

  - [ ] 3.3 Increase `OLLAMA_TIMEOUT_SECONDS` in `.env`
    - Open `.env`
    - Change `OLLAMA_TIMEOUT_SECONDS=120` to `OLLAMA_TIMEOUT_SECONDS=600`
    - Optionally set `OLLAMA_MAX_RETRIES=1` (or `0` for max throughput) to bound worst-case blocking per paper
    - Document the rationale in a comment: slow CPU inference on `qwen2.5:1.5b` with large prompts exceeds 120 s
    - _Bug_Condition: isBugCondition_B2 — `ollama_timeout_seconds < llm_response_time`_
    - _Expected_Behavior: LLM responds within new timeout window; no spurious `OllamaTimeoutError` for slow-but-valid responses_
    - _Preservation: `OLLAMA_FALLBACK_TO_GEMINI`, `OLLAMA_RETRY_BACKOFF_BASE`, and all other env vars unchanged_
    - _Requirements: 2.3, 2.4, 3.4, 3.5, 3.6_

  - [ ] 3.4 Verify bug condition exploration tests now pass
    - **Property 1: Expected Behavior** — No Duplicate Keyword Argument, Timeout Respects Config
    - **IMPORTANT**: Re-run the SAME tests from task 1 — do NOT write new tests
    - The tests from task 1 encode the expected behavior
    - When these tests pass, it confirms the expected behavior is satisfied
    - Re-run B1a, B1b, and B2 tests from step 1
    - **EXPECTED OUTCOME**: B1a and B1b now succeed without `TypeError`; B2 continues to correctly demonstrate timeout behavior with the mock (behavior of `OllamaTimeoutError` raising is preserved)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 3.5 Verify preservation tests still pass
    - **Property 2: Preservation** — Fields Forwarded, Successful Extraction Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run all preservation property tests from step 2
    - **EXPECTED OUTCOME**: All preservation tests PASS (confirms no regressions — all non-`full_text` fields still forwarded, successful Ollama path still returns entities correctly)
    - Confirm all tests still pass after fix (no regressions)

- [ ] 4. Checkpoint — Ensure all tests pass
  - Run the full test suite (or at minimum the tests written in steps 1 and 2)
  - Confirm `process_one()` constructs `EnrichedPaperRecord` without error for papers with `full_text=None` and `full_text="..."` alike
  - Confirm `BACKEND_CONFIG.ollama_timeout_seconds == 600` when `.env` has `OLLAMA_TIMEOUT_SECONDS=600`
  - Ensure all tests pass; ask the user if questions arise
