# NLP Pipeline Bugs — Bugfix Design

## Overview

Two bugs affect the Layer 2 NLP Processing Pipeline (`nlp/pipeline.py` and its dependencies):

**Bug 1 — Duplicate `full_text` kwarg crash**: `process_one()` spreads the base `PaperRecord` via `**paper.model_dump()` and simultaneously passes `full_text=full_text` as an explicit keyword argument. Because `PaperRecord` declares `full_text` as a field, `model_dump()` includes it, causing Python to raise `TypeError: got multiple values for keyword argument 'full_text'`. Fix: exclude `full_text` from the spread (use `model_dump(exclude={"full_text"})`) and remove the redundant `full_text` re-declaration from `EnrichedPaperRecord`.

**Bug 2 — Ollama timeout too short**: The default `OLLAMA_TIMEOUT_SECONDS=30` was overridden to `120` in `.env`, but CPU inference on large prompts still exceeds this, triggering 2 retries × 120 s = 240 s of blocking per paper before a graceful empty-result return. Fix: increase the default timeout in `.env` to a value that accommodates slow CPU inference (e.g. `600` s), and reduce `OLLAMA_MAX_RETRIES` to `1` to cap worst-case blocking at `600 s × 2 = 1200 s` (or configure to `0` retries for maximum throughput at the cost of a single-shot attempt).

---

## Glossary

- **Bug_Condition (C)**: The set of inputs / states that trigger a bug.
- **Property (P)**: The correct behavior asserted to hold for all inputs in C after the fix.
- **Preservation**: All behaviors that must remain identical for inputs where C does **not** hold.
- **`process_one(paper)`**: The method in `nlp/pipeline.py` that orchestrates all NLP modules for a single `PaperRecord`.
- **`model_dump()`**: The Pydantic v2 method that serializes a model instance to a plain dict, including all declared fields.
- **`full_text` collision**: The condition where `PaperRecord.full_text` appears in `model_dump()` output and is also passed as an explicit kwarg to `EnrichedPaperRecord(...)`.
- **`OLLAMA_TIMEOUT_SECONDS`**: The per-request read timeout for the Ollama HTTP client, read from the environment at startup.
- **`OLLAMA_MAX_RETRIES`**: Number of retry attempts after the first failure, read from the environment.

---

## Bug Details

### Bug 1 — Bug Condition: Duplicate `full_text` Keyword Argument

The crash occurs every time `process_one()` runs, because the condition is structural (not data-dependent): `PaperRecord` always declares `full_text`, so `model_dump()` always includes it, and the call site always passes it again explicitly.

**Formal Specification:**
```
FUNCTION isBugCondition_B1(paper, pipeline_call)
  INPUT: paper of type PaperRecord, pipeline_call = EnrichedPaperRecord(**paper.model_dump(), ..., full_text=full_text, ...)
  OUTPUT: boolean

  RETURN "full_text" IN paper.model_dump().keys()
         AND full_text IS PASSED AS explicit_kwarg
END FUNCTION
```

**Examples:**
- `PaperRecord(full_text=None)` → `model_dump()` contains `"full_text": None`; explicit `full_text=""` → TypeError crash.
- `PaperRecord(full_text="PMC full text...")` → `model_dump()` contains `"full_text": "PMC full text..."`; explicit `full_text="fetched text"` → TypeError crash.
- After fix: `model_dump(exclude={"full_text"})` → `"full_text"` not in spread; explicit `full_text=full_text` → no collision, correct override.

---

### Bug 2 — Bug Condition: Ollama Timeout Too Short

The slow-inference timeout condition is data/environment-dependent: it triggers when a paper's text (after prompt construction) exceeds the LLM's fast-path capacity and the model responds after `OLLAMA_TIMEOUT_SECONDS`.

**Formal Specification:**
```
FUNCTION isBugCondition_B2(env_config, llm_response_time)
  INPUT: env_config of type BackendConfig, llm_response_time in seconds
  OUTPUT: boolean

  RETURN env_config.ollama_timeout_seconds < llm_response_time
         AND env_config.llm_backend = "ollama"
END FUNCTION
```

**Examples:**
- `OLLAMA_TIMEOUT_SECONDS=120`, model responds in 150 s → timeout fires, retries, 240 s blocked, empty result returned.
- `OLLAMA_TIMEOUT_SECONDS=120`, model responds in 90 s → no bug, entities returned.
- After fix: `OLLAMA_TIMEOUT_SECONDS=600`, model responds in 150 s → no timeout, entities returned.

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- All other `PaperRecord` fields (`doi`, `title`, `abstract`, `authors`, `keywords`, `journal`, `issn`, `publication_date`, `article_types`, `is_open_access`, `full_text_url`, `pdf_url`, `citation_count`, `mesh_terms`, etc.) must continue to be forwarded via `**paper.model_dump(...)` without modification.
- `EnrichedPaperRecord` must continue to inherit all `PaperRecord` fields correctly (no schema changes to inheritance).
- Successful Ollama responses must continue to be parsed, cached, and returned as `List[NamedEntity]` exactly as before.
- The `OLLAMA_FALLBACK_TO_GEMINI`, `OLLAMA_MAX_RETRIES`, and `OLLAMA_RETRY_BACKOFF_BASE` env vars must continue to control retry and fallback behavior unchanged.
- The `full_text` field on `EnrichedPaperRecord` must continue to hold the value fetched by `FullTextOrchestrator` (not the base record's `full_text`), preserving the override semantics that were clearly intended by the original code.

**Scope:**
All inputs where neither bug condition holds — i.e., papers processed with a working `model_dump(exclude={"full_text"})` call, and Ollama responding within the configured timeout — should be completely unaffected.

---

## Hypothesized Root Cause

### Bug 1

1. **Pydantic `model_dump()` includes all declared fields**: `PaperRecord.full_text` is declared at line 65 of `models.py`. `model_dump()` serializes it unconditionally, so it always appears in the spread dict.
2. **Redundant re-declaration in subclass**: `EnrichedPaperRecord` re-declares `full_text: str | None = None` near the end of the class body. While this doesn't directly cause the TypeError (Pydantic would merge the declarations), it is misleading and should be removed.
3. **No exclusion at call site**: `process_one()` does not exclude `full_text` from `model_dump()` before spreading, so Python sees both the spread `full_text` and the explicit `full_text=full_text` kwarg simultaneously.

### Bug 2

1. **`OLLAMA_TIMEOUT_SECONDS` default is 30 s** in `config.py`; it was overridden to `120` in `.env`, but the model (`qwen2.5:1.5b`) running on CPU can take longer than 120 s on large prompts.
2. **`OLLAMA_MAX_RETRIES=1`** (default 3 is overridden; the logs show "Attempt 1/2" suggesting `max_retries=1`): with 2 total attempts × 120 s each = 240 s blocked per paper.
3. **No adaptive timeout**: the timeout is fixed regardless of prompt length or context size. Longer papers with full text appended produce larger prompts and therefore slower inference.

---

## Correctness Properties

Property 1: Bug Condition — No Duplicate Keyword Argument in EnrichedPaperRecord Construction

_For any_ `PaperRecord` instance `paper` (with `full_text` being `None` or any string value), calling `process_one(paper)` on a fixed pipeline SHALL construct `EnrichedPaperRecord` without raising `TypeError`, and the resulting record's `full_text` field SHALL equal the value returned by `FullTextOrchestrator.fetch()` (not the value from `paper.full_text`).

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation — All Other PaperRecord Fields Forwarded Unchanged

_For any_ `PaperRecord` instance where the bug condition does NOT hold (i.e., after the fix is applied), calling `process_one(paper)` SHALL produce an `EnrichedPaperRecord` where every field that was present in `paper.model_dump()` (excluding `full_text`) retains the same value in the enriched record's corresponding field. The fixed `model_dump(exclude={"full_text"})` call SHALL NOT drop or alter any other field.

**Validates: Requirements 3.1, 3.2, 3.3**

---

## Fix Implementation

### Changes Required

**Fix 1 — Remove duplicate `full_text` kwarg**

**File**: `nlp/pipeline.py`

**Function**: `process_one()`

**Specific Changes**:
1. **Exclude `full_text` from spread**: Change `EnrichedPaperRecord(**paper.model_dump(), ...)` to `EnrichedPaperRecord(**paper.model_dump(exclude={"full_text"}), ...)`. This removes the collision while keeping all other base fields.
2. **Keep explicit override**: The existing `full_text=full_text` kwarg stays — it correctly sets the field to the freshly-fetched full text value.

**File**: `nlp/enriched_record.py`

**Class**: `EnrichedPaperRecord`

**Specific Changes**:
3. **Remove redundant re-declaration**: Delete the `full_text: str | None = None` line from `EnrichedPaperRecord`. The field is already inherited from `PaperRecord`; the explicit keyword arg in `process_one()` sets the value at construction time.

---

**Fix 2 — Increase Ollama timeout in `.env`**

**File**: `.env`

**Specific Changes**:
4. **Increase `OLLAMA_TIMEOUT_SECONDS`**: Change from `120` to `600` (or higher if needed). This gives the LLM up to 10 minutes to respond on CPU before timing out.
5. **Optionally reduce `OLLAMA_MAX_RETRIES`**: Set to `0` or `1` to bound worst-case blocking. With `OLLAMA_TIMEOUT_SECONDS=600` and `OLLAMA_MAX_RETRIES=1`, worst case is `600 × 2 = 1200 s` per paper, but timeouts should now be rare.

---

## Testing Strategy

### Validation Approach

The testing strategy follows the four-phase bug condition methodology: first write exploration tests on **unfixed** code to confirm the bugs exist and understand root causes, then write preservation tests on unfixed code to record the baseline, then apply the fixes, then verify both property tests pass.

---

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples demonstrating each bug BEFORE implementing the fix. Confirm root cause analysis.

**Test Plan**: Construct minimal `PaperRecord` instances with and without `full_text`, invoke `process_one()` (or a trimmed version of the `EnrichedPaperRecord(...)` call), and assert on the crash / timeout behavior.

**Test Cases**:

1. **B1 — Null full_text crash**: Create `PaperRecord(title="t", full_text=None)`, call `EnrichedPaperRecord(**paper.model_dump(), full_text="fetched")` → expect `TypeError` on unfixed code.
2. **B1 — Non-null full_text crash**: Create `PaperRecord(title="t", full_text="original")`, call same construction → expect `TypeError` on unfixed code.
3. **B1 — model_dump() always includes full_text**: Assert `"full_text" in PaperRecord(title="t").model_dump()` → True on both fixed and unfixed code (this is not the bug; the bug is the unchecked spread).
4. **B2 — Timeout fires**: Mock `requests.post` to sleep for `timeout + 1` seconds; assert `OllamaTimeoutError` is raised and pipeline returns `([], [])`.

**Expected Counterexamples**:
- `EnrichedPaperRecord(**PaperRecord(title="t").model_dump(), full_text="x")` raises `TypeError: got multiple values for keyword argument 'full_text'`.
- With `OLLAMA_TIMEOUT_SECONDS=1` and a mock that sleeps 2 s, the client raises `OllamaTimeoutError` after 1 retry, returning empty entities.

---

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL paper WHERE isBugCondition_B1(paper, pipeline_call) DO
  result := process_one_fixed(paper)
  ASSERT result.full_text == fetched_full_text
  ASSERT no TypeError raised
END FOR

FOR ALL (env_config, response_time) WHERE isBugCondition_B2(env_config, response_time) DO
  result := ollama_client_fixed.generate(model, prompt)
  ASSERT result returned within env_config.ollama_timeout_seconds
  ASSERT no spurious timeout when response_time < ollama_timeout_seconds
END FOR
```

---

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL paper WHERE NOT isBugCondition_B1(paper, pipeline_call) DO
  ASSERT fields_excluding_full_text(original_dump) = fields_excluding_full_text(fixed_dump)
END FOR

FOR ALL (env_config, response_time) WHERE NOT isBugCondition_B2(env_config, response_time) DO
  ASSERT ollama_client_original(response_time) = ollama_client_fixed(response_time)
END FOR
```

**Testing Approach**: Property-based testing is recommended for B1 preservation — generate random `PaperRecord` instances with arbitrary field values and assert all non-`full_text` fields are preserved identically in the enriched record after the fix.

**Test Cases**:
1. **Field preservation PBT**: For any `PaperRecord` with arbitrary `doi`, `title`, `abstract`, `authors`, `keywords`, `issn`, `publication_date`, `article_types`, `is_open_access`, `citation_count`, and `mesh_terms` values, assert all those fields appear unchanged in the constructed `EnrichedPaperRecord`.
2. **Successful Ollama path unchanged**: Mock Ollama to return a valid JSON response immediately; assert extracted entities are identical before and after the timeout config change.
3. **Fallback path unchanged**: Set `OLLAMA_FALLBACK_TO_GEMINI=true`, mock Ollama to raise `OllamaTimeoutError`, assert Gemini fallback is invoked exactly as before.

---

### Unit Tests

- Test `EnrichedPaperRecord(**PaperRecord(...).model_dump(exclude={"full_text"}), full_text="x")` succeeds without error.
- Test `EnrichedPaperRecord` no longer declares `full_text` redundantly (field defined exactly once, in `PaperRecord`).
- Test `OllamaClient.generate()` raises `OllamaTimeoutError` when mock sleeps beyond timeout, and returns the correct string when mock responds in time.
- Test `OLLAMA_TIMEOUT_SECONDS` is read from env and honored by `BackendConfig`.

### Property-Based Tests

- **Property 1 (PBT)**: For all `PaperRecord` instances generated by Hypothesis (varying `full_text` = None / any string), `process_one()` (with NLP modules mocked) SHALL return an `EnrichedPaperRecord` with `full_text` equal to the mocked orchestrator return value.
- **Property 2 (PBT)**: For all `PaperRecord` instances generated by Hypothesis (varying all non-`full_text` fields), every field key in `paper.model_dump(exclude={"full_text"})` SHALL appear with the same value in the constructed `EnrichedPaperRecord`.

### Integration Tests

- End-to-end `process_one()` on a real `PaperRecord` fixture with a `full_text` value to confirm no crash and correct field assignment.
- Pipeline batch `process_all()` on 3 synthetic papers to confirm all records are returned (no crash, `errors == 0`).
- Timeout config integration: run `check_ollama_health()` with `OLLAMA_TIMEOUT_SECONDS=600` in env and assert `BackendConfig.ollama_timeout_seconds == 600`.
