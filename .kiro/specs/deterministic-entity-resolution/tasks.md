  # Implementation Plan: Deterministic Entity Resolution Pipeline

## Overview

Implement the 12-component Deterministic Entity Resolution Pipeline (Spec 2) in Python 3.8+. The pipeline replaces the Spec 1 exact-match/fuzzy-fallback normalizer with a seven-strategy, fully auditable, deterministic resolution system backed by three separate SQLite databases and Neo4j. All components are implemented as Pydantic-modelled classes with pytest + Hypothesis property-based tests (≥100 examples each).

Implementation order: shared data models and SQLite schemas first, then the three storage backends, then the six resolution strategies, then the orchestration layer (pipeline + ranking + merger), then metrics and override management, and finally integration wiring and shadow mode.

## Tasks

- [x] 1. Set up project structure, shared data models, and SQLite schemas
  - Create `entity_resolution/` package with `__init__.py`, `models.py`, `db_schema.py`, `conftest.py`
  - Implement all Pydantic models: `EntityType`, `NormalizationResult`, `ResolutionResult`, `CandidateScore`, `CanonicalEntityRecord`, `SynonymRecord`, `SynonymProvenance`, `UnresolvedEntity`, `ShadowModeDiscrepancy`, `SynonymConflictRecord`, `ManualOverride`, `BulkImportResult`, `ResolutionRecord`, `AuditQuery`, `CacheEntry`, `RunMetricsSnapshot`, `EntityTypeMetrics`, `MergeLogEntry`, `MergeRollbackEntry`
  - Implement `validate_canonical_id(canonical_id, entity_type)` and `normalize_surface_form(surface_form)` utility functions
  - Create all three SQLite schemas (`canonical_registry.db`, `resolution_cache.db`, `resolution_audit.db`) via `db_schema.py` with `create_all_schemas(base_path)` helper
  - Set up `conftest.py` with Hypothesis profiles (`ci` = 100 examples, `dev` = 20 examples) and shared pytest fixtures (in-memory SQLite databases, fresh pipeline instances)
  - _Requirements: 1.1, 2.1, 3.1, 7.2, 8.4_

  - [x] 1.1 Implement shared Pydantic models and utility functions
    - Write all models listed above in `models.py`
    - Write `validate_canonical_id` and `normalize_surface_form` in `utils.py`
    - _Requirements: 2.1, 3.1, 3.2, 3.3, 3.4_

  - [x] 1.2 Write property test for canonical ID format validation (Property 9)
    - **Property 9: Canonical ID Format Validation**
    - **Validates: Requirements 3.2, 3.3, 3.4**
    - Generate arbitrary strings for each entity type; assert valid IDs are accepted and invalid IDs are rejected without partial records

  - [x] 1.3 Create SQLite schema initialisation module
    - Write `db_schema.py` with DDL for all three databases matching the design's SQL schema exactly
    - Write `create_all_schemas(base_path: str)` that creates the three `.db` files and all tables/indexes
    - _Requirements: 3.1, 7.4, 8.2_

- [x] 2. Implement CanonicalRegistry
  - Create `entity_resolution/canonical_registry.py`
  - Implement `CanonicalRegistry` backed by `canonical_registry.db`
  - `register()`: validate canonical_id format, persist `canonical_entities` row, update `synonyms` table and in-memory `SynonymIndex` in one SQLite transaction; return `(False, RegistrationError)` on any validation failure without creating partial records
  - `lookup_by_surface_form()`: case-insensitive, NFC-normalised lookup against `synonyms.surface_form_normalized`; return `None` on miss
  - `add_synonym()`: validate length ≤ 500, check for cross-entity duplicate, update `synonyms` + `SynonymIndex` atomically; log `SynonymConflictRecord` on duplicate
  - `get_registry_version()`: read from `registry_version` table; bump version on every write
  - _Requirements: 3.1–3.7, 5.1, 5.3, 5.4_

  - [x] 2.1 Implement CanonicalRegistry core (register, lookup, add_synonym, get_registry_version)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 2.2 Write property test for synonym conflict rejection (Property 4)
    - **Property 4: Synonym Conflict Rejection**
    - **Validates: Requirements 3.7, 5.4**
    - Generate two distinct canonical entities and a shared surface form; assert second registration is rejected and a `SynonymConflictRecord` is written

- [x] 3. Implement SynonymIndex
  - Create `entity_resolution/synonym_index.py`
  - Implement `SynonymIndex` with `_index: dict[str, str]` and `threading.RLock`
  - `lookup()`: NFC + lowercase before dict lookup; return `None` on miss; hold read lock
  - `add()`: acquire write lock (≤100ms), update `_index` and SQLite `synonyms` table atomically
  - `prefix_lookup()`: iterate `_index` under read lock, filter by normalised prefix, cap at 50, sort lexicographically
  - `rebuild_from_registry()`: clear `_index`, reload all rows from `synonyms` table
  - _Requirements: 5.1, 5.2, 5.5_

  - [x] 3.1 Implement SynonymIndex with RW lock and SQLite backing
    - _Requirements: 5.1, 5.2, 5.5_

- [x] 4. Implement ResolutionAuditStore
  - Create `entity_resolution/audit_store.py`
  - Implement `ResolutionAuditStore` backed by `resolution_audit.db`
  - `write()`: INSERT `ResolutionRecord` (serialize `conflict_set` as JSON); catch all exceptions, log to `logging.error`, return `False` on failure — never raise
  - `query()`: build parameterised SELECT with AND-chained filters from `AuditQuery`; ORDER BY `timestamp DESC`; return `[]` on no match
  - _Requirements: 7.1–7.6_

  - [x] 4.1 Implement ResolutionAuditStore (write, query)
    - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6_

  - [x] 4.2 Write property test for audit completeness (Property 5)
    - **Property 5: Audit Completeness**
    - **Validates: Requirements 7.1, 7.2, 15.4**
    - For any `resolve()` call, assert a `ResolutionRecord` exists in the audit store with non-empty `winning_strategy`, non-null `timestamp`, and matching `paper_id`

- [x] 5. Implement ResolutionCache
  - Create `entity_resolution/resolution_cache.py`
  - Implement `ResolutionCache` with in-memory `OrderedDict`-based LRU (default 10,000 entries) + SQLite `resolution_cache.db`
  - `get()`: check memory tier first (≤10ms SLA), then SQLite tier (≤100ms SLA); return `None` if `entry.registry_version != current_registry_version`
  - `put()`: write to both tiers with `registry_version`
  - `invalidate_version()`: evict all entries with `old_version` from both tiers; return count
  - _Requirements: 8.2–8.6, 2.5_

  - [x] 5.1 Implement ResolutionCache (LRU + SQLite, version-based invalidation)
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 5.2 Write property test for cache version invalidation (Property 11)
    - **Property 11: Cache Version Invalidation**
    - **Validates: Requirements 2.5, 8.5**
    - Cache a result under version V; advance registry to V+1; assert next `resolve()` re-executes full strategy sequence and returns result cached under V+1

- [x] 6. Checkpoint — core storage layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement AbbreviationExpander
  - Create `entity_resolution/abbreviation_expander.py`
  - Implement `AbbreviationExpander` backed by `abbreviation_table` in `canonical_registry.db`
  - `expand()`: check curated table first; then apply genus-initial pattern (`^[A-Z]\. \w+$`); return `[]` on no match; sort candidates lexicographically; confidence = `1.0 / N`
  - `add_mapping()`: INSERT into `abbreviation_table`, reload in-memory table immediately (hot-reload, no restart)
  - _Requirements: 11.1–11.5_

  - [x] 7.1 Implement AbbreviationExpander (curated table + genus-initial pattern, hot-reload)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 7.2 Write property test for abbreviation confidence proportionality (Property 12)
    - **Property 12: Abbreviation Confidence Proportionality**
    - **Validates: Requirements 11.2, 11.4**
    - For N genera matching a genus-initial abbreviation, assert exactly N candidates returned each with `confidence = 1.0 / N`; when N=1, confidence=1.0

- [x] 8. Implement FuzzyMatcher
  - Create `entity_resolution/fuzzy_matcher.py`
  - Implement `FuzzyMatcher`
  - `match()`: apply `normalize_surface_form()` to input; skip and return `[]` if `len(normalized) < 4` code points; compute Levenshtein distance against all `synonyms.surface_form_normalized` for the given `entity_type`; return candidates with `edit_distance ≤ 2`; sort by `edit_distance ASC`, then `canonical_id ASC`
  - `compute_confidence()`: static method implementing `1.0 - (d / max(len_s, len_c)) * 0.5`
  - _Requirements: 12.1–12.6_

  - [x] 8.1 Implement FuzzyMatcher (Levenshtein ≤ 2, short-form skip, confidence formula)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [x] 8.2 Write property test for fuzzy match confidence formula (Property 7)
    - **Property 7: Fuzzy Match Confidence Formula**
    - **Validates: Requirements 12.3**
    - For any `(edit_distance ∈ {0,1,2}, len_surface ≥ 4, len_candidate ≥ 4)`, assert `compute_confidence()` equals `1.0 - (d / max(len_s, len_c)) * 0.5` within floating-point tolerance

  - [x] 8.3 Write property test for fuzzy skip for short forms (Property 13)
    - **Property 13: Fuzzy Skip for Short Forms**
    - **Validates: Requirements 12.5**
    - For any surface form where `len(normalize(S)) < 4`, assert `FuzzyMatcher.match()` returns `[]` without performing any edit distance computation

- [x] 9. Implement OntologyTraverser
  - Create `entity_resolution/ontology_traverser.py`
  - Implement `OntologyTraverser`
  - `traverse()`: query NCBI Taxonomy (taxon) or MeSH (disease) hierarchy up to 3 levels; for each ancestor level N, check `CanonicalRegistry`; return first match as `OntologyCandidate` with `hierarchy_level=N`; if service unavailable, log warning and return `[]`
  - `compute_confidence()`: static method implementing `0.50 - (hierarchy_level - 1) * 0.10`
  - _Requirements: 13.1–13.6_

  - [x] 9.1 Implement OntologyTraverser (3-level traversal, graceful degradation, confidence formula)
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

  - [x] 9.2 Write property test for ontology traversal confidence formula (Property 8)
    - **Property 8: Ontology Traversal Confidence Formula**
    - **Validates: Requirements 13.3**
    - For each `hierarchy_level ∈ {1, 2, 3}`, assert `compute_confidence(N)` equals `0.50 - (N-1) * 0.10`

- [x] 10. Implement RankingFunction
  - Create `entity_resolution/ranking_function.py`
  - Implement `RankingFunction` with `PRIORITY_WEIGHTS` dict
  - `score_all()`: compute `composite_score = PRIORITY_WEIGHTS[strategy] × grounding_confidence` for each candidate; sort by `composite_score DESC`, then `PRIORITY_WEIGHTS[strategy] DESC`, then `canonical_id ASC`
  - `rank()`: return first element of `score_all()`; if single candidate, return directly
  - _Requirements: 4.1–4.6, 2.3_

  - [x] 10.1 Implement RankingFunction (composite scoring, deterministic tie-breaking)
    - _Requirements: 4.1, 4.2, 4.3, 4.6, 2.3_

  - [x] 10.2 Write property test for ranking composite score correctness (Property 15)
    - **Property 15: Ranking Composite Score Correctness**
    - **Validates: Requirements 4.1, 4.2, 4.3, 2.3**
    - For any non-empty conflict set, assert winner has highest composite score; assert tie-breaking by strategy priority then lexicographic `canonical_id` is deterministic

- [x] 11. Implement ManualOverrideManager
  - Create `entity_resolution/manual_override_manager.py`
  - Implement `ManualOverrideManager` backed by `manual_overrides` table in `canonical_registry.db`
  - `get_override()`: SELECT by `surface_form`; return `None` on miss
  - `set_override()`: validate `canonical_id` format, validate `justification` ≤ 500 chars, INSERT/REPLACE, invalidate `ResolutionCache` for this surface form
  - `remove_override()`: DELETE row, invalidate `ResolutionCache` for this surface form
  - `bulk_import_csv()`: read CSV with columns `surface_form, canonical_id, entity_type, curator_id, justification`; skip and log malformed rows (missing columns, invalid `canonical_id`, duplicate override for different `canonical_id`); return `BulkImportResult`
  - _Requirements: 9.1–9.8_

  - [x] 11.1 Implement ManualOverrideManager (get, set, remove, bulk CSV import)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7, 9.8_

  - [x] 11.2 Write property test for manual override priority (Property 10)
    - **Property 10: Manual Override Priority**
    - **Validates: Requirements 9.1, 9.2, 9.4**
    - For any surface form with a `ManualOverride` set, assert `resolve()` returns the override's `canonical_id` with `grounding_confidence=1.0` and `winning_strategy="manual_override"` regardless of automated strategy results

- [x] 12. Checkpoint — all strategies and support components complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement ResolutionPipeline (orchestrator)
  - Create `entity_resolution/resolution_pipeline.py`
  - Implement `ResolutionPipeline` wiring all components
  - `resolve()`: execute the seven-strategy sequence in order; apply `normalize_surface_form()` before any comparison; handle abbreviation re-entry (at most once, try each expansion in lexicographic order through steps 2–6); collect all candidates into `conflict_set`; call `RankingFunction.rank()`; set `high_conflict=True` when ≥3 strategies produced candidates; set `hierarchy_match=True` and `hierarchy_level` when `OntologyTraverser` wins; create `UnresolvedEntity` when all strategies fail; write `ResolutionRecord` to `ResolutionAuditStore` (non-blocking); update `ResolutionMetrics`; store result in `ResolutionCache`
  - `normalize()`: thin wrapper calling `resolve()`, returning `NormalizationResult(canonical_id, grounded)` — drop-in Spec 1 interface
  - `batch_resolve()`: iterate `resolve()` for each form; return results in input order
  - _Requirements: 1.1–1.5, 2.1–2.5, 14.1–14.4_

  - [x] 13.1 Implement ResolutionPipeline.resolve() with full seven-strategy sequence and abbreviation re-entry
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2_

  - [x] 13.2 Implement ResolutionPipeline.normalize() drop-in interface and batch_resolve()
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 8.1_

  - [x] 13.3 Write property test for determinism and idempotency (Property 1)
    - **Property 1: Determinism and Idempotency**
    - **Validates: Requirements 2.1, 2.4, 15.1**
    - For any surface form, assert two sequential `resolve()` calls return identical `canonical_id`, `winning_strategy`, `grounding_confidence`, and `conflict_set`; also assert `resolve(canonical_id)` returns `canonical_id` and `grounded=True`

  - [x] 13.4 Write property test for synonym completeness (Property 2)
    - **Property 2: Synonym Completeness**
    - **Validates: Requirements 5.1, 15.2**
    - For any canonical entity with registered synonyms S₁…Sₙ, assert `resolve(Sᵢ).canonical_id == entity.canonical_id` for all i, regardless of case or Unicode normalization form

  - [x] 13.5 Write property test for no-spurious-merge (Property 3)
    - **Property 3: No-Spurious-Merge**
    - **Validates: Requirements 15.3, 6.5**
    - For two distinct canonical entities with disjoint synonym sets, assert no synonym of E₁ resolves to E₂'s `canonical_id`

  - [x] 13.6 Write property test for batch consistency (Property 6)
    - **Property 6: Batch Consistency**
    - **Validates: Requirements 8.1, 15.6**
    - For any list of 1–1000 surface forms, assert `batch_resolve([F₁…Fₙ])` equals `[resolve(Fᵢ) for Fᵢ in forms]` element-wise on all fields

- [x] 14. Implement ResolutionMetrics
  - Create `entity_resolution/resolution_metrics.py`
  - Implement `ResolutionMetrics`
  - `record_resolution()`: accumulate per-entity-type counts and confidence sums in memory
  - `finalize_run()`: compute `RunMetricsSnapshot`; persist to `metrics_snapshots` table in `resolution_audit.db` (failure logged, not raised); emit `logging.warning` if any entity type `resolution_rate < 0.70`; return snapshot
  - `query_snapshots()`: SELECT snapshots in `date_from..date_to` ascending; flag entity types where most recent rate is >5 points below historical average
  - _Requirements: 10.1–10.5_

  - [x] 14.1 Implement ResolutionMetrics (record, finalize_run, query_snapshots, degradation detection)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 14.2 Write property test for resolution rate warning threshold (Property 14)
    - **Property 14: Resolution Rate Warning Threshold**
    - **Validates: Requirements 10.5**
    - For any pipeline run where at least one surface form was processed and any entity type's resolution rate < 0.70, assert a warning is emitted to the system log containing `run_id`, entity type, observed rate, and the 0.70 threshold

- [x] 15. Implement EntityMerger
  - Create `entity_resolution/entity_merger.py`
  - Implement `EntityMerger` using Neo4j Python driver
  - `ensure_canonical_node()`: MERGE on `canonical_id` property; return Neo4j node ID
  - `merge()`: open Neo4j transaction; (1) verify both nodes have same `entity_type` — reject with type-conflict log if different; (2) transfer all inbound/outbound relationships; (3) deduplicate relationships (same type + counterpart + direction → keep higher confidence); (4) delete source node; (5) write `MergeLogEntry`; commit; on any exception, rollback and write `MergeRollbackEntry`
  - _Requirements: 6.1–6.7_

  - [x] 15.1 Implement EntityMerger (atomic Neo4j merge, relationship deduplication, rollback)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 16. Implement shadow mode and Spec 1 integration
  - Add `shadow_mode: bool` flag to `ResolutionPipeline.__init__()`
  - Implement `normalize_shadow_mode()`: run both Spec 1 and Spec 2 normalizers; log `ShadowModeDiscrepancy` whenever `canonical_id` or `grounded` differ; return Spec 1 result
  - Add `enable_shadow_mode()` / `disable_shadow_mode()` methods
  - Wire `normalize()` to call `normalize_shadow_mode()` when `shadow_mode=True`
  - _Requirements: 14.5, 14.6_

  - [x] 16.1 Implement shadow mode discrepancy logging and Spec 1 compatibility wiring
    - _Requirements: 14.5, 14.6_

- [x] 17. Final checkpoint — full pipeline integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The three SQLite databases (`canonical_registry.db`, `resolution_cache.db`, `resolution_audit.db`) must remain physically separate at all times
- Neo4j integration (Task 15) requires a running Neo4j instance; use the `neo4j` Python driver with explicit transaction management
- All Hypothesis property tests use `@settings(max_examples=100)` and the `ci` profile from `conftest.py`
- `normalize_surface_form()` (NFC + lowercase + strip punctuation + collapse whitespace) must be called consistently across all components before any comparison
- Abbreviation re-entry in the pipeline occurs at most once per `resolve()` call to prevent infinite cycles
- Write failures to `ResolutionAuditStore` and `ResolutionMetrics` snapshot persistence are non-blocking — log and continue
- `SynonymIndex` write lock must not block concurrent lookups for more than 100ms

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2", "2.1", "3.1", "4.1", "5.1"] },
    { "id": 2, "tasks": ["2.2", "4.2", "5.2", "7.1", "8.1", "9.1", "10.1"] },
    { "id": 3, "tasks": ["7.2", "8.2", "8.3", "9.2", "10.2", "11.1"] },
    { "id": 4, "tasks": ["11.2", "13.1"] },
    { "id": 5, "tasks": ["13.2", "14.1", "15.1"] },
    { "id": 6, "tasks": ["13.3", "13.4", "13.5", "13.6", "14.2"] },
    { "id": 7, "tasks": ["16.1"] }
  ]
}
```
