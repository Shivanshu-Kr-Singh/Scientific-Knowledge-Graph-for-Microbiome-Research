# Implementation Plan: Layer 1 Scale Roadmap

## Overview

This plan implements the three-phase Layer 1 relevance-filtering scale roadmap. Phase 1 (Foundation) tasks come first with no data-volume dependency. Phase 2 (Learning loop) and Phase 3 (Scale hardening) tasks follow, gated by store size thresholds. All new modules use Python 3.12, Pydantic for models, pytest +  for testing, and sentence-transformers for embeddings.

## Tasks

- [x] 1. Add dependencies and configuration
  - [x] 1.1 Add new dependencies to requirements.txt
    - Add `sentence-transformers>=2.2.0` and `filelock>=3.12.0`
    - Verify existing `numpy`, `scikit-learn`, `hypothesis` entries remain
    - _Requirements: 1.1, 1.5, 2.5_

  - [x] 1.2 Add embedding and pipeline configuration to `config.py`
    - Add all new environment variables: EMBEDDING_MODEL_NAME, EMBEDDING_FALLBACK_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_STORE_DIR, Stage 3.5 thresholds, SEMANTIC_CACHE_THRESHOLD, BATCH_LLM_SIZE, HYBRID_MIN_STORE_SIZE, HYBRID_MIN_TRAIN_SAMPLES, HYBRID_MIN_RETRAIN_NEW, BLENDED_CONFIDENCE_LOW/HIGH, GROWTH_KEEP_THRESHOLD, GROWTH_REJECT_THRESHOLD, EMBEDDING_LATENCY_WARN_MS, DRIFT_SAMPLE_RATE, DRIFT_MIN_SAMPLE, EMBEDDING_STORE_BACKEND
    - All values read from environment with sensible defaults per design
    - _Requirements: 5.3, 5.4, 5.7, 6.2, 7.1, 8.1, 9.1, 9.2, 10.1, 10.5, 13.2, 14.1, 14.4_

- [x] 2. Implement Embedding Model wrapper
  - [x] 2.1 Create `collectors/embedding_model.py`
    - Implement `EmbeddingModelInterface` Protocol
    - Implement `EmbeddingModel` class with SPECTER2 primary, all-MiniLM-L6-v2 fallback
    - Implement `encode(texts, batch_size)` returning `(n, dimension)` ndarray
    - Implement `encode_paper(title, abstract)` convenience method
    - Expose `dimension` property
    - Handle OOM by halving batch size and retrying
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 2.2 Write property tests for Embedding Model
    - **Property 1: Embedding Model Output Consistency** — for any non-empty (title, abstract), output shape is `(dimension,)`, all values finite, non-zero norm
    - **Property 2: Batch Encoding Equivalence** — batch encode equals individual encode + stack within float32 tolerance
    - **Validates: Requirements 1.3, 1.5**

- [x] 3. Implement Embedding Store
  - [x] 3.1 Create `collectors/embedding_store.py`
    - Implement `EmbeddingMetadata` dataclass and `SimilarityResult` class
    - Implement `EmbeddingStoreInterface` Protocol
    - Implement `EmbeddingStore` class with NumPy brute-force cosine similarity
    - Implement separate positive/negative partitions stored as `.npy` + `_meta.json`
    - Implement `query_similar(vector, partition, top_k)` with cosine similarity ranking
    - Implement `append(vector, metadata)` with filelock for atomic writes
    - Implement `contains(doi, pmid)` deduplication check
    - Implement `positive_count` and `negative_count` properties
    - Implement `query_latency_stats()` for latency monitoring
    - Handle corrupted `.npy` by reinitializing empty partition
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 3.2 Write property tests for Embedding Store
    - **Property 3: Embedding Store Round-Trip** — append + reload from disk yields identical vector and metadata
    - **Property 4: Partition Isolation** — positive append never appears in negative query results, and vice versa
    - **Property 5: Cosine Similarity Correctness** — top-k results match manually computed cosine, ordered descending
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.6**

- [x] 4. Enhance Audit Logger
  - [x] 4.1 Modify `collectors/audit_logger.py` to include DOI, PMID, and truncated abstract
    - Add `doi`, `pmid`, `abstract` (truncated to 2000 chars) fields to the audit record
    - Maintain backward compatibility — no existing fields removed or renamed
    - Ensure JSON schema remains a superset of old schema
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 4.2 Write property test for Audit Logger
    - **Property 6: Audit Logger Field Completeness** — for any PaperRecord with doi/pmid/abstract, the audit record contains all new fields plus all existing fields preserved
    - **Validates: Requirements 3.1, 3.2, 3.3**

- [x] 5. Checkpoint - Foundation modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Backfill Seeding Script
  - [x] 6.1 Create `scripts/seed_embedding_store.py`
    - Implement `BackfillSeeder` class that reads `data/audit/kept.json`, `rejected.json`, `llm_verified.json`
    - Encode records with non-empty abstracts using EmbeddingModel
    - Place kept papers in positive partition, rejected in negative
    - Skip records lacking abstract, log skip count
    - Deduplicate by DOI/PMID before inserting (idempotent)
    - Exit with clear message if no records have abstract fields
    - Return stats dict: positive_added, negative_added, skipped_no_abstract, skipped_duplicate
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 6.2 Write property tests for Backfill Script
    - **Property 7: Backfill Partition Correctness** — kept papers go positive, rejected go negative, skip count matches empty-abstract count
    - **Property 8: Backfill Idempotence** — running twice on same data produces same store size
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5**

- [x] 7. Implement Stage 3.5 Embedding Filter
  - [x] 7.1 Create `collectors/embedding_filter.py`
    - Implement `EmbeddingVerdict` dataclass
    - Implement `EmbeddingFilter` class with configurable thresholds from config.py
    - Implement `evaluate(paper)` that queries both partitions and applies threshold logic
    - Return KEEP if pos_sim ≥ 0.85 and neg_sim < 0.60
    - Return REJECT if neg_sim ≥ 0.85 and pos_sim < 0.60
    - Return BORDERLINE otherwise
    - Return INSUFFICIENT_DATA if either partition has < 50 embeddings
    - Log similarity scores and decision
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 7.2 Write property test for Stage 3.5 Classification
    - **Property 9: Stage 3.5 Classification Threshold Invariant** — for any (pos_sim, neg_sim) pair with ≥50 embeddings per partition, the decision matches the threshold rules exactly
    - **Validates: Requirements 5.3, 5.4, 5.5**

- [x] 8. Implement Disagreement Router
  - [x] 8.1 Implement disagreement routing logic in `collectors/relevance_filter.py`
    - Add `_disagreement_router()` method comparing Stage 2 verdict with Stage 3.5 verdict
    - Route to LLM if verdicts disagree (one keeps, other rejects)
    - Route to LLM if Blended Confidence in [0.40, 0.70]
    - Accept Stage 3.5 verdict as final if neither condition met
    - Log routing reason for each LLM invocation
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 8.2 Write property test for Disagreement Router
    - **Property 10: Disagreement Router Decision Logic** — for any (stage2_keep, stage3_5_keep, blended_confidence), the routing decision matches the logic rules exactly
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 9. Implement Batched LLM Verifier
  - [x] 9.1 Implement `BatchedVerifier` class in `collectors/llm_verifier.py`
    - Implement `BatchVerdict` dataclass
    - Implement `verify_batch(papers)` that groups papers into batches of ≤16
    - Format each batch as structured JSON array prompt for Ollama
    - Implement split-and-retry on unparseable response (split in half, retry sub-batches)
    - Fall back to single-paper for persistent failures → mark for human review
    - Respect existing OLLAMA_TIMEOUT_SECONDS and OLLAMA_MAX_RETRIES config
    - Log batch size, success count, retry count, split count
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 9.2 Write property test for Batch Size Invariant
    - **Property 11: Batch Size Invariant** — for any N papers, partitioned into ceil(N/16) batches of ≤16, union of all batches equals original set
    - **Validates: Requirements 7.1**

- [x] 10. Implement Semantic Cache
  - [x] 10.1 Implement `SemanticCache` class in `collectors/llm_verifier.py`
    - Implement `lookup(paper_embedding)` that checks cosine similarity > 0.97 against cached vectors
    - Implement `store_verdict(paper_embedding, verdict, paper)` that appends to cache store
    - Storage: `data/embeddings/llm_verdict_cache.npy` + `llm_verdict_cache_meta.json`
    - Operate independently from existing content-hash cache in `data/processed/llm_cache.json`
    - Handle corrupted cache by reinitializing empty
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 10.2 Write property tests for Semantic Cache
    - **Property 12: Semantic Cache Threshold Correctness** — similarity > 0.97 returns hit, ≤ 0.97 returns miss
    - **Property 13: Semantic Cache Growth** — after storing a verdict, cache size increases by exactly one
    - **Validates: Requirements 8.1, 8.2, 8.4**

- [x] 11. Checkpoint - Core pipeline stages complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement Embedding Store Growth and Metrics
  - [x] 12.1 Implement active store growth logic in `collectors/relevance_filter.py`
    - After final verdict, append embedding to positive partition if score ≥ 0.80 and paper not already in store
    - Append to negative partition if score ≤ 0.20 and paper not already in store
    - Skip if 0.20 < score < 0.80 or paper already present (by DOI/PMID)
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 12.2 Write property test for Store Growth
    - **Property 14: Embedding Store Growth Threshold Logic** — score ≥ 0.80 → positive, score ≤ 0.20 → negative, between → skip, duplicate → skip
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

  - [x] 12.3 Create `collectors/metrics_logger.py`
    - Implement `PipelineMetrics` dataclass with all fields per design
    - Implement `MetricsLogger.record(metrics)` appending JSON to `data/metrics/pipeline_runs.jsonl`
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [x] 12.4 Write property test for Pipeline Metrics
    - **Property 19: Pipeline Metrics Record Completeness** — for any pipeline run of N papers, the record contains all required fields and per-stage counts sum to N
    - **Validates: Requirements 15.1, 15.2, 15.3, 15.4**

- [x] 13. Implement Latency Monitoring
  - [x] 13.1 Add latency recording to `collectors/embedding_store.py`
    - Record wall-clock duration of each `query_similar` call
    - Compute rolling average and p95 latency
    - Emit warning log when rolling average exceeds 200ms
    - _Requirements: 13.1, 13.2, 13.3_

  - [x] 13.2 Write property test for Latency Monitoring
    - **Property 17: Embedding Query Latency Recording** — each query records a positive float duration (ms); rolling average > 200ms triggers warning
    - **Validates: Requirements 13.1, 13.2**

- [x] 14. Wire pipeline integration in `collectors/relevance_filter.py`
  - [x] 14.1 Add new instance variables and Stage 3.5 call into `_evaluate()` flow
    - Initialize EmbeddingModel, EmbeddingStore, EmbeddingFilter, HybridClassifier, SemanticCache, BatchedVerifier, MetricsLogger in `__init__()`
    - Insert `_stage3_5_embedding()` call after Stage 3 ML classifier
    - Apply `_disagreement_router()` before Stage 4
    - Replace direct `_stage4_llm()` with batch queue + semantic cache lookup
    - _Requirements: 5.1, 6.1, 6.3, 8.1_

  - [x] 14.2 Modify `filter()` method for batch processing
    - Collect papers needing LLM into a batch queue
    - After iterating all papers, process batch through SemanticCache → BatchedVerifier
    - Invoke store growth after final verdicts
    - Append pipeline run metrics at end
    - _Requirements: 7.1, 8.1, 9.1, 15.1_

- [x] 15. Checkpoint - Full pipeline integration complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement Phase 2 components (activate at ~2000 papers)
  - [x] 16.1 Create `collectors/calibration.py`
    - Implement `CalibrationResult` dataclass
    - Implement `PlattCalibrator` class with `fit(logits, labels)` and `calibrate(raw_logit)`
    - Enforce monotonicity and [0, 1] output range
    - Minimum 200 samples for calibration; skip with warning if insufficient
    - Persist calibration parameters alongside hybrid_classifier.pkl
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 16.2 Write property test for Platt Scaling
    - **Property 15: Platt Scaling Monotonicity and Range** — for logit_a < logit_b, calibrated(a) ≤ calibrated(b), and all outputs in [0, 1]
    - **Validates: Requirements 10.3, 11.2**

  - [x] 16.3 Create `collectors/hybrid_classifier.py`
    - Implement `HybridVerdict` dataclass
    - Implement `HybridClassifier` class with `predict()`, `train()`, `is_active` property
    - Combine rule_score, pos_sim, neg_sim, ml_prob features
    - Activate only when store has ≥ 2000 papers
    - Train on LLM-verified papers; discard model if F1 < 0.80
    - Persist model to `data/models/hybrid_classifier.pkl`
    - Integrate PlattCalibrator for calibrated probabilities
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 16.4 Write property test for Active Learning Retrain Guard
    - **Property 16: Active Learning Retrain Guard** — retrain only when new_count ≥ 100; discard if F1 < 0.80
    - **Validates: Requirements 12.2, 12.5**

- [x] 17. Implement Phase 3 components (scale hardening)
  - [x] 17.1 Create `scripts/drift_monitor.py`
    - Implement `DriftMonitor` class
    - Sample 1% of automated (non-LLM-verified) decisions from past month
    - Ensure minimum 10 papers selected (increase rate if needed)
    - Write sampled papers to `data/audit/drift_review_YYYYMM.json`
    - Log sample size and keep/reject distribution
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 17.2 Write property test for Drift Monitor
    - **Property 18: Drift Monitor Sampling Guarantee** — sample size is max(ceil(population * 0.01), 10); at least 10 papers when population ≥ 10
    - **Validates: Requirements 14.1, 14.4**

  - [x] 17.3 Modify `scheduler/jobs.py` for weekly retrain and monthly drift
    - Implement `weekly_refresh()` calling `HybridClassifier.retrain_if_needed(store)` (only when ≥ 100 new papers)
    - Implement `monthly_rescan()` calling `DriftMonitor.run()`
    - Ensure `daily_update()` triggers the enhanced pipeline (includes Stage 3.5)
    - _Requirements: 12.1, 12.2, 14.1_

- [x] 18. Wire entry points in `main.py`
  - [x] 18.1 Add `RUN_LAYER=seed` and `RUN_LAYER=drift` entry points
    - `RUN_LAYER=seed` → execute `BackfillSeeder.run()`
    - `RUN_LAYER=drift` → execute `DriftMonitor.run()`
    - Add startup check verifying sentence-transformers is importable for Layer 1
    - _Requirements: 4.1, 14.1_

- [x] 19. Final checkpoint - All phases implemented
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based test sub-tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical boundaries
- Phase 1 tasks (1–15) have no data-volume dependency and can run immediately
- Phase 2 tasks (16) activate only when Embedding Store reaches ~2000 papers
- Phase 3 tasks (17–18) provide scale hardening for 50K–100K paper volumes
- The existing ML classifier (Stage 3) and metagenomics gate remain unchanged
- All LLM operations are Ollama-only — no cloud API fallbacks
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.1", "4.2"] },
    { "id": 3, "tasks": ["3.2", "6.1"] },
    { "id": 4, "tasks": ["6.2", "7.1", "12.3"] },
    { "id": 5, "tasks": ["7.2", "8.1", "9.1", "10.1", "12.4"] },
    { "id": 6, "tasks": ["8.2", "9.2", "10.2", "12.1", "13.1"] },
    { "id": 7, "tasks": ["12.2", "13.2", "14.1"] },
    { "id": 8, "tasks": ["14.2"] },
    { "id": 9, "tasks": ["16.1", "17.1"] },
    { "id": 10, "tasks": ["16.2", "16.3", "17.2", "17.3"] },
    { "id": 11, "tasks": ["16.4", "18.1"] }
  ]
}
```
