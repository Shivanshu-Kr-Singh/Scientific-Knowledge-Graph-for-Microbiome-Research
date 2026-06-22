# Layer 1 Scale Roadmap — Implementation Summary

## Overview

The Layer 1 relevance-filtering pipeline has been scaled from ~5,000 to ~100,000 papers by progressively reducing LLM verification calls. The strategy introduces **embedding-based similarity classification**, **intelligent LLM routing**, **batched verification**, **semantic caching**, and **learning-loop feedback** into the existing 4-stage pipeline.

**Target:** Reduce LLM calls by 60–80% as the Embedding Store accumulates knowledge while maintaining ≥0.90 F1 precision.

## Pipeline Flow (After Integration)

```
Stage 1 (MeSH Metadata)
  → Stage 2 (Weighted Rules)
    → Metagenomics Gate
      → Stage 3 (ML Classifier)
        → Stage 3.5 (Embedding Classifier) ← NEW
          → Disagreement Router ← NEW
            → Semantic Cache ← NEW
              → Stage 4 (Batched LLM Verifier) ← ENHANCED
                → Embedding Store Growth ← NEW
                  → Pipeline Metrics ← NEW
```

---

## Phase 1: Foundation (No Data-Volume Dependency)

### New Modules

| Module | Purpose |
|--------|---------|
| `collectors/embedding_model.py` | SPECTER2 sentence-transformer wrapper with all-MiniLM-L6-v2 fallback, batch encoding, OOM retry |
| `collectors/embedding_store.py` | NumPy-backed persistent store with positive/negative partitions, filelock for concurrency, latency monitoring |
| `collectors/embedding_filter.py` | Stage 3.5 classifier — KEEP if pos_sim ≥ 0.85 & neg_sim < 0.60, REJECT if neg_sim ≥ 0.85 & pos_sim < 0.60, BORDERLINE otherwise |
| `collectors/metrics_logger.py` | Appends per-run JSONL metrics (stage resolution counts, LLM calls, cache hits, store sizes, latency stats) |
| `scripts/seed_embedding_store.py` | One-time backfill from historical `data/audit/{kept,rejected,llm_verified}.json` — idempotent |

### Modified Modules

| Module | Changes |
|--------|---------|
| `collectors/audit_logger.py` | Added DOI, PMID, abstract (truncated 2000 chars) to audit records — backward compatible |
| `config.py` | Added 20+ environment variables for embedding model, thresholds, caching, growth, latency, drift |
| `requirements.txt` | Added `sentence-transformers>=2.2.0`, `filelock>=3.12.0` |

---

## Phase 2: Learning Loop (Activates at ~2,000 Papers)

### New Modules

| Module | Purpose |
|--------|---------|
| `collectors/calibration.py` | Platt scaling — fits sigmoid on held-out logits for calibrated probabilities. Enforces monotonicity, [0,1] range. Min 200 samples. |
| `collectors/hybrid_classifier.py` | Stacked meta-classifier combining [rule_score, pos_sim, neg_sim, ml_prob]. LogisticRegression with Platt calibration. Activates at ≥2000 papers. Discards model if F1 < 0.80. |

---

## Phase 3: Scale Hardening (50K–100K Papers)

### New Modules

| Module | Purpose |
|--------|---------|
| `scripts/drift_monitor.py` | Monthly sampling of 1% automated decisions (min 10) for manual review. Writes `data/audit/drift_review_YYYYMM.json`. |

### Modified Modules

| Module | Changes |
|--------|---------|
| `scheduler/jobs.py` | `daily_update()` runs enhanced pipeline; `weekly_refresh()` retrains hybrid classifier; `monthly_rescan()` runs drift monitor |
| `main.py` | Added `RUN_LAYER=seed` and `RUN_LAYER=drift` entry points; startup check for sentence-transformers |

---

## Pipeline Integration (`collectors/relevance_filter.py`)

### New Components Wired in `__init__()`
- `EmbeddingModel` (lazy, fail-graceful)
- `EmbeddingStore` (lazy, fail-graceful)
- `EmbeddingFilter` (requires model + store)
- `SemanticCache` (lazy, fail-graceful)
- `BatchedVerifier` (lazy, fail-graceful)
- `MetricsLogger`

### Modified `_evaluate()` Flow
1. After Stage 3 returns BORDERLINE → calls `_stage3_5_embedding()`
2. If Stage 3.5 returns KEEP/REJECT → `_disagreement_router()` decides if LLM needed
3. If no LLM needed → accepts Stage 3.5 verdict as final
4. If LLM needed → checks `SemanticCache` first (cosine > 0.97 = cache hit)
5. Falls through to `_stage4_llm()` only when truly uncertain

### Modified `filter()` Flow
1. Tracks per-stage counts including `stage3_5` and `stage4`
2. Counts LLM calls vs semantic cache hits
3. **Post-loop**: feeds confident verdicts into embedding store growth (score ≥ 0.80 → positive, ≤ 0.20 → negative)
4. **Post-loop**: records `PipelineMetrics` to JSONL

### New Methods
- `_stage3_5_embedding(paper)` — delegates to EmbeddingFilter
- `_disagreement_router(paper, stage2_verdict, stage3_5_verdict, blended_confidence)` — routes to LLM on disagreement or borderline confidence [0.40, 0.70]
- `_embedding_store_growth(paper, score)` — appends confident decisions back to store

---

## Batched LLM Verifier (`collectors/llm_verifier.py`)

| Class | Purpose |
|-------|---------|
| `BatchedVerifier` | Groups up to 16 papers per Ollama call. Structured JSON prompt/response. Split-and-retry on parse failure. Single-paper fallback → human review. |
| `SemanticCache` | Cosine similarity > 0.97 reuses cached LLM verdicts. Independent from content-hash cache. Stored in `data/embeddings/llm_verdict_cache.npy`. |

---

## Property Tests (19 Properties)

| # | Property | File | Validates |
|---|----------|------|-----------|
| 1 | Embedding Model Output Consistency | `test_property_embedding_model.py` | Req 1.3 |
| 2 | Batch Encoding Equivalence | `test_property_embedding_model.py` | Req 1.5 |
| 3 | Embedding Store Round-Trip | `test_property_embedding_store.py` | Req 2.1, 2.6 |
| 4 | Partition Isolation | `test_property_embedding_store.py` | Req 2.2 |
| 5 | Cosine Similarity Correctness | `test_property_embedding_store.py` | Req 2.3 |
| 6 | Audit Logger Field Completeness | `test_property_audit_logger.py` | Req 3.1–3.3 |
| 7 | Backfill Partition Correctness | `test_property_backfill.py` | Req 4.2–4.4 |
| 8 | Backfill Idempotence | `test_property_backfill.py` | Req 4.5 |
| 9 | Stage 3.5 Threshold Invariant | `test_property_embedding_filter.py` | Req 5.3–5.5 |
| 10 | Disagreement Router Logic | `test_property_disagreement_router.py` | Req 6.1–6.3 |
| 11 | Batch Size Invariant | `test_property_batch_size.py` | Req 7.1 |
| 12 | Semantic Cache Threshold | `test_property_semantic_cache.py` | Req 8.1–8.2 |
| 13 | Semantic Cache Growth | `test_property_semantic_cache.py` | Req 8.4 |
| 14 | Store Growth Threshold Logic | `test_property_store_growth.py` | Req 9.1–9.4 |
| 15 | Platt Scaling Monotonicity & Range | `test_property_platt_scaling.py` | Req 10.3, 11.2 |
| 16 | Active Learning Retrain Guard | `test_property_retrain_guard.py` | Req 12.2, 12.5 |
| 17 | Latency Recording & Warning | `test_property_latency_monitoring.py` | Req 13.1–13.2 |
| 18 | Drift Monitor Sampling Guarantee | `test_property_drift_monitor.py` | Req 14.1, 14.4 |
| 19 | Pipeline Metrics Completeness | `test_property_metrics_logger.py` | Req 15.1–15.4 |

---

## Configuration (Environment Variables)

```bash
# Embedding Model
EMBEDDING_MODEL_NAME=allenai/specter2
EMBEDDING_FALLBACK_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=64

# Embedding Store
EMBEDDING_STORE_DIR=data/embeddings
EMBEDDING_STORE_BACKEND=numpy  # "numpy" | "faiss" (future)

# Stage 3.5 Thresholds
EMBEDDING_POS_KEEP_THRESHOLD=0.85
EMBEDDING_NEG_REJECT_THRESHOLD=0.85
EMBEDDING_CROSS_CEILING=0.60
EMBEDDING_MIN_PARTITION_SIZE=50

# Semantic Cache
SEMANTIC_CACHE_THRESHOLD=0.97

# Batched Verifier
BATCH_LLM_SIZE=16

# Hybrid Classifier
HYBRID_MIN_STORE_SIZE=2000
HYBRID_MIN_TRAIN_SAMPLES=200
HYBRID_MIN_RETRAIN_NEW=100

# Disagreement Router
BLENDED_CONFIDENCE_LOW=0.40
BLENDED_CONFIDENCE_HIGH=0.70

# Embedding Store Growth
GROWTH_KEEP_THRESHOLD=0.80
GROWTH_REJECT_THRESHOLD=0.20

# Latency Monitoring
EMBEDDING_LATENCY_WARN_MS=200.0

# Drift Monitor
DRIFT_SAMPLE_RATE=0.01
DRIFT_MIN_SAMPLE=10
```

---

## Data Layout

```
data/
├── embeddings/
│   ├── positive.npy              # (N, dim) float32 — relevant papers
│   ├── positive_meta.json        # metadata for each positive embedding
│   ├── negative.npy              # (M, dim) float32 — irrelevant papers
│   ├── negative_meta.json        # metadata for each negative embedding
│   ├── llm_verdict_cache.npy     # semantic cache vectors
│   └── llm_verdict_cache_meta.json
├── models/
│   ├── hybrid_classifier.pkl     # stacked classifier + metadata
│   └── calibration_params.json   # Platt scaling slope/intercept
├── metrics/
│   └── pipeline_runs.jsonl       # one JSON record per pipeline run
└── audit/
    ├── kept.json
    ├── rejected.json
    ├── llm_verified.json
    └── drift_review_YYYYMM.json  # monthly drift samples
```

---

## How to Run

```bash
# 1. Install dependencies
./venv/bin/pip install sentence-transformers>=2.2.0 filelock>=3.12.0

# 2. Seed embedding store from historical audit data (one-time)
RUN_LAYER=seed ./venv/bin/python3 main.py

# 3. Run the full pipeline (now with Stage 3.5)
RUN_LAYER=1 MAX_PER_SOURCE=100 ./venv/bin/python3 main.py

# 4. Run drift monitoring (monthly)
RUN_LAYER=drift ./venv/bin/python3 main.py

# 5. Run all property tests
./venv/bin/python3 -m pytest tests/ -v
```

---

## Progressive Behavior

| Store Size | Pipeline Behavior |
|-----------|-------------------|
| 0–49 per partition | Stage 3.5 returns INSUFFICIENT_DATA → all papers go to LLM (same as before) |
| 50–1999 | Stage 3.5 makes KEEP/REJECT decisions. Disagreement router sends conflicts to LLM. LLM calls reduce ~40%. |
| 2000+ | Hybrid meta-classifier activates. Calibrated confidence drives routing. LLM calls reduce ~60–80%. |
| 50,000+ | Drift monitor ensures no silent degradation. Weekly retrain keeps model fresh. |

---

## Architecture Decisions

1. **NumPy brute-force over FAISS**: At <100k embeddings (384-dim), brute-force cosine completes in <50ms. Interface abstracts storage for future FAISS swap.
2. **Batch size 16 for LLM**: Ollama context window supports ~65k tokens (16 × 4k per paper). Larger batches risk JSON parse failures.
3. **0.97 semantic cache threshold**: Papers above this are near-duplicates (preprint vs. published). Lower thresholds risk verdict reuse across different papers.
4. **Separate positive/negative partitions**: Enables two distinct similarity signals that the meta-classifier combines for higher accuracy.
5. **All fail-graceful**: Every new component initializes in try/except. If sentence-transformers is missing, the pipeline falls back to the original Stage 4 path.
