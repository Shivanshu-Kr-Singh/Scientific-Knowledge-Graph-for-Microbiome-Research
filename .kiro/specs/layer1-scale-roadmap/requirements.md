# Requirements Document

## Introduction

This document specifies requirements for a three-phase Layer 1 relevance-filtering scale roadmap targeting ~100,000 papers. The core objective is to progressively reduce LLM verification calls without sacrificing precision by introducing embedding-based similarity classification, intelligent LLM routing, batched verification, semantic caching, and learning-loop feedback mechanisms into the existing 4-stage pipeline.

The existing pipeline flow is preserved: Stage 1 (MeSH metadata) → Stage 2 (weighted rules) → Metagenomics Gate → Stage 3 (ML classifier) → Stage 3.5 (new embedding classifier) → Stage 4 (LLM verifier). The LLM backend is Ollama only — no cloud API fallbacks exist.

## Glossary

- **Pipeline**: The multi-stage relevance filtering system in `collectors/relevance_filter.py` that evaluates papers for inclusion in the knowledge graph
- **Embedding_Store**: A persistent storage system holding paper embeddings partitioned into positive (relevant) and negative (irrelevant) sets, backed by NumPy `.npy` files and JSON metadata
- **Embedding_Model**: A domain-tuned sentence-transformer model (SPECTER2 or PubMedBERT) used to encode paper titles and abstracts into dense vectors, with fallback to all-MiniLM-L6-v2
- **Stage_3_5_Classifier**: The new embedding-based classification stage inserted after the existing ML classifier (Stage 3) and before the LLM verifier (Stage 4)
- **Semantic_Cache**: A cosine-similarity-based deduplication layer that reuses cached LLM verdicts for near-duplicate papers (cosine similarity > 0.97)
- **Batched_Verifier**: A grouped LLM verification system that sends up to 16 papers per Ollama call using structured JSON array responses
- **Meta_Classifier**: A stacked classifier combining rule_score, embedding_positive_similarity, embedding_negative_similarity, and ml_probability into a single calibrated confidence score
- **Audit_Logger**: The existing `collectors/audit_logger.py` component that records paper decisions to JSON files in `data/audit/`
- **Backfill_Script**: A one-time seeding utility that populates the Embedding_Store from historical audit data
- **Active_Learning_Job**: A scheduled weekly job that retrains the Meta_Classifier using accumulated feedback data
- **Drift_Monitor**: A monthly sampling process that selects 1% of automated decisions for manual review to detect classification drift
- **Ollama**: The local LLM inference server used exclusively for Stage 4 verification
- **Blended_Confidence**: A weighted combination of rule score and embedding similarity scores used to determine routing decisions
- **Disagreement_Router**: Logic that triggers LLM verification when the rule stage and embedding stage produce conflicting verdicts

## Requirements

### Requirement 1: Domain-Tuned Embedding Model Wrapper

**User Story:** As a pipeline developer, I want a domain-tuned embedding model wrapper, so that paper embeddings capture biomedical semantic similarity more accurately than general-purpose models.

#### Acceptance Criteria

1. THE Embedding_Model SHALL load SPECTER2 or PubMedBERT sentence-transformer as the primary encoding model
2. IF the primary model fails to load, THEN THE Embedding_Model SHALL fall back to all-MiniLM-L6-v2 and log a warning
3. WHEN a paper title and abstract are provided, THE Embedding_Model SHALL return a dense vector of fixed dimensionality
4. THE Embedding_Model SHALL expose a swappable interface allowing future model replacement without modifying downstream consumers
5. THE Embedding_Model SHALL encode papers in batches of configurable size to support throughput scaling

### Requirement 2: Persistent Embedding Store

**User Story:** As a pipeline developer, I want a persistent embedding store with positive and negative partitions, so that the system accumulates relevance knowledge across runs and supports similarity-based classification.

#### Acceptance Criteria

1. THE Embedding_Store SHALL persist embeddings as NumPy `.npy` files with a companion JSON metadata index
2. THE Embedding_Store SHALL maintain separate partitions for positive (relevant) and negative (irrelevant) paper embeddings
3. WHEN a similarity query is issued, THE Embedding_Store SHALL compute brute-force cosine similarity against the requested partition and return the top-k results with scores
4. THE Embedding_Store SHALL expose a storage interface that permits future replacement of brute-force search with FAISS without modifying calling code
5. THE Embedding_Store SHALL support atomic append operations so that concurrent pipeline runs do not corrupt stored data
6. THE Embedding_Store SHALL store DOI, PMID, and title as metadata alongside each embedding for traceability

### Requirement 3: Audit Logger Enhancement

**User Story:** As a pipeline developer, I want the Audit_Logger to save DOI, PMID, and abstract alongside existing fields, so that historical audit data can seed the Embedding_Store.

#### Acceptance Criteria

1. WHEN logging a paper decision, THE Audit_Logger SHALL include the paper DOI in the audit record
2. WHEN logging a paper decision, THE Audit_Logger SHALL include the paper PMID in the audit record
3. WHEN logging a paper decision, THE Audit_Logger SHALL include the paper abstract (truncated to 2000 characters) in the audit record
4. THE Audit_Logger SHALL maintain backward compatibility with existing audit file schema by adding new fields without removing or renaming existing ones

### Requirement 4: Backfill Seeding Script

**User Story:** As a pipeline developer, I want a one-time backfill script that seeds the Embedding_Store from historical audit data, so that the embedding classifier has initial training signal from day one.

#### Acceptance Criteria

1. WHEN executed, THE Backfill_Script SHALL read all records from `data/audit/kept.json`, `data/audit/rejected.json`, and `data/audit/llm_verified.json`
2. THE Backfill_Script SHALL encode each record that has a non-empty abstract using the Embedding_Model
3. THE Backfill_Script SHALL place embeddings into the positive partition for kept papers and the negative partition for rejected papers
4. THE Backfill_Script SHALL skip records that lack an abstract field and log the count of skipped records
5. THE Backfill_Script SHALL be idempotent — re-running the script on the same data SHALL NOT create duplicate embeddings
6. IF no audit records contain abstract fields, THEN THE Backfill_Script SHALL exit with a clear message instructing the user to run the enhanced Audit_Logger first

### Requirement 5: Embedding-Based Classification Stage (Stage 3.5)

**User Story:** As a pipeline developer, I want an embedding-based classification stage inserted after the ML classifier, so that papers benefit from accumulated similarity knowledge before reaching the expensive LLM verifier.

#### Acceptance Criteria

1. THE Stage_3_5_Classifier SHALL execute after Stage 3 (ML classifier) and before Stage 4 (LLM verifier) in the pipeline evaluation flow
2. WHEN evaluating a paper, THE Stage_3_5_Classifier SHALL compute cosine similarity against both positive and negative partitions of the Embedding_Store
3. WHEN the positive similarity exceeds 0.85 and the negative similarity is below 0.60, THE Stage_3_5_Classifier SHALL classify the paper as relevant without invoking the LLM
4. WHEN the negative similarity exceeds 0.85 and the positive similarity is below 0.60, THE Stage_3_5_Classifier SHALL classify the paper as irrelevant without invoking the LLM
5. WHEN neither confident-relevant nor confident-irrelevant thresholds are met, THE Stage_3_5_Classifier SHALL pass the paper to Stage 4 (LLM verifier) via the Disagreement_Router
6. THE Stage_3_5_Classifier SHALL log its similarity scores and decision for auditability
7. IF the Embedding_Store contains fewer than 50 embeddings in either partition, THEN THE Stage_3_5_Classifier SHALL pass all papers through to Stage 4 without making autonomous decisions

### Requirement 6: Disagreement-Triggered LLM Routing

**User Story:** As a pipeline developer, I want the LLM to be invoked only when stages disagree or confidence is borderline, so that LLM calls decrease as the pipeline accumulates knowledge.

#### Acceptance Criteria

1. WHEN the Stage 2 rule verdict and the Stage_3_5_Classifier verdict disagree on keep/reject, THE Disagreement_Router SHALL route the paper to Stage 4 (LLM verifier)
2. WHEN the Blended_Confidence score falls between 0.40 and 0.70 (inclusive), THE Disagreement_Router SHALL route the paper to Stage 4 (LLM verifier)
3. WHEN neither disagreement nor borderline confidence is detected, THE Disagreement_Router SHALL accept the Stage_3_5_Classifier verdict as final without invoking the LLM
4. THE Disagreement_Router SHALL log the routing reason (disagreement or borderline confidence) for each LLM invocation

### Requirement 7: Batched LLM Verification

**User Story:** As a pipeline developer, I want papers to be verified by the LLM in batches, so that throughput increases and per-paper overhead decreases at scale.

#### Acceptance Criteria

1. WHEN multiple papers require LLM verification in a single pipeline run, THE Batched_Verifier SHALL group them into batches of up to 16 papers per Ollama call
2. THE Batched_Verifier SHALL format each batch as a structured JSON array prompt and expect a JSON array response from Ollama
3. IF Ollama returns an unparseable response for a batch, THEN THE Batched_Verifier SHALL split the batch in half and retry each sub-batch independently
4. IF a single-paper retry still fails to parse, THEN THE Batched_Verifier SHALL mark the paper for human review
5. THE Batched_Verifier SHALL respect the existing OLLAMA_TIMEOUT_SECONDS and OLLAMA_MAX_RETRIES configuration from BackendConfig
6. THE Batched_Verifier SHALL log batch size, success count, retry count, and split count per batch for performance monitoring

### Requirement 8: Embedding-Distance Semantic Cache

**User Story:** As a pipeline developer, I want near-duplicate papers to reuse cached LLM verdicts based on embedding similarity, so that semantically identical papers never trigger redundant LLM calls.

#### Acceptance Criteria

1. WHEN a paper is routed to Stage 4, THE Semantic_Cache SHALL first check if any previously verified paper has cosine similarity greater than 0.97 with the candidate
2. IF a cached verdict with similarity greater than 0.97 exists, THEN THE Semantic_Cache SHALL return the cached verdict without calling Ollama
3. WHEN the Semantic_Cache returns a cached verdict, THE Pipeline SHALL log the decision as "stage4_llm (semantic_cache)" with the similarity score
4. THE Semantic_Cache SHALL store the embedding and verdict of every LLM-verified paper for future cache lookups
5. THE Semantic_Cache SHALL operate independently from the existing content-hash cache in `data/processed/llm_cache.json`

### Requirement 9: Active Embedding Store Growth

**User Story:** As a pipeline developer, I want every confident decision at any stage to feed back into the Embedding_Store, so that the store grows richer with each pipeline run and classification improves over time.

#### Acceptance Criteria

1. WHEN any pipeline stage makes a confident keep decision (score ≥ 0.80), THE Pipeline SHALL append the paper embedding to the positive partition of the Embedding_Store
2. WHEN any pipeline stage makes a confident reject decision (score ≤ 0.20), THE Pipeline SHALL append the paper embedding to the negative partition of the Embedding_Store
3. THE Pipeline SHALL NOT append embeddings for borderline decisions (score between 0.20 and 0.80 exclusive) to either partition
4. THE Pipeline SHALL deduplicate before appending — a paper already present in the Embedding_Store (matched by DOI or PMID) SHALL NOT be added again

### Requirement 10: Hybrid Stacked Meta-Classifier

**User Story:** As a pipeline developer, I want a meta-classifier that combines all signal sources into one calibrated confidence score, so that routing decisions are optimally informed after sufficient data accumulates.

#### Acceptance Criteria

1. WHILE the Embedding_Store contains at least 2000 papers across both partitions, THE Meta_Classifier SHALL combine rule_score, embedding_positive_similarity, embedding_negative_similarity, and ml_probability into a single confidence output
2. THE Meta_Classifier SHALL be implemented as a stacked classifier (logistic regression or gradient-boosted model) trained on labeled data from LLM-verified papers
3. THE Meta_Classifier SHALL output a probability calibrated to reflect true precision (Platt scaling)
4. WHEN the Meta_Classifier is active, THE Pipeline SHALL use its output as the Blended_Confidence for routing decisions
5. IF the Embedding_Store contains fewer than 2000 papers, THEN THE Meta_Classifier SHALL remain inactive and the Pipeline SHALL use the existing per-stage routing logic

### Requirement 11: Platt Scaling Calibration

**User Story:** As a pipeline developer, I want confidence scores to be calibrated via Platt scaling, so that a reported confidence of 0.90 corresponds to 90% true precision.

#### Acceptance Criteria

1. WHEN the Meta_Classifier is trained, THE Pipeline SHALL fit a Platt scaling layer on a held-out validation set of LLM-verified papers
2. THE Platt scaling layer SHALL transform raw Meta_Classifier logits into calibrated probabilities
3. THE Pipeline SHALL persist the calibration parameters alongside the Meta_Classifier model file
4. WHEN calibration data is insufficient (fewer than 200 LLM-verified papers), THE Pipeline SHALL skip Platt scaling and use raw probabilities with a logged warning

### Requirement 12: Active Learning Retrain Job

**User Story:** As a pipeline developer, I want the existing weekly_refresh scheduler stub to trigger active learning retraining, so that the Meta_Classifier improves automatically as new verified data accumulates.

#### Acceptance Criteria

1. WHEN the weekly_refresh job executes, THE Active_Learning_Job SHALL retrain the Meta_Classifier using all available LLM-verified papers as ground truth
2. THE Active_Learning_Job SHALL only retrain when at least 100 new LLM-verified papers have accumulated since the last training run
3. WHEN retraining completes, THE Active_Learning_Job SHALL hot-reload the new model into the running pipeline without requiring a process restart
4. THE Active_Learning_Job SHALL log training metrics (F1, AUC, calibration error) and the count of training samples used
5. IF retraining produces a model with F1 below 0.80, THEN THE Active_Learning_Job SHALL discard the new model, retain the previous model, and log a warning

### Requirement 13: Embedding Query Latency Monitoring

**User Story:** As a pipeline developer, I want latency monitoring for embedding queries, so that performance degradation is detected before it impacts pipeline throughput at scale.

#### Acceptance Criteria

1. WHILE the Pipeline is processing papers, THE Embedding_Store SHALL record the wall-clock duration of each similarity query
2. WHEN the rolling average latency of embedding queries exceeds 200 milliseconds, THE Pipeline SHALL emit a warning log indicating potential scale bottleneck
3. THE Pipeline SHALL include per-run average and p95 embedding query latency in the pipeline metrics log

### Requirement 14: Monthly Drift Monitoring

**User Story:** As a pipeline developer, I want monthly sampling of automated decisions for manual review, so that classification drift is detected and corrected before precision degrades.

#### Acceptance Criteria

1. WHEN the monthly_rescan job executes, THE Drift_Monitor SHALL select a 1% random sample of papers that were automatically classified (not LLM-verified) in the preceding month
2. THE Drift_Monitor SHALL write the sampled papers (with their automated verdicts) to a dedicated review file for manual inspection
3. THE Drift_Monitor SHALL log the sample size and the distribution of automated decisions (keep vs reject) in the sample
4. WHEN the sample size would be fewer than 10 papers, THE Drift_Monitor SHALL increase the sampling rate to ensure at least 10 papers are selected

### Requirement 15: Pipeline Metrics Logging

**User Story:** As a pipeline developer, I want per-run metrics logged in JSONL format, so that pipeline performance is observable over time and stage resolution trends are tracked.

#### Acceptance Criteria

1. WHEN a pipeline run completes, THE Pipeline SHALL append one JSON record to a JSONL metrics file containing: timestamp, total papers processed, and per-stage resolution counts
2. THE metrics record SHALL include the percentage of papers resolved at each stage (Stage 1, Stage 2, Gate, Stage 3, Stage 3.5, Stage 4)
3. THE metrics record SHALL include the count of LLM calls made, semantic cache hits, and batch verification stats for the run
4. THE metrics record SHALL include the Embedding_Store size (positive count, negative count) at run completion
5. THE Pipeline SHALL write metrics to `data/metrics/pipeline_runs.jsonl`
