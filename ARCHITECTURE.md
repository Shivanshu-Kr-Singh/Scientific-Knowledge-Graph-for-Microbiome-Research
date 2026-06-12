# Architecture: Scientific Knowledge Graph for Microbiome Research

> Last updated: June 2026  
> Reflects actual codebase state as of the latest pipeline run (27 papers processed end-to-end).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Layer 1 — Data Collection](#3-layer-1--data-collection)
4. [Layer 2 — NLP Enrichment](#4-layer-2--nlp-enrichment)
5. [Layer 3 — Knowledge Graph Construction](#5-layer-3--knowledge-graph-construction)
6. [Query Layer — REST API](#6-query-layer--rest-api)
7. [Supporting Subsystems](#7-supporting-subsystems)
8. [Data Flow & Persistence](#8-data-flow--persistence)
9. [Configuration & Environment](#9-configuration--environment)
10. [Infrastructure](#10-infrastructure)
11. [Testing Architecture](#11-testing-architecture)
12. [Performance Characteristics](#12-performance-characteristics)
13. [Known Limitations & Future Work](#13-known-limitations--future-work)

---

## 1. System Overview

This system transforms microbiome research literature into a queryable Neo4j knowledge graph. It answers five specific scientific questions:

1. Which taxa show consistent disease associations across multiple studies?
2. What interventions have RCT-level evidence for modifying specific taxa?
3. What are the methodology and data availability trends over time?
4. Which taxa have the strongest evidence quality for disease associations?
5. Which taxa show conflicting evidence requiring further investigation?

The pipeline operates in three sequential layers with a separate query layer:

```
[APIs] → Layer 1 (Collection) → Layer 2 (NLP) → Layer 3 (Graph) → Query API
```

Each layer writes its output to disk as timestamped JSON, making every stage independently resumable.

---

## 2. High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1 — DATA COLLECTION                           │
│                                                                               │
│  PubMed ─┐                                                                    │
│  EuropePMC ─┤                                                                 │
│  Semantic Scholar ─┼──► CollectionOrchestrator ──► RelevanceFilter ──► PMCEnricher │
│  OpenAlex ─┤             (deduplicate & merge)     (4-stage pipeline)  (full text) │
│  Crossref ─┤                                                                  │
│  CORE ─────┘                                                                  │
│  bioRxiv/medRxiv ─────────────────────► inline pre-filter                    │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │  collected_YYYYMMDD.json
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 2 — NLP ENRICHMENT                            │
│                                                                               │
│  ArticleClassifier ─┐                                                        │
│  JournalClassifier ─┤                                                         │
│  SectionParser ─────┼──► NLPPipeline.process_one() ──► EnrichedPaperRecord   │
│  NERExtractor ──────┤    (per paper, all modules)      (grounding included)  │
│  DataAvailability ──┤                                                         │
│  FullTextOrchestrator─┘                                                       │
│                                                                               │
│  EntityNormalizer (inline grounding at Layer 2 time, SQLite cache)            │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │  enriched_YYYYMMDD.json
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 3 — KNOWLEDGE GRAPH CONSTRUCTION                    │
│                                                                               │
│  EnhancedGraphBuilder ──► SemanticRelationshipExtractor                      │
│       │                     ├── extract_associations()     → REPORTS_ASSOCIATION │
│       │                     ├── extract_intervention_effects() → REPORTS_INTERVENTION_EFFECT │
│       │                     └── extract_methodology_usage()  → USES_METHODOLOGY   │
│       │                                                                       │
│       ├──► LLMTripleExtractor ──────────────────────────────► open-world triples │
│       │                                                                       │
│       └──► RelationshipReifier ──────────────────────────────► ScientificClaim nodes │
│                                                                               │
│  EnhancedKGPipeline (8–16 parallel workers, batch_size=100)                  │
│       └──► EnhancedNeo4jLoader ──────────────────────────────► Neo4j         │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │  Neo4j (bolt://localhost:7687)
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           QUERY LAYER — REST API                               │
│                                                                               │
│  FastAPI ──► ResearchQueryEngine                                              │
│               ├── query_cross_study_associations()                             │
│               ├── query_intervention_evidence()                                │
│               ├── query_methodology_landscape()                                │
│               ├── query_top_associations_by_evidence()                         │
│               └── query_conflicting_evidence()                                 │
│                                                                               │
│  QueryCache (in-memory, SHA-256 keyed, 24-hour TTL)                          │
│  InputValidator + RateLimiter (10 req/min) + QueryComplexityLimiter           │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1 — Data Collection

### Entry Point

`main.py → run_layer1()` → `CollectionOrchestrator.collect_all()`

Invoked with: `RUN_LAYER=1 python main.py`

### Data Sources

| Collector | File | Strategy | Notes |
|---|---|---|---|
| PubMed | `collectors/pubmed_collector.py` | Keyword + MeSH query, date range | `biopython` Entrez E-utilities; best for MeSH terms and article types |
| EuropePMC | `collectors/europepmc_collector.py` | Keyword query, date range | Best for PMCID and open-access detection |
| Semantic Scholar | `collectors/semantic_scholar_collector.py` | Keyword query, token-based pagination | Best for citation counts; continuation token persisted across runs |
| bioRxiv/medRxiv | `collectors/biorxiv_collector.py` | Date range only (no keyword search) | Inline relevance pre-filter after every 30 papers |
| OpenAlex | `collectors/openalex_collector.py` | Keyword + date | Polite pool: requires `NCBI_EMAIL` in `User-Agent` |
| Crossref | `collectors/crossref_collector.py` | Keyword + date | 50 req/sec polite pool |
| CORE | `collectors/core_collector.py` | Keyword + date | Requires `CORE_API_KEY` |

All collectors implement the same interface via `collectors/base_collector.py`:
- `collect(query, date_from, date_to, max_results, start_offset) → List[PaperRecord]`

### Orchestration

`CollectionOrchestrator` (`collectors/orchestrator.py`) coordinates all collectors sequentially:

1. Loads per-source fetch cursors from `data/processed/collector_cursors.json` (enables incremental runs)
2. Runs each collector
3. Deduplicates and merges across sources using:
   - DOI (primary key)
   - PMID (secondary key)
   - Normalized title (fuzzy fallback)
4. When a paper appears in multiple sources, fields are merged by source priority: `pubmed > europepmc > semantic_scholar > biorxiv`; boolean fields (e.g. `is_open_access`) are OR-ed; list fields (authors, mesh_terms) are union-deduplicated
5. Runs the 4-stage relevance filter on the merged pool
6. Runs PMC full-text enrichment for papers with a PMCID
7. Saves output to `data/processed/collected_YYYYMMDD_HHMMSS.json`

### Relevance Filter

`collectors/relevance_filter.py` implements a 4-stage pipeline with full audit logging:

```
Paper input
   │
   ├── PubMed papers → Stage 1: MeSH Metadata Filter
   │                   (microbiome MeSH + human MeSH → KEEP/REJECT/UNKNOWN)
   │
   ├─────────────────► Stage 2: Weighted Rule Scorer
   │                   (~60+ terms from config/organisms.yaml)
   │                   Score < 0.40 → REJECT
   │                   Score ≥ 0.70 → KEEP (after metagenomics gate)
   │                   0.40–0.70 → continue
   │
   ├─────────────────► Metagenomics Gate
   │                   (must mention sequencing/microbiome term — project requirement)
   │
   ├─────────────────► Stage 3: ML Classifier
   │                   (sentence-transformers + LogisticRegression)
   │                   Inactive until 500+ collected papers for training
   │                   Model saved at config/relevance_model.pkl
   │
   └─────────────────► Stage 4: LLM Verifier (borderline only)
                       Ollama qwen2.5:1.5b (or configured OLLAMA_VERIFIER_MODEL)
                       Results cached in data/processed/llm_cache.json by content hash
```

Every decision is logged to `data/audit/{kept,rejected,review}.json` via `collectors/audit_logger.py`.

The `metadata_filter.py` and `ml_classifier.py` modules are independently testable components extracted from the filter pipeline. The `llm_verifier.py` module is only instantiated lazily when a borderline paper reaches Stage 4.

### PMC Enrichment

`collectors/pmc_enricher.py` fetches structured full-text XML from NCBI PMC for papers that have a `pmcid`. This upgrades the paper record's `full_text` field and makes structured section parsing available in Layer 2. Maximum enrichments per run is configurable.

### Data Model

`models.py` defines `PaperRecord` (Pydantic v2), the normalized schema all collectors output. Key deduplication logic lives in `get_dedup_key()`: `doi` → `pmid` → `title[:80]`.

---

## 4. Layer 2 — NLP Enrichment

### Entry Point

`main.py → run_layer2()` → `NLPPipeline.process_all()`

Invoked with: `RUN_LAYER=2 python main.py`

Optional flags:
- `USE_NER_MODEL=true` — loads BioBERT (d4data/biomedical-ner-all, ~440 MB, GPU recommended)
- `USE_LLM=true` — enables Tier 3 Ollama LLM entity extraction

### NLP Pipeline

`nlp/pipeline.py` runs `process_one()` on each paper sequentially:

```
PaperRecord
   │
   ├── FullTextOrchestrator.fetch()    → full_text (Unpaywall PDF, PMC XML fallback)
   ├── ArticleClassifier.classify()   → article_type_normalized, confidence
   ├── JournalClassifier.classify()   → JournalInfo (quartile, IF, open_access)
   ├── SectionParser.parse_abstract() → List[ParsedSection]
   │   SectionParser.parse_full_text()  (if full text available)
   ├── NERExtractor.extract()         → List[NamedEntity]  (3-tier)
   ├── EntityNormalizer.normalize()   → grounding fields on each NamedEntity
   ├── DataAvailabilityExtractor      → DataAvailabilityInfo
   ├── StudyDesignExtractor           → study_design dict
   ├── EvidenceExtractor              → evidence_score, datasets
   └── QualityScorer                  → quality_score
                                    ↓
                           EnrichedPaperRecord
```

### NER — 3-Tier Entity Extraction

`nlp/ner.py` implements a cascading extraction strategy:

**Tier 1 — Rule-based Dictionary** (always runs)
- Regex patterns covering ~500+ known terms
- Entity types: `taxon`, `disease`, `method`, `body_site`, `treatment`, `metabolite`, `gene`, `protein`, `biomarker`, `pathway`, `population`, `dietary_component`, `immune_cell`, `clinical_outcome`, `environmental_factor`, `sequencing_platform`, `omics_feature`, `dataset`

**Tier 2 — BioBERT** (activated by `USE_NER_MODEL=true`)
- HuggingFace `transformers`: model `d4data/biomedical-ner-all`
- Catches novel entities not in the Tier 1 dictionary
- Returns entity spans with confidence scores

**Tier 3 — Ollama LLM** (activated by `USE_LLM=true`)
- Last resort for complex entities
- Configured via `OLLAMA_EXTRACTION_MODEL` (default: `llama3`)
- Gracefully skips on timeout or Ollama unavailability

Entities that don't match the 18 known categories (from Tier 2 or 3) are stored in `other_entities: Dict[str, List[str]]` for open-world discovery.

### Inline Entity Grounding (Entity Normalization)

After NER, each entity is grounded inline before the `EnrichedPaperRecord` is written to disk:

`graph/entity_normalizer.py` routes by entity type to authoritative APIs:

| Entity Type | Primary API | Ontology |
|---|---|---|
| taxon | NCBI Taxonomy (Entrez) | NCBI Taxonomy ID (ncbi:XXXX) |
| disease | NCBI MeSH (Entrez) | MeSH descriptor ID (mesh:DXXXXXX) |
| gene | NCBI Gene (Entrez) | NCBI Gene ID (ncbi_gene:XXXX) |
| protein | UniProt REST | UniProt accession (uniprot:PXXXXX) |
| metabolite | EMBL-EBI OLS4 (ChEBI) | ChEBI ID (chebi:XXXXX) |
| pathway | EMBL-EBI OLS4 (GO, PW) | KEGG/Reactome/GO ID |
| body_site | EMBL-EBI OLS4 (UBERON, BTO) | UBERON ID |
| immune_cell | EMBL-EBI OLS4 (Cell Ontology) | CL:XXXXXXX |
| method | EMBL-EBI OLS4 (OBI, EFO) | OBI:XXXXXXX |
| unknown | OLS cross-search | best available ontology |

Fallback chain per entity: `YAML abbreviation map → authoritative API → OLS cross-search → LLM grounder → ungrounded`

Results are cached in `graph/grounding_cache.db` (SQLite, `UNIQUE(entity_text, entity_type)`) so the same entity is never looked up twice across pipeline runs.

The `semantic/` module provides the LLM grounding layer (`semantic/llm_grounder.py`) which routes to Ollama (primary) or Gemini (fallback) based on `BACKEND_CONFIG` and caches results in `semantic/cache/llm_ground_cache.json`.

A more comprehensive entity resolution system is available in `entity_resolution/` (7-strategy pipeline including fuzzy matching, synonym index, abbreviation expansion, and ontology traversal). It can run in shadow mode alongside the simpler `EntityNormalizer` for comparison and gradual rollout.

### Output Schema

`nlp/enriched_record.py` defines `EnrichedPaperRecord`, which extends `PaperRecord` with:
- 18 named entity group lists (`taxa`, `diseases`, `methods`, `metabolites`, `genes`, etc.)
- `other_entities: Dict[str, List[str]]` for open-world entities
- `sections: List[ParsedSection]` with `section_type`, `header`, `content`
- `data_availability: DataAvailabilityInfo` with `status`, `accession_numbers`, `repositories`, `urls`
- `journal_info: JournalInfo` with `quartile`, `impact_factor`, `is_open_access`
- `study_design`, `evidence_score`, `quality_score`
- Grounding fields per entity: `canonical_name`, `ontology_id`, `ontology_name`, `grounded`, `grounding_confidence`, `grounding_source`

---

## 5. Layer 3 — Knowledge Graph Construction

### Entry Point

`main.py → run_layer3()` → `EnhancedKGPipeline.run()`

Invoked with: `RUN_LAYER=3 python main.py`

Key env vars:
- `ENHANCED_PIPELINE_ENABLED=true`
- `LOAD_TO_NEO4J=true`
- `ENHANCED_BATCH_SIZE=100`
- `ENHANCED_NUM_WORKERS=8`

### Pipeline Architecture

```
EnhancedKGPipeline.run()
   │
   ├── Split into batches of 100 papers
   │
   ├── ThreadPoolExecutor (8–16 workers)
   │   └── _process_batch() per batch
   │       └── EnhancedGraphBuilder.process_papers()
   │           ├── SemanticRelationshipExtractor.extract_associations()
   │           ├── SemanticRelationshipExtractor.extract_intervention_effects()
   │           ├── SemanticRelationshipExtractor.extract_methodology_usage()
   │           └── LLMTripleExtractor (open-world, only if USE_LLM=true)
   │
   ├── Merge all builders → _merge_builders()
   │   └── RelationshipReifier.reify_claim() per unique (subject, predicate, object) triple
   │
   └── EnhancedNeo4jLoader
       ├── create_indexes()
       ├── load_edges() (REPORTS_ASSOCIATION, REPORTS_INTERVENTION_EFFECT, USES_METHODOLOGY)
       ├── load_claims() (ScientificClaim nodes)
       └── load_open_world_triples() (RELATES_TO + canonical predicate)
```

### Semantic Relationship Extraction

`graph/semantic_extractor.py` — `SemanticRelationshipExtractor`:

**REPORTS_ASSOCIATION** (taxon–disease links)
- Source sections: `results`, `abstract`, `discussion`, `conclusion`
- Requires: ≥1 taxon AND ≥1 disease in paper
- Extracts: `direction` (increased/decreased/no_change via regex), `p_value`, `effect_size`, `statistical_measure`, `comparison_context`
- Confidence: 0.5 (direction only) → 1.0 (direction + p-value + effect size + stat measure)
- Minimum confidence: 0.5 to create a relationship

**REPORTS_INTERVENTION_EFFECT** (intervention → taxon effects)
- Article types: `original_research`, `meta_analysis`, `systematic_review`, `narrative_review`
- Extracts: `intervention_type` (probiotic/FMT/diet/antibiotic/prebiotic/synbiotic), `effect_direction`, `duration`, `dosage`, `sample_size`
- Only includes results with p < 0.05 or explicit significance statement

**USES_METHODOLOGY** (paper → method)
- Source sections: `methods` (fallback: `abstract`)
- Extracts: `method_name`, `sequencing_platform`, `sample_size`, `data_availability_status`

### Provenance Tracking

Every relationship carries a `ProvenanceMetadata` object (`graph/provenance.py`):

```python
ProvenanceMetadata:
  paper_id               # DOI | PMID | title
  section_type           # abstract | methods | results | discussion | ...
  source_sentence        # exact sentence supporting the relationship
  sentence_offset        # position in section
  extraction_method      # registered extractor ID (validated against extractor_registry)
  extraction_timestamp   # UTC datetime
  extractor_version      # "1.0"
  llm_prompt_hash        # SHA of prompt (if LLM extracted)
  confidence_score       # 0.0–1.0
  validation_status      # unvalidated | human_verified | cross_validated
  surrounding_context    # ±2 sentences
  figure_table_ref       # if claim references a figure/table
```

The `extraction_method` field is validated against `graph/extractor_registry.py` — only registered extractor IDs are accepted, preventing invalid provenance records.

### Evidence Reification

`graph/relationship_reifier.py` and `graph/reified_claims.py` aggregate relationships across papers:

When multiple papers report the same `(subject, predicate, object)` triple, `RelationshipReifier.reify_claim()` creates a `ScientificClaim` node that aggregates:
- `supporting_papers` and `contradicting_papers` (no overlap allowed)
- `consensus_confidence` (weighted average across papers)
- `effect_direction_consistency` (% agreement on dominant direction)
- `evidence_strength`: `strong` (p<0.01, RCT/meta) | `moderate` (p<0.05) | `weak` | `conflicting`
- `total_sample_size` (sum across supporting papers)
- `first_reported` / `last_updated` (temporal tracking)

Predicate normalization for deduplication key:
- Associations: `associated_with_{direction}`
- Interventions: `{intervention_type}_effect_{direction}`
- Methodology: `uses_methodology`
- Entity key: canonical ontology ID (if grounded) for merging synonyms

### Open-World Triple Extraction

`graph/llm_triple_extractor.py` uses Ollama to extract free-form `(subject, predicate, object)` triples from paper text (results/discussion sections prioritized, abstract fallback).

These supplement the three fixed relationship types without replacing them. Each triple is stored with:
- `raw_predicate`, `canonical_predicate`, `predicate_category`, `is_novel_predicate`
- `subject_type`, `object_type` (used as Neo4j node labels)
- `confidence`, `evidence`, `paper_id`, `section_type`

Active only when `USE_LLM=true` and Ollama is reachable.

### Neo4j Graph Schema

**Node types:**

| Label | Key Properties |
|---|---|
| `Paper` | `id` (DOI/PMID), `title`, `year`, `article_type`, `data_availability`, `accession_numbers` |
| `Taxon` | `id` (ontology ID), `name`, `canonical_name`, `ontology`, `grounded` |
| `Method` | `id`, `name`, `canonical_name`, `ontology`, `grounded` |
| `ScientificClaim` | `claim_id`, `claim_type`, `subject_entity`, `predicate`, `object_entity`, `evidence_strength`, `consensus_confidence`, `effect_direction_consistency`, `total_sample_size`, `first_reported`, `last_updated` |
| Dynamic (open-world) | `Disease`, `Gene`, `Metabolite`, `Pathway`, `ImmuneCell`, `Biomarker`, `Population`, `DietaryComponent`, `ClinicalOutcome`, etc. |

**Relationship types:**

| Type | From → To | Key Properties |
|---|---|---|
| `REPORTS_ASSOCIATION` | Paper → Taxon | `disease`, `direction`, `confidence`, `p_value`, `effect_size`, `evidence_strength`, `source_sentence`, `extraction_method`, `extraction_timestamp` |
| `REPORTS_INTERVENTION_EFFECT` | Paper → Taxon | `intervention_type`, `effect_direction`, `duration`, `dosage`, `sample_size`, `confidence`, `evidence_strength`, `source_sentence` |
| `USES_METHODOLOGY` | Paper → Method | `sequencing_platform`, `sample_size`, `data_availability`, `confidence` |
| `SUPPORTED_BY` | ScientificClaim → Paper | — |
| `CONTRADICTED_BY` | ScientificClaim → Paper | — |
| `RELATES_TO` / canonical | Entity → Entity | `raw_predicate`, `canonical_predicate`, `confidence`, `evidence` |

**Indexes created on startup:**

- `paper_year`, `paper_article_type`, `paper_data_availability`
- `paper_year_type` (composite)
- `taxon_name`, `disease_name`, `method_name`
- `rel_association_confidence`, `rel_association_p_value`
- `rel_intervention_confidence`, `rel_intervention_p_value`, `rel_intervention_type`
- `rel_association_evidence_consensus_composite`, `rel_intervention_evidence_consensus_composite`
- Dynamic indexes for common open-world entity types (Metabolite, Gene, Protein, etc.)

### Output Files

Saved to `data/processed/` after each Layer 3 run:

| File | Contents |
|---|---|
| `enhanced_edges_*.json` | All `EnhancedGraphEdge` objects with full provenance |
| `enhanced_claims_*.json` | All `ScientificClaim` objects |
| `enhanced_stats_*.json` | Pipeline statistics (relationship counts, timing) |
| `entities_*.json` | Unique nodes: papers, taxa, methods — with ontology IDs |
| `relationships_*.json` | All edges with human-readable from/to names |

---

## 6. Query Layer — REST API

### Entry Point

`python -m api.query_api` → FastAPI app on `http://localhost:8000`

Swagger UI: `http://localhost:8000/docs`

### Architecture

```
HTTP Request
   │
   ├── RateLimiter (10 req/min per client IP, api/rate_limiter.py)
   ├── Pydantic Request Model validation (field validators)
   ├── InputValidator (entity existence check in Neo4j, api/input_validator.py)
   ├── QueryComplexityLimiter (max 1000 results, max depth 5, api/query_complexity_limiter.py)
   │
   └── ResearchQueryEngine (graph/research_query_engine.py)
       ├── QueryCache.get() (in-memory, SHA-256 keyed)
       │   └── cache hit → return cached QueryResult
       │   └── cache miss → execute parameterized Cypher
       └── Neo4j session.run(cypher, parameters)
```

### Endpoints

| Endpoint | Method | Query Engine Method |
|---|---|---|
| `/query/cross-study-associations` | POST | `query_cross_study_associations()` |
| `/query/intervention-evidence` | POST | `query_intervention_evidence()` |
| `/query/methodology-landscape` | POST | `query_methodology_landscape()` |
| `/query/top-associations` | POST | `query_top_associations_by_evidence()` |
| `/query/conflicting-evidence` | POST | `query_conflicting_evidence()` |
| `/health` | GET | Neo4j connectivity check |
| `/cache/stats` | GET | Hit rate, size, TTL |
| `/cache/invalidate` | POST | Flush all cached entries |
| `/limits` | GET | Rate limit and complexity limits |

All query endpoints return `QueryResponse { success, query_result: QueryResult, error }`.

`QueryResult` carries: `query_id` (UUID), `query_description`, `results`, `result_count`, `execution_time_ms`, `executed_at`, `aggregation_method`, `confidence_threshold`, `timeout`, `error`.

### Security

- **Parameterized Cypher** — all user inputs go through Neo4j parameters dict, never concatenated
- **Input sanitization** — `sanitize_string_parameter()` strips null bytes and trims whitespace
- **Type validation** — `validate_parameter()` enforces type and allowed-value constraints
- **Rate limiting** — 10 queries per minute per user (configurable)
- **Result count limits** — max 1000 results per query
- **Query complexity limits** — max depth 5 enforced by `query_complexity_limiter`

### Query Cache

`graph/query_cache.py` — `QueryCache`:
- In-memory, thread-safe (uses `threading.Lock`)
- Cache key: SHA-256 of `{query_name}:{sorted_params_json}`
- TTL: 24 hours (configurable via `QUERY_CACHE_TTL_HOURS`)
- Automatic expiry on `get()`, periodic cleanup available
- Invalidated explicitly via `invalidate_all()` after new data loads

---

## 7. Supporting Subsystems

### Scheduler

`scheduler/` implements automated pipeline re-runs via APScheduler:

| Job | Schedule | Description |
|---|---|---|
| `daily_update` | Daily at 2:00 AM | Fetch papers added in last 24 hours |
| `weekly_refresh` | Sundays at 3:00 AM | Re-scan all sources for updated metadata |
| `monthly_rescan` | 1st of month at 4:00 AM | Full re-scan |

`scheduler/change_detector.py` and `scheduler/hash_tracker.py` detect when paper metadata has changed and trigger incremental updates. `graph/incremental_processor.py` processes only changed/new papers rather than re-running the full pipeline.

### Advanced Entity Resolution

`entity_resolution/` provides a full 7-strategy resolution pipeline as an upgrade path from the simpler `EntityNormalizer`:

| Strategy | Component | Description |
|---|---|---|
| 1 | `ManualOverrideManager` | Explicit curator overrides (confidence=1.0) |
| 2 | `CanonicalRegistry` | Exact string match |
| 3 | Normalized match | Case-fold, strip punctuation, collapse whitespace |
| 4 | `AbbreviationExpander` | Expand and re-enter from Strategy 2 (at most once) |
| 5 | `SynonymIndex` | Known synonyms lookup |
| 6 | `FuzzyMatcher` | Edit distance ≤ 2 (skipped for strings < 4 code points) |
| 7 | `OntologyTraverser` | Hierarchy search |

The `ResolutionPipeline` supports **shadow mode** — running both the simple normalizer and the advanced pipeline in parallel, logging discrepancies without changing behavior. Full rollout happens in task 16.1 by wiring in the spec1_normalizer.

Components: `abbreviation_expander.py`, `audit_store.py`, `canonical_registry.py`, `entity_merger.py`, `fuzzy_matcher.py`, `manual_override_manager.py`, `ontology_traverser.py`, `ranking_function.py`, `resolution_cache.py`, `resolution_metrics.py`, `synonym_index.py`.

All resolution decisions are audited via `audit_store.py` and metrics tracked via `resolution_metrics.py`.

### Semantic / LLM Grounding

`semantic/` provides the LLM-based entity grounding infrastructure:

- `ollama_client.py` — HTTP client for Ollama API with retry, timeout, configurable model
- `llm_grounder.py` — Routes to Ollama or Gemini, caches to `semantic/cache/llm_ground_cache.json`
- `llm_extractor.py` — LLM-based entity extraction (used by NERExtractor Tier 3)
- `ontology_grounder.py` — Ontology-based grounding (rule-based pre-filter before LLM)
- `schema_inducer.py` — Induces entity schemas from text
- `entity_registry.py` — Registry of known entities
- `candidate_store.py` — `CandidateEntity` model used across grounding modules
- `ground_cache.py` — Ground-truth cache interface

### Data Validation Queue

`data/` contains several operational data files:

| File | Purpose |
|---|---|
| `data/audit_log.db` | SQLite audit log for graph operations |
| `data/curator_review_queue.json` | Papers flagged for human review (relevance filter borderline cases) |
| `data/validation_queue.json` | Entities pending manual ontology validation |
| `data/incomplete_extraction_queue.json` | Papers where extraction partially failed |
| `data/conflicting_statistics_log.json` | Papers reporting contradictory statistical values |
| `data/query_timeout_log.json` | Query execution timeout events |

### Graph Utilities

`graph/` contains several utility modules beyond the core pipeline:

| Module | Purpose |
|---|---|
| `data_validator.py` | Validates graph data integrity before Neo4j loading |
| `error_handler.py` | Centralized error handling and recovery |
| `rollback_manager.py` | Rolls back partial graph loads on failure |
| `audit_log.py` | Graph-level audit logging |
| `predicate_registry.py` | Registry of known predicates with canonical mappings (SQLite: `predicate_registry.db`) |
| `extractor_registry.py` | Registry of valid extraction method IDs |
| `evidence_ranker.py` | Ranks evidence by quality |
| `query_engine.py` | Lower-level query infrastructure (base for `ResearchQueryEngine`) |
| `create_paper_indexes.py` | Standalone script to create Neo4j indexes |

---

## 8. Data Flow & Persistence

### File Hierarchy

```
data/
├── raw/                          # Cached raw API responses
├── processed/
│   ├── collected_YYYYMMDD.json   # Layer 1 output (PaperRecord list)
│   ├── enriched_YYYYMMDD.json    # Layer 2 output (EnrichedPaperRecord list)
│   ├── rejected_YYYYMMDD.json    # Relevance-filtered papers (audit)
│   ├── collector_cursors.json    # Per-source pagination cursors
│   ├── llm_cache.json            # LLM verifier results cache
│   ├── enhanced_edges_*.json     # Layer 3 relationship output
│   ├── enhanced_claims_*.json    # Layer 3 reified claims
│   ├── enhanced_stats_*.json     # Layer 3 pipeline statistics
│   ├── entities_*.json           # Layer 3 node catalog
│   └── relationships_*.json      # Layer 3 edge catalog (human-readable)
├── audit/
│   ├── kept.json                 # Papers kept by relevance filter
│   ├── rejected.json             # Papers rejected by relevance filter
│   ├── review.json               # Papers flagged for review
│   └── llm_verified.json         # Papers verified by LLM stage
├── training/                     # ML classifier training data
├── audit_log.db                  # SQLite audit log
├── curator_review_queue.json
├── validation_queue.json
├── incomplete_extraction_queue.json
├── conflicting_statistics_log.json
└── query_timeout_log.json

graph/
├── grounding_cache.db            # SQLite: entity grounding cache
├── entity_normalization.db       # SQLite: entity normalization failures
├── predicate_registry.db         # SQLite: predicate registry
└── triple_cache.json             # LLM triple extraction cache

semantic/
└── cache/
    └── llm_ground_cache.json     # LLM grounding cache

config/
├── organisms.yaml                # Relevance filter term lists + thresholds
└── relevance_model.pkl           # Trained ML classifier (after training run)

logs/
└── miner.log                     # Rotating log (10 MB limit, 30-day retention)
```

### Layer Handoffs

```
Layer 1 → Layer 2:  data/processed/collected_YYYYMMDD.json  (latest file loaded by NLPPipeline.load_latest())
Layer 2 → Layer 3:  data/processed/enriched_YYYYMMDD.json   (latest file loaded by NLPPipeline.load_latest())
Layer 3 → Query:    Neo4j database (neo4j_enhanced)
```

Each layer's `load_latest()` uses `glob("*_*.json")` sorted descending by timestamp — no explicit pointer needed.

---

## 9. Configuration & Environment

### Environment Variables (`.env`)

**Required:**
```
NCBI_EMAIL=your_email@example.com
```

**Neo4j:**
```
NEO4J_ENHANCED_URI=bolt://localhost:7687
NEO4J_ENHANCED_USER=neo4j
NEO4J_ENHANCED_PASSWORD=your_password
NEO4J_ENHANCED_DATABASE=neo4j_enhanced
```

**API Keys (optional but recommended):**
```
NCBI_API_KEY=...                    # 10 req/sec vs 3 req/sec without
SEMANTIC_SCHOLAR_API_KEY=...
GEMINI_API_KEY=...                  # Required if LLM_BACKEND=gemini or OLLAMA_FALLBACK_TO_GEMINI=true
```

**LLM Backend:**
```
LLM_BACKEND=ollama                  # "ollama" | "gemini"
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EXTRACTION_MODEL=llama3
OLLAMA_GROUNDING_MODEL=llama3
OLLAMA_VERIFIER_MODEL=llama3
OLLAMA_TIMEOUT_SECONDS=30
OLLAMA_MAX_RETRIES=3
OLLAMA_RETRY_BACKOFF_BASE=2.0
OLLAMA_FALLBACK_TO_GEMINI=false
GEMINI_EXTRACTION_MODEL=gemini-2.0-flash
GEMINI_GROUNDING_MODEL=gemini-2.5-flash
```

**Pipeline tuning:**
```
ENHANCED_PIPELINE_ENABLED=true
ENHANCED_BATCH_SIZE=100
ENHANCED_NUM_WORKERS=8
LOAD_TO_NEO4J=true
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL_HOURS=24
QUERY_TIMEOUT_SECONDS=30
ENTITY_NORMALIZATION_ENABLED=true
ENTITY_FUZZY_MATCH_THRESHOLD=2
PROVENANCE_CONTEXT_SENTENCES=2
MIN_CONFIDENCE_THRESHOLD=0.5
REIFICATION_ENABLED=true
```

**Search scope:**
```
MAX_PER_SOURCE=500                  # Papers per source per run (use 20–50 during development)
```

`config.py` performs typed validation of all LLM-related settings at import time, raising `ConfigurationError` for invalid values.

### `config/organisms.yaml`

Controls the relevance filter. Key sections:
- `positive_terms` — weighted terms that increase relevance score (e.g., `"gut microbiome": 0.8`)
- `negative_terms` — weighted terms that decrease score (e.g., `"zebrafish": -0.6`)
- `thresholds` — `keep: 0.70`, `review: 0.40`
- `metagenomics_gate` — `enabled: true`, list of required sequencing terms
- `mesh_keep` — MeSH terms that signal relevance
- `mesh_human_signal` — MeSH terms indicating human studies
- `mesh_animal_only` — MeSH terms indicating animal-only studies

---

## 10. Infrastructure

### Neo4j

- **Version**: Neo4j 5.x (Community Edition)
- **Database**: `neo4j_enhanced` (single database on Community Edition maps to `neo4j`)
- **Docker Compose**: `docker-compose.neo4j-dual.yml` (supports dual-instance setup for migration)
- **Browser**: `http://localhost:7474`
- **Bolt**: `bolt://localhost:7687`

### Python Stack

| Package | Version | Role |
|---|---|---|
| `pydantic` | 2.7.1 | Data models and validation |
| `neo4j` | 5.20.0 | Graph database driver |
| `fastapi` | 0.109.0 | REST API framework |
| `uvicorn` | 0.27.0 | ASGI server |
| `biopython` | 1.83 | NCBI Entrez API |
| `transformers` | 4.40.0 | BioBERT NER |
| `torch` | 2.3.0 | PyTorch backend |
| `spacy` | 3.7.4 | Rule-based NLP |
| `scikit-learn` | 1.4.2 | ML classifier |
| `sentence-transformers` | 2.7.0 | Paper embeddings |
| `tenacity` | 8.2.3 | HTTP retry logic |
| `apscheduler` | 3.10.4 | Scheduler |
| `loguru` | 0.7.2 | Structured logging |
| `hypothesis` | 6.155.0 | Property-based testing |
| `google-genai` | latest | Gemini API |

### Start Scripts

- `RUN_ME.sh` — full pipeline runner
- `start_api.sh` — starts the query API server

---

## 11. Testing Architecture

### Test Files

Tests are distributed alongside source modules:

**Graph layer** (`graph/test_*.py`):
- `test_research_query_engine.py` — query correctness
- `test_research_query_engine_caching.py` — cache behavior
- `test_semantic_extractor.py` — relationship extraction
- `test_provenance.py` — provenance metadata
- `test_provenance_properties.py`, `test_provenance_traceability_properties.py` — **property-based tests**
- `test_reified_claims.py` — claim aggregation
- `test_reified_claims_properties.py` — **property-based tests**
- `test_query_threshold_properties.py` — **property-based tests**
- `test_entity_normalizer.py` — entity grounding
- `test_relationship_reifier.py` — evidence aggregation
- `test_enhanced_graph_builder.py` — full graph building
- `test_enhanced_kg_pipeline.py` — end-to-end pipeline
- `test_enhanced_neo4j_loader.py` — Neo4j loading
- `test_audit_log.py`, `test_data_validator.py`, `test_error_handler.py`
- `test_rollback_manager.py`, `test_extractor_registry.py`
- `test_query_cache.py`, `test_query_performance.py`
- `test_incremental_processor.py`
- `test_batch_processing_throughput.py`, `test_scalability_10k_papers.py`
- `test_multi_paper_aggregation.py`, `test_comprehensive_coverage.py`
- `test_integration_graph_construction.py`
- `test_semantic_relationships.py`

**API layer** (`api/test_*.py`):
- `test_query_api.py`, `test_input_validator.py`
- `test_rate_limiter.py`, `test_query_complexity_limiter.py`

**Top-level integration tests**:
- `test_e2e_workflow.py` — end-to-end pipeline
- `test_main_layer3_wiring.py` — Layer 3 component wiring
- `test_layer2.py`, `test_filter.py`
- `test_semantic.py`, `test_queries.py`
- `test_neo4j_connection.py`

### Property-Based Testing

The project uses **Hypothesis** for correctness properties:
- Provenance completeness: all required fields populated for any valid extraction
- Provenance traceability: source sentences are substrings of section content
- Reified claims: consensus metrics always in [0.0, 1.0]
- Query thresholds: results above threshold are never returned below threshold
- Evidence strength: strong claims have p < 0.05, weak claims have p ≥ 0.05

Hypothesis database is at `.hypothesis/`.

### Running Tests

```bash
pytest                                    # all tests
pytest graph/test_research_query_engine.py  # specific module
pytest graph/test_provenance_properties.py  # property-based tests
pytest api/test_query_api.py
```

---

## 12. Performance Characteristics

### Observed Throughput (27-paper run, June 2026)

| Layer | Duration | Throughput |
|---|---|---|
| Layer 1 (Collection) | ~2 min | 27 papers from 4 sources |
| Layer 2 (NLP, GPU) | ~8 min | ~3.4 papers/min; ~17s/paper avg |
| Layer 3 (Graph) | ~5 min | 103 relationships, 77 claims, 71 open-world triples |

### Query Performance Benchmarks (10,000 papers, 50,000+ relationships)

| Query | Target | Observed |
|---|---|---|
| Simple lookup | <50ms | ~35ms |
| Cross-study associations | <2s | ~1.2s |
| Intervention evidence | <2s | ~1.5s |
| Methodology landscape | <2s | ~1.8s |
| Conflicting evidence | <5s | ~3.2s |

Cache hit rate: ~75% for common queries (24-hour TTL)

### Rate Limits

| Source | Delay | Effective Rate |
|---|---|---|
| PubMed (no key) | 0.40s | 2.5 req/sec |
| PubMed (with API key) | 0.12s | ~8 req/sec |
| EuropePMC | 0.50s | 2 req/sec |
| Semantic Scholar | 1.00s | 1 req/sec |
| bioRxiv | 0.50s | 2 req/sec |
| OpenAlex | 0.10s | 10 req/sec |
| Crossref | 0.02s | 50 req/sec |
| CORE | 0.60s | ~1.7 req/sec |

---

## 13. Known Limitations & Future Work

### Current Limitations

1. **ML classifier inactive** — Stage 3 ML classifier requires 500+ collected papers to train. Run `RUN_LAYER=train_filter python main.py` once sufficient data is collected.

2. **Neo4j Community Edition** — Community Edition only supports a single database named `neo4j`. The `NEO4J_ENHANCED_DATABASE` config value is overridden to `neo4j` in Community Edition deployments.

3. **LLM extraction reliability** — Tier 3 NER and open-world triple extraction can time out on long papers (300s limit). The pipeline handles this gracefully: papers are fully processed with Tier 1+2 data even when Tier 3 times out.

4. **Data availability detection** — 26/27 papers in the sample run reported `not_stated` for data availability. This is accurate but highlights that most papers in the 2024–2026 date range don't include structured data availability sections.

5. **Entity resolution pipeline** — The advanced 7-strategy `entity_resolution/` pipeline is implemented and tested but not yet wired into the main pipeline. The simpler `EntityNormalizer` is currently used. Shadow mode comparison is available but requires wiring in task 16.1.

6. **Deduplication key** — Title-based deduplication (`title[:80]`) used as fallback when DOI and PMID are both absent. This can produce false positives for papers with very similar titles.

7. **Journal classifier** — Many papers return `quartile: unknown` when the journal isn't in the classifier's reference list (14/27 in sample run).

### Planned Enhancements

- Full wiring of `entity_resolution/ResolutionPipeline` as the primary normalizer
- PostgreSQL (`sqlalchemy`) integration for relational metadata storage (configured but not yet used)
- Elasticsearch integration for full-text search (configured in `requirements.txt` but not yet wired)
- Redis/Celery for distributed task scheduling (configured in `requirements.txt`)
- Playwright-based JavaScript-rendered full-text extraction for paywalled journals
- Bidirectional linking between `ScientificClaim` nodes and `Taxon`/`Disease` nodes (currently claims link to Papers only)
