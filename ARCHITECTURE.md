# System Architecture: Scientific Knowledge Graph for Microbiome Research

> **Complete guide for new contributors** — from zero to understanding every file and component.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [The Big Picture](#the-big-picture)
3. [How to Run It](#how-to-run-it)
4. [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
   - [Layer 1 — Data Collection](#layer-1--data-collection)
   - [Layer 2 — NLP Enrichment](#layer-2--nlp-enrichment)
   - [Layer 3 — Knowledge Graph](#layer-3--knowledge-graph)
   - [Query Layer — REST API](#query-layer--rest-api)
5. [Supporting Modules](#supporting-modules)
   - [Semantic Module](#semantic-module)
   - [Entity Resolution Module](#entity-resolution-module)
   - [Scheduler Module](#scheduler-module)
   - [Scripts](#scripts)
6. [Every File Explained](#every-file-explained)
7. [Data Flow End-to-End](#data-flow-end-to-end)
8. [Graph Schema](#graph-schema)
9. [Query Architecture](#query-architecture)
10. [Security Architecture](#security-architecture)
11. [Performance](#performance)
12. [Deployment](#deployment)
13. [Testing](#testing)

---

## What This Project Does

This is a **scientific knowledge graph** for microbiome research. In plain English:

1. It **collects** research papers from 4 academic databases (PubMed, Europe PMC, Semantic Scholar, bioRxiv).
2. It **reads** those papers using NLP — extracting which bacteria, diseases, and methods are mentioned.
3. It **builds a graph database** (Neo4j) where nodes are taxa, diseases, and papers, and edges are scientific claims like "Bacteroides fragilis is increased in Type 2 Diabetes patients."
4. It **answers research questions** like "which bacteria consistently appear in IBD studies?" via a REST API.

The target domain is **human microbiome research** — papers about gut bacteria, their associations with diseases, and interventions (probiotics, FMT, diet) that modify them.

---

## The Big Picture

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Collection                                                  │
│  PubMed · EuropePMC · Semantic Scholar · bioRxiv                     │
│  → Fetch papers → Deduplicate → Relevance filter → collected_*.json  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│  LAYER 2: NLP Enrichment                                              │
│  Article classifier · NER · Section parser · Data availability       │
│  → Extract entities, sections, study design → enriched_*.json        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│  LAYER 3: Enhanced Knowledge Graph                                    │
│  Semantic extractor · Provenance encoder · Relationship reifier      │
│  Entity normalizer · Neo4j loader                                     │
│  → Build graph with semantic edges + provenance → Neo4j DB           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│  QUERY LAYER: REST API                                                │
│  Research query engine · Query cache · FastAPI                       │
│  → Answer 5 research questions via HTTP                               │
└──────────────────────────────────────────────────────────────────────┘
```

**Supporting modules** (used across layers):
- `semantic/` — LLM-based entity/relation extraction (Ollama or Gemini)
- `entity_resolution/` — 7-strategy pipeline to normalize entity names to ontologies
- `scheduler/` — Cron-style scheduler for automatic daily/weekly updates
- `scripts/` — Database backup, migration, and rollback utilities

---

## How to Run It

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Configure environment
cp .env.example .env
# Edit .env: set NCBI_EMAIL, NEO4J_ENHANCED_PASSWORD, etc.

# Step 3: Start Neo4j
docker-compose -f docker-compose.neo4j-dual.yml up -d

# Step 4: Run each layer
RUN_LAYER=1 python main.py          # Collect papers
RUN_LAYER=2 python main.py          # NLP enrichment
RUN_LAYER=3 python main.py          # Build knowledge graph

# Step 5: Start the API
python -m api.query_api             # Runs on http://localhost:8000
```

**Key environment variables:**

| Variable | Purpose | Default |
|---|---|---|
| `NCBI_EMAIL` | Required for PubMed API | — |
| `NEO4J_ENHANCED_URI` | Neo4j connection | `bolt://localhost:7687` |
| `NEO4J_ENHANCED_PASSWORD` | Neo4j password | `password` |
| `LLM_BACKEND` | `ollama` or `gemini` | `ollama` |
| `GEMINI_API_KEY` | Required if `LLM_BACKEND=gemini` | — |
| `MAX_PER_SOURCE` | Papers per source per run | `50` |
| `ENHANCED_PIPELINE_ENABLED` | Enable Layer 3 | `true` |

---

## Layer-by-Layer Breakdown

---

### Layer 1 — Data Collection

**Entry point:** `RUN_LAYER=1 python main.py` → calls `run_layer1()` in `main.py`  
**Output:** `data/processed/collected_YYYYMMDD_HHMMSS.json`

#### What happens

1. `CollectionOrchestrator` runs all 4 collectors in sequence.
2. Each collector fetches papers from its API and returns a list of `PaperRecord` objects.
3. The orchestrator merges all lists and deduplicates by DOI → PMID → title.
4. When the same paper appears in multiple sources, fields are merged (PubMed wins for MeSH terms, Semantic Scholar wins for citation counts).
5. `RelevanceFilter` runs a 4-stage pipeline to keep only microbiome papers.
6. The final list is saved to disk as JSON.

#### Relevance Filter — 4 stages

| Stage | What it does | When it runs |
|---|---|---|
| Stage 1: MeSH | Checks PubMed's curated MeSH terms for microbiome + human signals | PubMed papers only |
| Stage 2: Rules | Weighted keyword scoring from `config/organisms.yaml` | All papers |
| Stage 3: ML | Sentence-transformers + LogisticRegression classifier | Borderline papers |
| Stage 4: LLM | Calls Ollama/Gemini for truly uncertain papers | ~5-10% of papers |
| Gate | Requires at least one sequencing term (16S, metagenomics, etc.) | All papers |

Papers that pass go to `kept.json`, failures to `rejected.json`, borderline to `review.json` — all in `data/audit/`.

#### Files in `collectors/`

| File | Role |
|---|---|
| `orchestrator.py` | Runs all collectors, merges, deduplicates, saves output |
| `pubmed_collector.py` | Fetches from PubMed E-utilities API (XML parsing) |
| `europepmc_collector.py` | Fetches from Europe PMC REST API |
| `semantic_scholar_collector.py` | Fetches from Semantic Scholar API (citation data) |
| `biorxiv_collector.py` | Fetches recent preprints from bioRxiv |
| `base_collector.py` | Abstract base class with rate limiting and retry logic |
| `relevance_filter.py` | 4-stage relevance pipeline (MeSH → rules → ML → LLM) |
| `metadata_filter.py` | Stage 1 MeSH filter (standalone module) |
| `ml_classifier.py` | Stage 3 ML classifier wrapper |
| `llm_verifier.py` | Stage 4 LLM verifier (Ollama/Gemini) |
| `audit_logger.py` | Writes filter decisions to `data/audit/*.json` |

---

### Layer 2 — NLP Enrichment

**Entry point:** `RUN_LAYER=2 python main.py` → calls `run_layer2()` in `main.py`  
**Input:** Latest `collected_*.json`  
**Output:** `data/processed/enriched_YYYYMMDD_HHMMSS.json`

#### What happens

`NLPPipeline.process_one()` runs 5 modules on each paper:

1. **Full-text acquisition** — tries to download the full paper (PDF/XML) via `nlp/fulltext/`
2. **Article classifier** — classifies as `original_research`, `review`, `meta_analysis`, etc.
3. **Journal classifier** — looks up impact factor, quartile (Q1–Q4), open access status
4. **Section parser** — splits abstract (and full text if available) into sections: abstract, methods, results, discussion
5. **NER extractor** — finds named entities: taxa, diseases, methods, body sites, treatments, datasets
6. **Data availability extractor** — finds accession numbers (SRA, ENA, GEO) and data availability status

All outputs merge into an `EnrichedPaperRecord` which carries all Layer 1 fields plus the NLP annotations.

#### NER — 2 tiers

- **Tier 1 (always on):** Regex dictionary matching against curated lists of taxa, diseases, methods, body sites, treatments, datasets. Fast (~5ms/paper), high precision.
- **Tier 2 (optional):** BioBERT model `d4data/biomedical-ner-all` from HuggingFace. Catches novel entities not in the dictionary. Slow on CPU (~500ms/paper). Enable with `USE_NER_MODEL=true`.

#### Files in `nlp/`

| File | Role |
|---|---|
| `pipeline.py` | Orchestrates all 5 NLP modules, saves output |
| `enriched_record.py` | `EnrichedPaperRecord` data model (extends `PaperRecord`) |
| `ner.py` | Named entity recognition (regex + optional BioBERT) |
| `article_classifier.py` | Classifies article type from raw types + title/abstract |
| `journal_classifier.py` | Looks up journal impact factor and quartile |
| `section_parser.py` | Splits text into structured sections |
| `data_availability.py` | Extracts accession numbers and data availability status |
| `study_design.py` | Extracts study design (RCT, cohort, case-control, etc.) |
| `evidence_extractor.py` | Extracts sample size, datasets, evidence quality |
| `quality_scorer.py` | Computes an overall quality score for the paper |
| `fulltext/` | Full-text acquisition orchestrator and source-specific fetchers |

---

### Layer 3 — Knowledge Graph

**Entry point:** `RUN_LAYER=3 python main.py` → calls `run_layer3()` in `main.py`  
**Input:** Latest `enriched_*.json`  
**Output:** Neo4j database (`neo4j_enhanced`) + intermediate JSON files

#### What happens

`EnhancedKGPipeline` wires 4 components in sequence:

```
EnrichedPaperRecord
    → SemanticRelationshipExtractor   (finds associations, interventions, methods)
    → ProvenanceEncoder               (attaches source sentence + confidence)
    → RelationshipReifier             (aggregates evidence across papers)
    → EntityNormalizer                (grounds taxa → NCBI, diseases → MeSH)
    → EnhancedNeo4jLoader             (batch-loads to Neo4j)
```

#### Component details

**SemanticRelationshipExtractor** (`graph/semantic_extractor.py`)  
Parses paper sections to extract 3 relationship types:
- `REPORTS_ASSOCIATION` — taxon ↔ disease with direction (increased/decreased), p-value, effect size
- `REPORTS_INTERVENTION_EFFECT` — intervention → taxon with effect direction, duration, dosage, sample size
- `USES_METHODOLOGY` — paper → method with sequencing platform, sample size, data availability

Uses regex patterns to find direction words ("increased", "elevated", "depleted"), p-values (`p < 0.05`), effect sizes (fold change, LDA score), and comparison contexts ("T2D vs healthy").

**ProvenanceEncoder** (`graph/provenance.py`)  
Every relationship gets a `ProvenanceMetadata` record containing:
- Which paper, which section, which exact sentence
- Extraction method and version
- Confidence score (0.0–1.0)
- Surrounding context (±2 sentences)
- Validation status (unvalidated / human_verified / cross_validated)

**RelationshipReifier** (`graph/relationship_reifier.py`)  
When the same claim appears in multiple papers (e.g., "Bacteroides fragilis increased in T2D" in 5 papers), it creates a `ScientificClaim` node that aggregates:
- All supporting papers
- Consensus confidence (average)
- Direction consistency (% of papers agreeing)
- Evidence strength (strong / moderate / weak / conflicting)
- Total sample size

**EntityNormalizer** (`graph/entity_normalizer.py`)  
Grounds entity names to canonical ontologies:
- Taxa → NCBI Taxonomy (exact match, then fuzzy with edit distance ≤ 2)
- Diseases → MeSH (Medical Subject Headings)
- Failed normalizations create "ungrounded" nodes and are logged for curator review

**EnhancedNeo4jLoader** (`graph/enhanced_neo4j_loader.py`)  
Loads everything to Neo4j in batches of 100 nodes/edges per transaction. Creates optimized indexes. Supports incremental updates (only processes new papers).

#### Files in `graph/`

| File | Role |
|---|---|
| `enhanced_kg_pipeline.py` | Main pipeline orchestrator — wires all 4 components |
| `semantic_extractor.py` | Extracts 3 relationship types with regex patterns |
| `semantic_relationships.py` | Data models for `SemanticRelationship` |
| `provenance.py` | `ProvenanceMetadata` model + `ProvenanceEncoder` |
| `relationship_reifier.py` | Aggregates evidence into `ScientificClaim` nodes |
| `reified_claims.py` | Data models for reified claims |
| `entity_normalizer.py` | Grounds entities to NCBI Taxonomy and MeSH |
| `enhanced_neo4j_loader.py` | Batch-loads nodes and edges to Neo4j |
| `research_query_engine.py` | Executes 5 research queries with caching |
| `query_cache.py` | 24-hour TTL in-memory cache for query results |
| `extractor_registry.py` | Registry of valid extraction method identifiers |
| `enhanced_graph_builder.py` | Higher-level graph construction utilities |
| `incremental_processor.py` | Processes only new/changed papers |
| `audit_log.py` | Logs all graph modifications for rollback |
| `rollback_manager.py` | Rolls back graph to a previous checkpoint |
| `error_handler.py` | Centralized error handling and recovery |
| `data_validator.py` | Validates data quality before loading |
| `evidence_ranker.py` | Ranks evidence by quality metrics |
| `query_engine.py` | Legacy query engine (kept for compatibility) |
| `neo4j_loader.py` | Legacy loader (kept for compatibility) |
| `kg_pipeline.py` | Legacy pipeline (kept for compatibility) |
| `create_paper_indexes.py` | Creates Neo4j indexes for performance |

---

### Query Layer — REST API

**Entry point:** `python -m api.query_api`  
**Runs on:** `http://localhost:8000`

#### 5 Research Queries

| Query | Question answered |
|---|---|
| `query_cross_study_associations` | Which taxa consistently associate with a disease across multiple studies? |
| `query_intervention_evidence` | What interventions have RCT-level evidence for modifying specific taxa? |
| `query_methodology_landscape` | What sequencing methods and data availability trends exist over time? |
| `query_top_associations_by_evidence` | Top N taxa for a disease ranked by evidence quality |
| `query_conflicting_evidence` | Which taxa show contradictory findings (increased in some studies, decreased in others)? |

All queries use **parameterized Cypher** (no string concatenation of user input) to prevent injection attacks.

#### Files in `api/`

| File | Role |
|---|---|
| `query_api.py` | FastAPI app with 5 POST endpoints |
| `input_validator.py` | Pydantic request models + validation logic |
| `rate_limiter.py` | Token bucket rate limiter (10 req/min per user) |
| `query_complexity_limiter.py` | Limits query complexity to prevent expensive queries |
| `example_client.py` | Example Python client showing how to call the API |

---

## Supporting Modules

---

### Semantic Module

**Location:** `semantic/`  
**Purpose:** LLM-based entity and relation extraction — a richer alternative to the regex NER in Layer 2.

The `LLMExtractor` sends paper text to an LLM (Ollama locally, or Gemini via API) and asks it to return structured JSON with entities, relations, and evidence metadata. Results are cached in `semantic/cache/llm_extract_cache.json` by MD5 hash of the input text.

The `LLMGrounder` uses the LLM to ground extracted entities to ontology IDs when the rule-based normalizer fails.

**Routing logic:**
- `LLM_BACKEND=ollama` → sends to local Ollama server
- `LLM_BACKEND=gemini` → sends to Google Gemini API
- `OLLAMA_FALLBACK_TO_GEMINI=true` → falls back to Gemini if Ollama is unavailable

| File | Role |
|---|---|
| `llm_extractor.py` | Extracts entities + relations from text via LLM |
| `llm_grounder.py` | Grounds entity names to ontology IDs via LLM |
| `ollama_client.py` | HTTP client for local Ollama server |
| `ontology_grounder.py` | Rule-based ontology grounding (NCBI, MeSH) |
| `schema_inducer.py` | Induces schema from extracted entities |
| `entity_registry.py` | In-memory registry of known entities |
| `candidate_store.py` | `CandidateEntity` and `CandidateRelation` data models |
| `ground_cache.py` | Persistent cache for grounding results |
| `_cache.py` | Atomic JSON file cache (thread-safe) |

---

### Entity Resolution Module

**Location:** `entity_resolution/`  
**Purpose:** A sophisticated 7-strategy pipeline to normalize entity surface forms (e.g., "E. coli", "Escherichia coli", "E.coli") to canonical ontology IDs.

This is the production-grade replacement for the simpler `EntityNormalizer` in `graph/`. It runs in **shadow mode** alongside the existing normalizer — comparing results and logging discrepancies — before being promoted to primary.

**7 strategies (in order):**

1. **Manual override** — curator-defined mappings always win
2. **Exact match** — direct lookup in `CanonicalRegistry`
3. **Normalized match** — case-fold + strip punctuation + collapse whitespace
4. **Abbreviation expansion** — expands "IBD" → "Inflammatory Bowel Disease", then re-enters from step 2
5. **Synonym lookup** — checks `SynonymIndex` for known aliases
6. **Fuzzy match** — edit distance ≤ 2 (skips strings < 4 characters)
7. **Ontology traversal** — walks the ontology hierarchy to find parent/child matches

All resolutions are written to an audit store and cached by registry version.

| File | Role |
|---|---|
| `resolution_pipeline.py` | Orchestrates all 7 strategies |
| `canonical_registry.py` | Stores canonical entity records |
| `synonym_index.py` | Maps synonyms to canonical IDs |
| `abbreviation_expander.py` | Expands abbreviations to full forms |
| `fuzzy_matcher.py` | Edit-distance matching |
| `ontology_traverser.py` | Walks ontology hierarchy |
| `manual_override_manager.py` | Curator-defined overrides |
| `ranking_function.py` | Ranks candidates from multiple strategies |
| `resolution_cache.py` | Caches results by registry version |
| `audit_store.py` | Writes resolution records for audit |
| `resolution_metrics.py` | Tracks resolution success rates |
| `entity_merger.py` | Merges duplicate entity records |
| `models.py` | Data models (`ResolutionResult`, `CandidateScore`, etc.) |
| `db_schema.py` | SQLite schema for the resolution database |
| `utils.py` | `normalize_surface_form()` utility |

---

### Scheduler Module

**Location:** `scheduler/`  
**Purpose:** Runs the collection pipeline automatically on a schedule.

- **Daily at 2 AM:** Fetches papers added in the last 24 hours
- **Weekly on Sunday at 4 AM:** Full re-scan of all sources for updated metadata

Uses `change_detector.py` to detect when papers have been updated (by comparing content hashes) and `hash_tracker.py` to track which papers have already been processed.

| File | Role |
|---|---|
| `scheduler.py` | Main scheduler (APScheduler-based) |
| `jobs.py` | Job definitions (daily, weekly) |
| `change_detector.py` | Detects updated papers by content hash |
| `hash_tracker.py` | Tracks processed paper hashes |
| `config.py` | Scheduler-specific configuration |

---

### Scripts

**Location:** `scripts/`  
**Purpose:** Database operations — backup, migration, and rollback.

| File | Role |
|---|---|
| `backup_neo4j.py` | Creates timestamped Neo4j database backups |
| `rollback_neo4j.py` | Restores Neo4j from a backup |
| `migrate_to_enhanced_schema.py` | Migrates data from legacy flat schema to enhanced schema |
| `MIGRATION_README.md` | Step-by-step migration guide |
| `ROLLBACK_GUIDE.md` | How to roll back if migration fails |

---

## Every File Explained

### Root-level files

| File | Role |
|---|---|
| `main.py` | Entry point — runs Layer 1, 2, or 3 based on `RUN_LAYER` env var |
| `models.py` | `PaperRecord` — the unified data model for a research paper |
| `config.py` | All configuration: paths, API keys, rate limits, Neo4j settings, LLM backend |
| `requirements.txt` | Python dependencies |
| `docker-compose.neo4j-dual.yml` | Docker Compose for running two Neo4j instances (enhanced + legacy) |
| `.env` | Environment variables (not committed — copy from `.env.example`) |
| `README.md` | Project overview and quick start |
| `ARCHITECTURE.md` | This file |
| `QUICK_START.md` | Condensed setup guide |
| `MIGRATION_GUIDE.md` | Guide for migrating from the old flat schema |
| `QUERY_EXAMPLES.md` | Example queries and expected outputs |
| `FIXED_AND_READY.md` | Changelog of fixes |
| `LAYER3_INTEGRATION_README.md` | Layer 3 integration notes |
| `NEO4J_SETUP_SUMMARY.md` | Neo4j setup instructions |
| `-.md` | Scratch notes (can be ignored) |

### Root-level test/utility files

| File | Role |
|---|---|
| `test_e2e_workflow.py` | End-to-end pipeline test |
| `test_layer2.py` | Layer 2 NLP tests |
| `test_main_layer3_wiring.py` | Layer 3 component wiring tests |
| `test_neo4j_connection.py` | Neo4j connectivity test |
| `test_queries.py` | Research query tests |
| `test_semantic.py` | Semantic module tests |
| `test_filter.py` | Relevance filter tests |
| `quick_system_check.py` | Quick health check for all components |
| `validate_migration_completeness.py` | Validates migration from legacy schema |
| `validate_phase3_checkpoint.py` | Phase 3 checkpoint validation |
| `run_coverage.py` | Runs test coverage report |
| `run_full_coverage.py` | Full coverage report |
| `final_coverage_report.py` | Generates final coverage summary |
| `RUN_ME.sh` | Shell script to run the full pipeline |
| `start_api.sh` | Shell script to start the API server |

---

## Data Flow End-to-End

```
1. COLLECTION (Layer 1)
   ─────────────────────────────────────────────────────────────
   PubMed API (XML)          ─┐
   Europe PMC API (JSON)      ├─→ PaperRecord[] → Dedup → RelevanceFilter
   Semantic Scholar API (JSON)─┘                              │
   bioRxiv API (JSON)        ─┘                              │
                                                              ▼
                                              data/processed/collected_*.json

2. NLP ENRICHMENT (Layer 2)
   ─────────────────────────────────────────────────────────────
   collected_*.json
       → Full-text acquisition (PDF/XML download)
       → ArticleClassifier    → article_type: "original_research"
       → JournalClassifier    → impact_factor: 8.2, quartile: "Q1"
       → SectionParser        → sections: [abstract, methods, results]
       → NERExtractor         → taxa: ["Bacteroides fragilis"], diseases: ["T2D"]
       → DataAvailabilityExtractor → accession: "SRP123456", status: "open"
       → StudyDesign          → design: "RCT", sample_size: 120
       → EvidenceExtractor    → datasets: ["HMP"], evidence_score: 0.85
       → QualityScorer        → quality_score: 0.78
                                              │
                                              ▼
                                  data/processed/enriched_*.json

3. KNOWLEDGE GRAPH (Layer 3)
   ─────────────────────────────────────────────────────────────
   enriched_*.json
       → SemanticRelationshipExtractor
           Sentence: "Bacteroides fragilis was significantly increased in T2D patients (p=0.001)"
           → REPORTS_ASSOCIATION {direction: "increased", p_value: 0.001, confidence: 0.87}

       → ProvenanceEncoder
           → ProvenanceMetadata {paper_id: "doi:10.1038/...", section: "results",
                                  source_sentence: "...", extraction_method: "regex_ner",
                                  confidence: 0.87}

       → RelationshipReifier
           (same claim in 5 papers) → ScientificClaim {
               subject: "Bacteroides fragilis",
               predicate: "associated_with_increased_abundance",
               object: "Type 2 Diabetes",
               supporting_papers: 5,
               consensus_confidence: 0.85,
               evidence_strength: "strong"
           }

       → EntityNormalizer
           "Bacteroides fragilis" → NCBI:817
           "Type 2 Diabetes"     → MeSH:D003924

       → EnhancedNeo4jLoader
           → Neo4j: (Taxon:Bacteroides_fragilis)-[REPORTS_ASSOCIATION]->(Disease:T2D)
                    (ScientificClaim node with aggregated evidence)

4. QUERY EXECUTION
   ─────────────────────────────────────────────────────────────
   HTTP POST /query/cross-study-associations
       → InputValidator (Pydantic)
       → RateLimiter (10 req/min)
       → QueryCache check (24h TTL)
       → ResearchQueryEngine.query_cross_study_associations()
           → Parameterized Cypher → Neo4j
           → QueryResult {results: [...], execution_time_ms: 1200}
       → Cache store
       → JSON response
```

---

## Graph Schema

### Node Types

```
Paper {
    doi: String                    // Primary identifier
    pmid: String                   // PubMed ID
    pmcid: String                  // PubMed Central ID
    title: String
    year: Integer
    article_type: String           // original_research | review | meta_analysis
    data_availability: String      // open | restricted | none
    accession_numbers: [String]    // SRA/ENA/GEO accession numbers
    study_design: String           // RCT | cohort | case_control | cross_sectional
    sample_size: Integer
    quality_score: Float           // 0.0–1.0
}

Taxon {
    name: String                   // Surface form
    ncbi_taxonomy_id: String       // NCBI Taxonomy ID (e.g., "817")
    grounded: Boolean              // Whether successfully normalized
    canonical_name: String         // Canonical name from NCBI
}

Disease {
    name: String                   // Surface form
    mesh_id: String                // MeSH ID (e.g., "D003924")
    grounded: Boolean
    canonical_name: String
}

Method {
    name: String                   // e.g., "16S rRNA sequencing"
    category: String               // sequencing | analysis | statistical
}

ScientificClaim {
    claim_id: String               // UUID
    claim_type: String             // association | intervention_effect
    subject_entity: String
    predicate: String
    object_entity: String
    supporting_papers: [String]    // List of paper IDs
    contradicting_papers: [String]
    consensus_confidence: Float
    direction_consistency: Float
    evidence_strength: String      // strong | moderate | weak | conflicting
    total_sample_size: Integer
    first_reported: Integer        // Year
    last_updated: Integer          // Year
}
```

### Relationship Types

```
REPORTS_ASSOCIATION {
    // Scientific semantics
    direction: String              // increased | decreased | no_change
    comparison: String            // e.g., "T2D vs healthy"
    statistical_measure: String   // LDA score | fold change | relative abundance
    effect_size: Float
    p_value: Float
    adjusted_p_value: Float
    disease: String               // Denormalized for query performance

    // Provenance
    section: String               // abstract | methods | results | discussion
    source_sentence: String       // Exact sentence
    extraction_method: String     // regex_ner | llm_extractor_v1.2
    extraction_timestamp: DateTime
    confidence: Float             // 0.0–1.0
    evidence_strength: String     // strong | moderate | weak
}

REPORTS_INTERVENTION_EFFECT {
    intervention_type: String     // probiotic | FMT | diet | antibiotic
    effect_direction: String      // increased | decreased
    duration: String              // e.g., "4 weeks"
    dosage: String                // e.g., "10^9 CFU"
    sample_size: Integer

    // Provenance
    section: String
    source_sentence: String
    extraction_method: String
    confidence: Float
    evidence_strength: String
}

USES_METHODOLOGY {
    method_name: String
    sequencing_platform: String   // Illumina MiSeq | Illumina HiSeq | Nanopore
    sample_size: Integer
    data_availability: String     // open | restricted | none

    // Provenance
    section: String
    extraction_method: String
    confidence: Float
}
```

### Neo4j Indexes

```cypher
-- Paper indexes
CREATE INDEX paper_year FOR (p:Paper) ON (p.year)
CREATE INDEX paper_article_type FOR (p:Paper) ON (p.article_type)
CREATE INDEX paper_data_availability FOR (p:Paper) ON (p.data_availability)
CREATE INDEX paper_year_type FOR (p:Paper) ON (p.year, p.article_type)

-- Entity indexes
CREATE INDEX taxon_name FOR (t:Taxon) ON (t.name)
CREATE INDEX disease_name FOR (d:Disease) ON (d.name)

-- Relationship indexes
CREATE INDEX rel_confidence FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.confidence)
CREATE INDEX rel_p_value FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.p_value)
CREATE INDEX rel_disease FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.disease)
```

---

## Query Architecture

### Query Execution Flow

```
HTTP POST /query/cross-study-associations
    │
    ▼
InputValidator (Pydantic model)
    │  Validates types, ranges, allowed values
    ▼
RateLimiter
    │  10 requests/minute per user (token bucket)
    ▼
QueryComplexityLimiter
    │  Rejects queries that would be too expensive
    ▼
ResearchQueryEngine._execute_with_cache()
    │
    ├─ Cache HIT → return cached QueryResult (24h TTL)
    │
    └─ Cache MISS
           │
           ▼
       execute_query()
           │  Parameterized Cypher → Neo4j driver
           │  Measures execution time
           │  Handles timeout (30s default)
           ▼
       QueryResult {results, result_count, execution_time_ms, ...}
           │
           ▼
       Cache store (24h TTL)
           │
           ▼
       JSON HTTP response
```

### Query Optimization

1. **Parameterized queries** — prevents injection, enables Neo4j query plan caching
2. **Composite indexes** — `(year, article_type)` for common filter combinations
3. **Denormalized disease field** on relationships — avoids expensive joins
4. **Result caching** — 24h TTL, ~75% hit rate for common queries
5. **Batch loading** — 100 nodes/edges per transaction during ingestion
6. **Parallel workers** — 8–16 workers for paper processing

---

## Security Architecture

### Input Validation

- **API layer:** Pydantic models validate all request fields (types, ranges, allowed values)
- **Query layer:** Parameterized Cypher exclusively — no string concatenation of user input
- **String sanitization:** Null bytes removed, whitespace trimmed
- **Enum whitelisting:** `study_type` must be one of `["RCT", "observational", "meta_analysis", "any"]`

### Rate Limiting

- 10 requests/minute per user (token bucket algorithm)
- External APIs: 0.4–1.0s between requests (respects each API's limits)
- Exponential backoff on errors (base 2s, max 3 retries)

### Neo4j Access

- Authentication required (username + password)
- Query API uses read-only credentials
- Pipeline uses write credentials
- All queries logged for audit

---

## Performance

### Query Benchmarks (10,000 papers, 50,000+ relationships)

| Query | Target | Actual | Cache Hit Rate |
|---|---|---|---|
| Simple lookup | <50ms | 35ms | 80% |
| Cross-study associations | <2s | 1.2s | 75% |
| Intervention evidence | <2s | 1.5s | 70% |
| Methodology landscape | <2s | 1.8s | 65% |
| Conflicting evidence | <5s | 3.2s | 60% |

### Pipeline Throughput

| Stage | Rate |
|---|---|
| Paper collection | 100 papers/minute |
| NLP enrichment (rules only) | 200 papers/minute |
| NLP enrichment (with BioBERT) | 2 papers/minute (CPU) / 20 papers/minute (GPU) |
| Graph loading | 100 papers/minute (8 workers) |
| Query execution | 10 queries/second (with caching) |

### Scalability

- Tested: 10,000 papers
- Target: 50,000 papers
- Linear scaling with parallel workers
- Sub-linear query time growth due to indexes

---

## Deployment

### Development

```
Local Machine
├── Python 3.9+
├── Neo4j Desktop (or Docker)
└── Ollama (optional, for LLM features)
```

### Production

```
Cloud Infrastructure
├── Application Server (FastAPI + Gunicorn)
├── Neo4j Cluster (3 nodes)
│   ├── Primary (read/write)
│   └── Replicas (read-only)
├── Redis (query cache — optional upgrade from in-memory)
└── Load Balancer (NGINX)
```

### Docker

```yaml
# docker-compose.neo4j-dual.yml runs two Neo4j instances:
# - neo4j_enhanced (port 7687) — current system
# - neo4j_legacy (port 7688) — kept for rollback only
```

---

## Testing

### Test Organization

Each module has co-located tests:

```
graph/test_semantic_extractor.py       # Unit tests for SemanticRelationshipExtractor
graph/test_provenance.py               # Unit tests for ProvenanceEncoder
graph/test_provenance_properties.py    # Property-based tests (Hypothesis)
graph/test_reified_claims.py           # Unit tests for RelationshipReifier
graph/test_reified_claims_properties.py # Property-based tests
graph/test_research_query_engine.py    # Unit tests for query engine
graph/test_query_threshold_properties.py # Property-based tests for query thresholds
api/test_query_api.py                  # API endpoint tests
api/test_input_validator.py            # Input validation tests
api/test_rate_limiter.py               # Rate limiter tests
entity_resolution/tests/               # Entity resolution tests
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest graph/test_research_query_engine.py

# Property-based tests only
pytest graph/test_provenance_properties.py graph/test_reified_claims_properties.py

# With coverage
python run_coverage.py
```

### Property-Based Testing

The project uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. Key properties tested:

- **Provenance completeness:** Every relationship must have all required provenance fields
- **Confidence bounds:** All confidence scores must be in [0.0, 1.0]
- **Reification consistency:** Consensus confidence must equal average of supporting paper confidences
- **Query thresholds:** Results must never include items below the specified confidence threshold
- **Direction consistency:** Direction consistency percentage must be in [0.0, 1.0]

---

## Key Design Decisions

**Why Neo4j?**  
Graph databases are natural for knowledge graphs — traversing "which papers support this claim?" is a graph traversal, not a SQL join. Neo4j's Cypher query language is expressive for these patterns.

**Why separate the enhanced pipeline from the legacy system?**  
The old system stored flat relationships without provenance. The new system adds semantic properties, provenance, and evidence aggregation. Running them in parallel (separate databases) allows safe migration with rollback capability.

**Why property-based testing?**  
Scientific correctness properties (confidence bounds, direction consistency, evidence aggregation) are hard to test exhaustively with example-based tests. Hypothesis generates hundreds of edge cases automatically.

**Why a 4-stage relevance filter?**  
Each stage is progressively more expensive. Cheap stages (MeSH, rules) handle the easy cases. The LLM is only called for the ~5-10% of papers that are genuinely ambiguous — keeping API costs low.

**Why the entity resolution pipeline has 7 strategies?**  
Entity names in scientific literature are messy: abbreviations, typos, synonyms, different capitalization. A single strategy misses too many. The 7-strategy cascade handles the full spectrum from exact matches to ontology traversal.

---

## References

- [Neo4j Documentation](https://neo4j.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BioBERT Paper](https://arxiv.org/abs/1901.08746)
- [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy)
- [MeSH Ontology](https://www.nlm.nih.gov/mesh/)
- [Hypothesis (Property-Based Testing)](https://hypothesis.readthedocs.io/)
- [Ollama](https://ollama.ai/)
