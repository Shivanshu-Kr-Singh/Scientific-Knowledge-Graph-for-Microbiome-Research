# System Architecture: Scientific Knowledge Graph for Microbiome Research

> **Complete guide for anyone new to this project** — no prior knowledge required.  
> Covers what the system does, how every component works, how to run it, and how data flows end-to-end.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Why It Exists](#2-why-it-exists)
3. [The Big Picture](#3-the-big-picture)
4. [Current Status](#4-current-status)
5. [Prerequisites & Setup](#5-prerequisites--setup)
6. [How to Run It](#6-how-to-run-it)
7. [Layer 1 — Data Collection](#7-layer-1--data-collection)
8. [Layer 2 — NLP Enrichment](#8-layer-2--nlp-enrichment)
9. [Layer 3 — Knowledge Graph](#9-layer-3--knowledge-graph)
10. [Query Layer — REST API](#10-query-layer--rest-api)
11. [Supporting Modules](#11-supporting-modules)
12. [Data Flow End-to-End](#12-data-flow-end-to-end)
13. [Graph Schema](#13-graph-schema)
14. [Query Architecture](#14-query-architecture)
15. [Data Directory Structure](#15-data-directory-structure)
16. [Configuration Reference](#16-configuration-reference)
17. [Testing](#17-testing)
18. [Performance](#18-performance)
19. [Every File Explained](#19-every-file-explained)

---

## 1. What This Project Does

This is a **scientific knowledge graph** for human microbiome research. In plain English:

1. It **collects** research papers automatically from 4 academic databases: PubMed, Europe PMC, Semantic Scholar, and bioRxiv.
2. It **filters** those papers using a multi-stage relevance pipeline so only genuine human microbiome studies are kept.
3. It **reads** each paper using NLP — identifying bacteria, diseases, methods, study designs, and statistical findings.
4. It **builds a graph database** (Neo4j) where nodes are taxa (bacteria), diseases, papers, and methods, and edges are scientific claims like *"Bacteroides fragilis is increased in Type 2 Diabetes patients (p=0.001, LDA score 3.2)"*.
5. It **tracks provenance** on every single claim — which paper, which section, which exact sentence, what extraction method, what confidence score.
6. It **aggregates evidence** across papers — when 5 papers all say the same thing, the system reifies that into a `ScientificClaim` node with consensus confidence and direction consistency.
7. It **answers research questions** via a REST API — e.g., "which bacteria consistently appear across RCT studies for IBD?"

**Target domain:** Human microbiome research — papers about gut, oral, skin, or lung bacteria and their associations with diseases, plus interventions (probiotics, FMT, diet, antibiotics) that modify them.

**Target users:** Researchers who want to survey literature programmatically rather than reading hundreds of papers manually.

---

## 2. Why It Exists

The traditional approach to literature review is slow and error-prone:
- A researcher manually searches PubMed, reads abstracts, and takes notes.
- Findings from different papers are not systematically compared.
- There is no easy way to ask "do all RCT papers agree that taxon X is decreased in disease Y?"

This system automates that process end-to-end — from fetching papers to answering cross-study questions — and records exactly where every fact came from so it can be audited or reproduced.


---

## 3. The Big Picture

The system is organized into four sequential layers. Each layer reads the output of the previous one and produces its own output file or database state.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Collection                                                         │
│                                                                              │
│  PubMed API (XML)           ─┐                                               │
│  Europe PMC API (JSON)       ├─→ PaperRecord[] → Dedup → RelevanceFilter    │
│  Semantic Scholar API (JSON) ┘                       │                       │
│  bioRxiv API (JSON)          ┘                       │                       │
│                                                      ▼                       │
│                              data/processed/collected_YYYYMMDD_HHMMSS.json  │
└──────────────────────────────────────────────────────┬──────────────────────┘
                                                       │
┌──────────────────────────────────────────────────────▼──────────────────────┐
│  LAYER 2: NLP Enrichment                                                     │
│                                                                              │
│  ArticleClassifier  → article_type: "original_research"                      │
│  JournalClassifier  → quartile: "Q1", impact_factor: 8.2                    │
│  SectionParser      → sections: [abstract, methods, results, discussion]    │
│  NERExtractor       → taxa: ["Bacteroides fragilis"], diseases: ["T2D"]     │
│  DataAvailability   → accession: "SRP123456", status: "open"                │
│  StudyDesign        → design: "RCT", sample_size: 120                       │
│  EvidenceExtractor  → datasets: ["HMP"], quality_score: 0.85                │
│                                                      │                       │
│                              data/processed/enriched_YYYYMMDD_HHMMSS.json  │
└──────────────────────────────────────────────────────┬──────────────────────┘
                                                       │
┌──────────────────────────────────────────────────────▼──────────────────────┐
│  LAYER 3: Enhanced Knowledge Graph                                           │
│                                                                              │
│  SemanticRelationshipExtractor  → 3 relationship types with rich properties  │
│  ProvenanceEncoder              → source sentence + confidence + method      │
│  RelationshipReifier            → aggregates evidence across papers          │
│  EntityNormalizer               → taxa→NCBI Taxonomy, diseases→MeSH         │
│  EnhancedNeo4jLoader            → batch-loads to Neo4j                       │
│                                                      │                       │
│                              Neo4j database (bolt://localhost:7687)          │
└──────────────────────────────────────────────────────┬──────────────────────┘
                                                       │
┌──────────────────────────────────────────────────────▼──────────────────────┐
│  QUERY LAYER: REST API                                                       │
│                                                                              │
│  InputValidator (Pydantic) → RateLimiter → QueryCache → QueryEngine         │
│                                                                              │
│  POST /query/cross-study-associations                                        │
│  POST /query/intervention-evidence                                           │
│  POST /query/methodology-landscape                                           │
│  POST /query/top-associations                                                │
│  POST /query/conflicting-evidence                                            │
│                                                                              │
│  http://localhost:8000                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Supporting modules** that are used across layers:
- `semantic/` — LLM-based extraction (Ollama or Gemini) as a higher-quality alternative to regex NER
- `entity_resolution/` — 7-strategy pipeline to normalize entity names to canonical ontology IDs
- `scheduler/` — APScheduler-based cron jobs for automatic daily/weekly updates
- `scripts/` — Database backup, migration, and rollback utilities


---

## 4. Current Status

As of the latest run (June 2026):

| Layer | Status | Output |
|-------|--------|--------|
| Layer 1 — Collection | ✅ Complete | `collected_20260602_015012.json` — 21 papers |
| Layer 2 — NLP Enrichment | ✅ Complete | `enriched_20260602_070900.json` — 21 enriched records |
| Layer 3 — Knowledge Graph | ⏳ Ready to run | Requires Neo4j running on port 7687 |
| Query Layer — REST API | ✅ Implemented | Runs on `http://localhost:8000` |

**Layer 2 summary from last run:**
- 21/21 papers processed, 0 errors
- Article types: 8 original research, 6 narrative review, 2 systematic review, 5 unknown
- Journal quartiles: Q1 (7), Q2 (2), unknown (12)
- Open access: 3 papers; data availability stated: 1 paper
- 2 Ollama LLM timeout events (papers 13 and 19) — fell back to empty Tier 3 extraction

**To run Layer 3:** Start Docker Desktop, then:
```bash
docker-compose -f docker-compose.neo4j-dual.yml up -d
RUN_LAYER=3 python main.py
```

Or to extract relationships to JSON only (no Neo4j needed):
```bash
LOAD_TO_NEO4J=false RUN_LAYER=3 python main.py
```

---

## 5. Prerequisites & Setup

### System requirements
- Python 3.12
- Docker Desktop (for Neo4j)
- 8 GB RAM minimum (BioBERT model needs ~2 GB, Neo4j needs ~2 GB)

### Python dependencies (key packages)

| Package | Purpose |
|---------|---------|
| `neo4j` | Neo4j graph database driver |
| `fastapi` + `uvicorn` | REST API framework |
| `pydantic` | Data validation and schemas |
| `transformers` + `torch` | BioBERT NER model (optional, ~440 MB download) |
| `biopython` | PubMed E-utilities XML parsing |
| `sentence-transformers` | ML classifier embeddings (Stage 3 relevance filter) |
| `requests` + `tenacity` | HTTP with automatic retry |
| `loguru` | Structured logging to file and console |
| `hypothesis` | Property-based testing |
| `apscheduler` | Cron-style scheduler |
| `python-dotenv` | `.env` file loading |

### Installation
```bash
cd /path/to/IP
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

All secrets and configuration live in `.env`. Never commit this file.

| Variable | Required | Purpose | Example |
|----------|----------|---------|---------|
| `NCBI_EMAIL` | Yes | PubMed API identification | `you@example.com` |
| `NCBI_API_KEY` | Recommended | Raises PubMed rate limit 3→10 req/s | from ncbi.nlm.nih.gov |
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended | 100 req/min vs 1 req/s without | from semanticscholar.org |
| `NEO4J_ENHANCED_URI` | Layer 3 | Neo4j connection string | `bolt://localhost:7687` |
| `NEO4J_ENHANCED_PASSWORD` | Layer 3 | Neo4j password | `microbiome2024` |
| `LLM_BACKEND` | No | `ollama` (local) or `gemini` (API) | `ollama` |
| `OLLAMA_BASE_URL` | If using Ollama | Ollama server address | `http://localhost:11434` |
| `OLLAMA_EXTRACTION_MODEL` | If using Ollama | Model name | `qwen2.5:1.5b` |
| `OLLAMA_TIMEOUT_SECONDS` | No | Request timeout | `120` |
| `GEMINI_API_KEY` | If backend=gemini | Google Gemini API key | from makersuite.google.com |
| `MAX_PER_SOURCE` | No | Papers per source per run | `10` (dev), `500` (prod) |
| `USE_NER_MODEL` | No | Enable BioBERT NER | `true` |
| `USE_LLM` | No | Enable Ollama/Gemini extraction | `true` |
| `LOG_LEVEL` | No | Logging verbosity | `INFO` |


---

## 6. How to Run It

### Step-by-step from scratch

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. (Layer 3 only) Start Neo4j via Docker
docker-compose -f docker-compose.neo4j-dual.yml up -d

# 3. Collect papers
RUN_LAYER=1 python main.py
# → Output: data/processed/collected_YYYYMMDD_HHMMSS.json

# 4. NLP enrichment
RUN_LAYER=2 python main.py
# → Output: data/processed/enriched_YYYYMMDD_HHMMSS.json
# Optional flags:
#   USE_NER_MODEL=true  → enables BioBERT (slower, catches more entities)
#   USE_LLM=true        → enables Ollama Tier 3 extraction

# 5. Build knowledge graph
RUN_LAYER=3 python main.py
# → Populates Neo4j database
# Optional flags:
#   LOAD_TO_NEO4J=false            → extract to JSON only, skip Neo4j
#   ENHANCED_BATCH_SIZE=50         → smaller batches (less memory)
#   ENHANCED_NUM_WORKERS=4         → fewer parallel workers

# 6. Start the REST API
python -m api.query_api
# → Runs on http://localhost:8000
# → Docs at http://localhost:8000/docs (Swagger UI)
```

### Development shortcuts

```bash
# Small test run (fast, ~2 minutes total for layers 1+2)
MAX_PER_SOURCE=5 RUN_LAYER=1 python main.py
USE_NER_MODEL=false USE_LLM=false RUN_LAYER=2 python main.py

# Train the ML relevance classifier (do this after a large Layer 1 run)
RUN_LAYER=train_filter python main.py

# Quick health check
python quick_system_check.py

# Validate Neo4j connection
python test_neo4j_connection.py
```

### The `RUN_ME.sh` script
A convenience shell script at the root that runs layers 1, 2, and 3 in sequence:
```bash
bash RUN_ME.sh
```

---

## 7. Layer 1 — Data Collection

**Entry point:** `RUN_LAYER=1 python main.py` → `run_layer1()` in `main.py`  
**Output:** `data/processed/collected_YYYYMMDD_HHMMSS.json`  
**Audit outputs:** `data/audit/kept.json`, `rejected.json`, `review.json`, `llm_verified.json`

### What happens step by step

1. **Audit files reset** — `kept.json`, `rejected.json`, `review.json`, `llm_verified.json` are cleared for a fresh run.
2. **CollectionOrchestrator** initializes all 4 collectors and runs them in sequence.
3. Each collector fetches papers from its API and returns a list of `PaperRecord` objects (defined in `models.py`).
4. The orchestrator **merges** all lists. When the same paper appears in multiple sources, fields are merged using a priority system: PubMed wins for MeSH terms, Semantic Scholar wins for citation counts.
5. **Deduplication** happens by DOI → PMID → content hash (in that priority order). One canonical record survives per paper.
6. **RelevanceFilter** runs a 4-stage pipeline on every remaining paper.
7. The final kept papers are saved to disk as a JSON array.

### The 4 collectors

#### PubMedCollector (`collectors/pubmed_collector.py`)
- Uses NCBI E-utilities API (esearch + efetch)
- Parses PubMed XML records
- Rate: 10 req/s with API key, 3 req/s without
- Captures: DOI, PMID, PMCID, MeSH terms, publication types, abstract, authors, journal

#### EuropePMCCollector (`collectors/europepmc_collector.py`)
- Uses Europe PMC REST API
- Returns JSON; parses into `PaperRecord`
- Rate: 2 req/s
- Additional value: often has open-access full text and accession numbers

#### SemanticScholarCollector (`collectors/semantic_scholar_collector.py`)
- Uses Semantic Scholar Graph API
- Rate: 1 req/s (registered key), higher without
- Additional value: citation counts, influential citations, fields of study

#### BioRxivCollector (`collectors/biorxiv_collector.py`)
- Fetches recent preprints from bioRxiv REST API
- Rate: 2 req/s
- Additional value: captures work not yet peer-reviewed

### The Relevance Filter — 4 stages

Every paper passes through these stages in order. The first stage that makes a confident decision stops the chain.

```
Paper
  │
  ▼
Stage 1: MeSH Check (PubMed papers only)
  │  If MeSH terms include both a microbiome term AND a human term → KEEP
  │  If MeSH terms clearly exclude microbiome → REJECT
  │  Otherwise → continue
  ▼
Stage 2: Keyword Rules
  │  Weighted scoring from config/organisms.yaml
  │  High score → KEEP, Low score → REJECT, Middle → continue
  │  GATE: Must contain at least one sequencing term (16S, metagenomics, WGS, etc.)
  ▼
Stage 3: ML Classifier
  │  Sentence-transformers embeds title+abstract
  │  LogisticRegression predicts probability (trained on pseudo-labeled data)
  │  High confidence → KEEP or REJECT, borderline → continue
  ▼
Stage 4: LLM Verifier (~5-10% of papers reach here)
     Calls Ollama (or Gemini) with structured prompt
     Returns: keep | reject | uncertain
     Decision recorded in data/audit/llm_verified.json
```

| Stage | Speed | Typical papers it handles |
|-------|-------|--------------------------|
| MeSH | ~1ms | ~60% of PubMed papers |
| Rules | ~5ms | ~25% of all papers |
| ML | ~50ms | ~10% of papers |
| LLM | ~5–30s | ~5% of papers |


### The PaperRecord data model (`models.py`)

Every paper that survives the filter is stored as a `PaperRecord`:

```json
{
  "doi":              "10.1038/s41586-024-07999-z",
  "pmid":             "38765432",
  "pmcid":            "PMC9876543",
  "title":            "Gut microbiome composition in Type 2 Diabetes...",
  "abstract":         "Background: ...",
  "authors":          ["Smith J", "Jones K"],
  "journal":          "Nature",
  "publication_year": 2024,
  "article_types":    ["Journal Article", "Randomized Controlled Trial"],
  "mesh_terms":       ["Microbiota", "Gastrointestinal Microbiome", "Diabetes Mellitus, Type 2"],
  "is_open_access":   true,
  "content_hash":     "a3f8b2c1...",
  "fetched_at":       "2024-05-21T02:00:00"
}
```

### Files in `collectors/`

| File | Role |
|------|------|
| `orchestrator.py` | Runs all 4 collectors, merges results, deduplicates, saves output. Also provides `load_latest()` for Layer 2 to read. |
| `pubmed_collector.py` | PubMed E-utilities XML fetcher and parser |
| `europepmc_collector.py` | Europe PMC REST API JSON fetcher |
| `semantic_scholar_collector.py` | Semantic Scholar API fetcher |
| `biorxiv_collector.py` | bioRxiv REST API preprint fetcher |
| `base_collector.py` | Abstract base class — rate limiting, retry logic, logging |
| `relevance_filter.py` | Orchestrates all 4 filter stages |
| `metadata_filter.py` | Stage 1: MeSH term checking |
| `ml_classifier.py` | Stage 3: sentence-transformers + LogisticRegression |
| `llm_verifier.py` | Stage 4: LLM-based verification (Ollama or Gemini) |
| `audit_logger.py` | Writes filter decisions to `data/audit/*.json` |

---

## 8. Layer 2 — NLP Enrichment

**Entry point:** `RUN_LAYER=2 python main.py` → `run_layer2()` in `main.py`  
**Input:** Latest `data/processed/collected_*.json` (loaded automatically by filename timestamp)  
**Output:** `data/processed/enriched_YYYYMMDD_HHMMSS.json`

### What happens step by step

`NLPPipeline.process_all()` iterates over every `PaperRecord` and runs 8 modules on each one:

```
PaperRecord
  │
  ├─→ 1. FullTextAcquisition   → tries to download PDF/XML from PMC, Unpaywall, etc.
  ├─→ 2. ArticleClassifier     → classifies article type
  ├─→ 3. JournalClassifier     → looks up impact factor, quartile, OA status
  ├─→ 4. SectionParser         → splits text into structured sections
  ├─→ 5. NERExtractor          → extracts named entities (3 tiers)
  ├─→ 6. DataAvailabilityExtractor → finds accession numbers and data status
  ├─→ 7. StudyDesignExtractor  → identifies study design and sample size
  └─→ 8. EvidenceExtractor     → extracts datasets, quality signals
       + QualityScorer         → computes overall quality score
  │
  ▼
EnrichedPaperRecord (all Layer 1 fields + NLP annotations)
```

All 8 modules run on every paper. Errors in one module do not stop the others — the field is left empty and processing continues.

### Module Details

#### ArticleClassifier (`nlp/article_classifier.py`)
Classifies each paper into one of:
- `original_research` — new experimental data
- `systematic_review` — structured literature synthesis
- `narrative_review` — non-systematic review
- `meta_analysis` — statistical aggregation of prior studies
- `case_report` — individual patient report
- `unknown` — cannot be determined

Uses PubMed publication type tags first; falls back to title/abstract keyword rules.

#### JournalClassifier (`nlp/journal_classifier.py`)
Looks up the journal name in a curated database to find:
- SCImago journal quartile (Q1–Q4)
- Impact factor (approximate)
- Open access status

Returns `unknown` if the journal is not in the database.

#### SectionParser (`nlp/section_parser.py`)
Splits the abstract (and full text if available) into named sections:
- `abstract`, `background`, `methods`, `results`, `discussion`, `conclusion`

Handles both structured abstracts (with explicit section headers) and unstructured ones (by keyword detection).

#### NERExtractor (`nlp/ner.py`) — 3 tiers

**Tier 1 — Regex dictionary (always active, ~5ms/paper):**  
Matches against curated lists of:
- Taxa: genus/species names, common names (e.g., "E. coli", "Escherichia coli")
- Diseases: condition names and abbreviations (e.g., "IBD", "Inflammatory Bowel Disease")
- Methods: sequencing and analysis methods (e.g., "16S rRNA", "metagenomics", "QIIME2")
- Body sites: gut, oral, skin, lung, vaginal
- Treatments: probiotics, FMT, antibiotics, dietary interventions
- Datasets: SRA/ENA/GEO accession number patterns

**Tier 2 — BioBERT model (optional, enable with `USE_NER_MODEL=true`, ~500ms/paper):**  
- Model: `d4data/biomedical-ner-all` from HuggingFace
- Downloaded on first use (~440 MB)
- Catches entities not in the Tier 1 dictionary — novel species names, less common disease names
- Results merged with Tier 1 output (deduplication by span overlap)

**Tier 3 — LLM extraction (optional, enable with `USE_LLM=true`, ~5–120s/paper):**  
- Sends the paper text to Ollama (default: qwen2.5:1.5b) or Gemini
- Structured JSON prompt returns entities and relations with confidence scores
- Timeout: 120s (configurable via `OLLAMA_TIMEOUT_SECONDS`)
- On timeout: logs warning, returns empty — does not block processing
- Results cached in `data/processed/llm_cache.json` by text MD5 hash

#### DataAvailabilityExtractor (`nlp/data_availability.py`)
Finds:
- Accession numbers matching patterns for SRA (`SRP*`, `SRX*`), ENA (`ERP*`, `ERX*`), GEO (`GSE*`)
- Data availability statement sections
- Open/restricted/not-stated classification

#### StudyDesignExtractor (`nlp/study_design.py`)
Identifies:
- Study design: RCT, cohort, case-control, cross-sectional, meta-analysis, in vitro
- Sample size (from "n=120", "120 patients", etc.)
- Comparison context ("patients vs healthy controls")

#### EvidenceExtractor (`nlp/evidence_extractor.py`)
Extracts:
- Named datasets (e.g., HMP, FINRISK, UK Biobank)
- Evidence quality signals (registered clinical trial, pre-registration, power calculation)
- Overall evidence score (0.0–1.0)

#### QualityScorer (`nlp/quality_scorer.py`)
Computes a composite quality score (0.0–1.0) from:
- Journal quartile (Q1 = 1.0, Q4 = 0.25)
- Study design (RCT = 1.0, case report = 0.1)
- Sample size (log-scaled)
- Data availability (open = +0.2 bonus)
- Evidence quality signals


### The EnrichedPaperRecord data model (`nlp/enriched_record.py`)

Extends `PaperRecord` with all NLP annotations:

```json
{
  "...all PaperRecord fields...",
  "article_type":        "original_research",
  "journal_quartile":    "Q1",
  "impact_factor":       8.2,
  "sections": {
    "abstract":  "Background: ...",
    "methods":   "We recruited 120 patients...",
    "results":   "Bacteroides fragilis was significantly increased...",
    "discussion": "These findings suggest..."
  },
  "entities": {
    "taxa":       [{"text": "Bacteroides fragilis", "span": [120, 140]}],
    "diseases":   [{"text": "Type 2 Diabetes", "span": [200, 215]}],
    "methods":    [{"text": "16S rRNA sequencing", "span": [300, 319]}],
    "body_sites": [{"text": "gut", "span": [50, 53]}],
    "treatments": [],
    "datasets":   [{"text": "SRP123456", "span": [450, 459]}]
  },
  "data_availability": {
    "status":      "open",
    "accessions":  ["SRP123456"],
    "repositories": ["NCBI SRA"]
  },
  "study_design": {
    "design":      "RCT",
    "sample_size": 120,
    "comparison":  "T2D patients vs healthy controls"
  },
  "quality_score": 0.85,
  "evidence_score": 0.78
}
```

### Files in `nlp/`

| File | Role |
|------|------|
| `pipeline.py` | Orchestrates all 8 modules, handles errors gracefully, saves output |
| `enriched_record.py` | `EnrichedPaperRecord` dataclass (extends `PaperRecord`) |
| `ner.py` | Named entity recognition — 3 tiers (regex + BioBERT + LLM) |
| `article_classifier.py` | Classifies article type from pub types and title/abstract |
| `journal_classifier.py` | Looks up journal quartile and impact factor |
| `section_parser.py` | Splits text into structured sections |
| `data_availability.py` | Extracts accession numbers and data availability status |
| `study_design.py` | Extracts study design, sample size, comparison context |
| `evidence_extractor.py` | Extracts datasets, evidence quality signals |
| `quality_scorer.py` | Computes composite quality score |
| `fulltext/` | Full-text acquisition: fetches PDFs/XMLs from PMC, Unpaywall, etc. |

---

## 9. Layer 3 — Knowledge Graph

**Entry point:** `RUN_LAYER=3 python main.py` → `run_layer3()` in `main.py`  
**Input:** Latest `data/processed/enriched_*.json`  
**Output:** Neo4j database at `bolt://localhost:7687` + intermediate JSON files

### What happens step by step

`EnhancedKGPipeline` wires 5 components in a sequential pipeline:

```
EnrichedPaperRecord[]
  │
  ▼
SemanticRelationshipExtractor
  │  Parses paper sections (especially results/abstract)
  │  Extracts 3 relationship types:
  │    REPORTS_ASSOCIATION      (taxon ↔ disease)
  │    REPORTS_INTERVENTION_EFFECT  (intervention → taxon)
  │    USES_METHODOLOGY         (paper → method)
  │  Captures: direction, p-value, effect size, comparison context
  ▼
ProvenanceEncoder
  │  Attaches to every relationship:
  │    source_sentence, section, extraction_method, confidence, timestamp
  ▼
RelationshipReifier
  │  Groups identical claims across papers
  │  Creates ScientificClaim nodes aggregating:
  │    supporting papers, consensus confidence, direction consistency,
  │    evidence strength, total sample size
  ▼
EntityNormalizer
  │  Grounds taxa names → NCBI Taxonomy IDs
  │  Grounds disease names → MeSH IDs
  │  Uses fuzzy matching (edit distance ≤ 2) for variants
  │  Ungrounded entities logged to curator review queue
  ▼
EnhancedNeo4jLoader
  │  Batch-loads nodes and relationships to Neo4j (100 per transaction)
  │  Creates optimized indexes on first run
  │  Supports incremental updates (skips already-processed papers)
  │  Handles connection errors with retry + exponential backoff
  ▼
Neo4j database (neo4j_enhanced)
```

### Component 1: SemanticRelationshipExtractor (`graph/semantic_extractor.py`)

Parses each paper's sections looking for three types of claims.

**REPORTS_ASSOCIATION** — Taxon associated with a disease:
```
Sentence: "Bacteroides fragilis was significantly increased in T2D patients compared 
           to healthy controls (LDA score 3.2, p=0.001)"

Extracted:
  subject:   "Bacteroides fragilis"
  predicate: REPORTS_ASSOCIATION
  object:    "Type 2 Diabetes"
  direction: "increased"
  comparison: "T2D patients vs healthy controls"
  statistical_measure: "LDA score"
  effect_size: 3.2
  p_value: 0.001
```

Detection uses regex patterns for:
- Direction words: "increased", "elevated", "higher", "enriched" → increased; "decreased", "depleted", "lower", "reduced" → decreased
- P-value patterns: `p[=<]\s*0?\.\d+`, `p-value`, `adjusted p`
- Effect size patterns: fold change, LDA score, relative abundance, odds ratio
- Comparison context: "X vs Y", "X compared to Y", "X relative to Y"

**REPORTS_INTERVENTION_EFFECT** — An intervention modifies a taxon:
```
Sentence: "Probiotic supplementation with Lactobacillus acidophilus for 8 weeks 
           significantly increased Akkermansia muciniphila abundance (n=60)"

Extracted:
  intervention_type: "probiotic"
  effect_direction:  "increased"
  taxon:             "Akkermansia muciniphila"
  duration:          "8 weeks"
  sample_size:       60
```

**USES_METHODOLOGY** — Paper uses a sequencing method:
```
Sentence: "Stool samples were analyzed by 16S rRNA sequencing on the Illumina MiSeq platform"

Extracted:
  method_name:         "16S rRNA sequencing"
  sequencing_platform: "Illumina MiSeq"
  data_availability:   (from DataAvailabilityExtractor result)
```


### Component 2: ProvenanceEncoder (`graph/provenance.py`)

Every relationship gets a `ProvenanceMetadata` record. This is critical for scientific credibility — every edge in the graph can be traced back to its exact source.

```python
ProvenanceMetadata(
    paper_id          = "doi:10.1038/s41586-024-07999-z",
    section           = "results",
    source_sentence   = "Bacteroides fragilis was significantly increased...",
    surrounding_context = ["Previous sentence.", "Next sentence."],  # ±2 sentences
    extraction_method  = "regex_ner_v1.0",   # or "llm_extractor_v1.2"
    extraction_version = "1.0",
    extraction_timestamp = "2026-06-02T12:39:00Z",
    confidence        = 0.87,               # 0.0–1.0
    validation_status = "unvalidated"       # unvalidated | human_verified | cross_validated
)
```

Confidence scoring rules:
- Base confidence from extraction method (LLM: 0.8, BioBERT: 0.75, regex: 0.7)
- Boosted if: p-value present (+0.1), effect size present (+0.05), from results section (+0.05)
- Reduced if: from abstract only (-0.1), no statistical measure (-0.05)

### Component 3: RelationshipReifier (`graph/relationship_reifier.py`)

When the same scientific claim appears in multiple papers, the reifier aggregates them into a single `ScientificClaim` node. This is what makes the system answer questions like "5 out of 5 RCT papers agree that Bacteroides fragilis is increased in T2D."

Matching logic: Two relationships describe the same claim if they share the same (subject entity, predicate type, object entity) triple, after entity normalization.

```
Paper 1: Bacteroides fragilis ↑ in T2D (confidence 0.87)
Paper 2: Bacteroides fragilis ↑ in T2D (confidence 0.82)
Paper 3: Bacteroides fragilis ↑ in T2D (confidence 0.91)
Paper 4: Bacteroides fragilis ↑ in T2D (confidence 0.85)
Paper 5: Bacteroides fragilis ↓ in T2D (confidence 0.73)  ← contradicts

→ ScientificClaim:
    subject_entity:          "Bacteroides fragilis"
    predicate:               "associated_with_increased_abundance"
    object_entity:           "Type 2 Diabetes"
    supporting_papers:       [paper1, paper2, paper3, paper4]
    contradicting_papers:    [paper5]
    consensus_confidence:    0.8625   (average of 4 supporting)
    direction_consistency:   0.80     (4/5 papers agree on "increased")
    evidence_strength:       "strong" (≥3 papers, ≥0.8 confidence, ≥0.75 consistency)
    total_sample_size:       450      (sum across supporting papers)
```

Evidence strength classification:
- `strong`: ≥3 papers, consensus confidence ≥0.8, direction consistency ≥0.75
- `moderate`: ≥2 papers, confidence ≥0.65
- `weak`: only 1 paper, or low confidence
- `conflicting`: direction consistency < 0.5

### Component 4: EntityNormalizer (`graph/entity_normalizer.py`)

Grounds surface form names to canonical ontology identifiers so that "E. coli", "Escherichia coli", and "E.coli" all map to the same node.

```
"Bacteroides fragilis" → NCBI Taxonomy ID: 817
"E. coli"             → NCBI Taxonomy ID: 562
"Type 2 Diabetes"     → MeSH ID: D003924
"IBD"                 → MeSH ID: D015212 (after abbreviation expansion)
```

Strategy (in order of priority):
1. Exact match in local NCBI/MeSH database
2. Case-folded + punctuation-stripped match
3. Abbreviation expansion then re-lookup
4. Fuzzy match with edit distance ≤ 2 (skips strings shorter than 4 chars)
5. Ontology parent/child traversal

Failed normalizations create `grounded=false` nodes and are logged to `data/curator_review_queue.json` for manual review.

### Component 5: EnhancedNeo4jLoader (`graph/enhanced_neo4j_loader.py`)

Writes everything to Neo4j in batches:
- 100 nodes/edges per Cypher transaction (configurable via `ENHANCED_BATCH_SIZE`)
- Uses `MERGE` statements so re-running is safe (idempotent)
- Creates indexes after first load
- `IncrementalProcessor` (`graph/incremental_processor.py`) tracks which paper DOIs have been processed, so re-running only processes new papers

### Supporting graph components

| File | Role |
|------|------|
| `enhanced_kg_pipeline.py` | Main orchestrator — wires all 5 components, handles config |
| `semantic_relationships.py` | Data models: `SemanticRelationship`, `AssociationRelationship`, etc. |
| `reified_claims.py` | Data models: `ScientificClaim`, `EvidenceAggregation` |
| `predicate_registry.py` | Registry of valid predicate names with validation |
| `extractor_registry.py` | Registry of valid extraction method identifiers |
| `enhanced_graph_builder.py` | Higher-level graph construction utilities |
| `research_query_engine.py` | Executes 5 research queries against Neo4j |
| `query_cache.py` | 24-hour TTL in-memory cache for query results |
| `query_engine.py` | Legacy query engine (kept for compatibility) |
| `neo4j_loader.py` | Legacy loader (kept for compatibility) |
| `kg_pipeline.py` | Legacy pipeline (kept for compatibility) |
| `create_paper_indexes.py` | Creates Neo4j indexes for query performance |
| `audit_log.py` | Logs all graph modifications (for rollback) |
| `rollback_manager.py` | Rolls back graph to a previous checkpoint |
| `error_handler.py` | Centralized error handling and recovery strategy |
| `data_validator.py` | Validates data quality before loading to Neo4j |
| `evidence_ranker.py` | Ranks relationships by evidence quality |
| `llm_triple_extractor.py` | LLM-based triple extraction (alternative to regex) |
| `incremental_processor.py` | Tracks processed papers to skip on re-run |


---

## 10. Query Layer — REST API

**Entry point:** `python -m api.query_api`  
**Runs on:** `http://localhost:8000`  
**Swagger docs:** `http://localhost:8000/docs`  
**OpenAPI schema:** `http://localhost:8000/openapi.json`

The API wraps the `ResearchQueryEngine` in a FastAPI app with input validation, rate limiting, and request complexity limits.

### Request lifecycle

```
HTTP POST /query/cross-study-associations
  │
  ▼
InputValidator (Pydantic model)
  │  Validates field types, allowed enum values, numeric ranges
  │  Rejects bad requests with 422 Unprocessable Entity
  ▼
RateLimiter (api/rate_limiter.py)
  │  Token bucket: 10 requests/minute per user (identified by IP)
  │  Rejects with 429 Too Many Requests when exceeded
  ▼
QueryComplexityLimiter (api/query_complexity_limiter.py)
  │  Estimates query cost based on parameters
  │  Rejects overly broad queries that would be very expensive
  ▼
ResearchQueryEngine._execute_with_cache()
  │
  ├─ Cache HIT  → return cached QueryResult immediately (24h TTL, ~50ms)
  │
  └─ Cache MISS
        │
        ▼
    Parameterized Cypher → Neo4j driver
    Timeout: 30s (configurable via QUERY_TIMEOUT_SECONDS)
        │
        ▼
    QueryResult {results, result_count, execution_time_ms, query_description}
        │
        ▼
    Store in cache
        │
        ▼
    JSON HTTP response
```

### The 5 research queries

#### Query 1: Cross-Study Disease-Microbiome Associations
**Endpoint:** `POST /query/cross-study-associations`  
**Question:** Which gut microbiome taxa show consistent association with a disease across multiple studies?

```bash
curl -X POST http://localhost:8000/query/cross-study-associations \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "Type 2 Diabetes",
    "study_type": "RCT",
    "min_papers": 3,
    "confidence_threshold": 0.7,
    "require_open_data": true
  }'
```

Returns per taxon: paper count, consensus confidence, consensus direction, direction consistency, per-direction breakdown, paper IDs.

#### Query 2: Intervention Effectiveness Evidence
**Endpoint:** `POST /query/intervention-evidence`  
**Question:** What interventions (probiotics, FMT, diet, antibiotics) have evidence for modifying specific taxa?

```bash
curl -X POST http://localhost:8000/query/intervention-evidence \
  -H "Content-Type: application/json" \
  -d '{
    "intervention_types": ["probiotic", "FMT", "diet"],
    "min_sample_size": 50,
    "evidence_strength": "strong"
  }'
```

Returns per intervention-taxon pair: effect direction, paper count, total sample size, avg confidence, paper IDs.

#### Query 3: Methodology Landscape and Data Availability
**Endpoint:** `POST /query/methodology-landscape`  
**Question:** What sequencing methods were used each year, and what fraction of papers deposited data?

```bash
curl -X POST http://localhost:8000/query/methodology-landscape \
  -H "Content-Type: application/json" \
  -d '{
    "year_start": 2020,
    "year_end": 2024,
    "sequencing_methods": ["16S rRNA sequencing", "shotgun metagenomics"],
    "require_deposited_data": true
  }'
```

Returns per method per year: total papers, papers with data, data availability %, NCBI SRA count, ENA count.

#### Query 4: Top Associations by Evidence Quality
**Endpoint:** `POST /query/top-associations`  
**Question:** What are the top N taxa associated with a disease, ranked by evidence quality?

```bash
curl -X POST http://localhost:8000/query/top-associations \
  -H "Content-Type: application/json" \
  -d '{"disease": "IBD", "top_n": 10, "min_confidence": 0.7}'
```

Returns ranked list with paper count, avg confidence, direction, consistency.

#### Query 5: Conflicting Evidence Detection
**Endpoint:** `POST /query/conflicting-evidence`  
**Question:** Which taxa show contradictory findings (increased in some studies, decreased in others)?

```bash
curl -X POST http://localhost:8000/query/conflicting-evidence \
  -H "Content-Type: application/json" \
  -d '{"disease": "Crohn'\''s Disease", "min_papers_per_direction": 2}'
```

Returns taxa with both increased and decreased papers, per-direction counts and percentages, and paper metadata.

### Files in `api/`

| File | Role |
|------|------|
| `query_api.py` | FastAPI app with 5 POST endpoints, startup/shutdown Neo4j connection |
| `input_validator.py` | Pydantic request/response models and validation logic |
| `rate_limiter.py` | Token bucket rate limiter (10 req/min per IP) |
| `query_complexity_limiter.py` | Limits query cost to prevent expensive queries |
| `example_client.py` | Python client showing how to call all 5 endpoints |
| `README.md` | API-specific documentation |
| `test_query_api.py` | FastAPI TestClient integration tests |
| `test_input_validator.py` | Pydantic validation tests |
| `test_rate_limiter.py` | Rate limiter unit tests |
| `test_query_complexity_limiter.py` | Complexity limiter tests |


---

## 11. Supporting Modules

### Semantic Module (`semantic/`)

**Purpose:** LLM-based entity and relation extraction — a higher-quality alternative to the regex NER in Layer 2, trading speed for richer output.

The `LLMExtractor` sends paper text to an LLM and asks it to return structured JSON containing entities, relations, and evidence metadata. Results are cached by MD5 hash of the input text.

**Routing logic:**
```
USE_LLM=true
  │
  ├─ LLM_BACKEND=ollama  → OllamaClient → http://localhost:11434
  │                           Model: OLLAMA_EXTRACTION_MODEL (qwen2.5:1.5b)
  │                           Timeout: OLLAMA_TIMEOUT_SECONDS (120)
  │                           Retries: OLLAMA_MAX_RETRIES (1)
  │
  ├─ LLM_BACKEND=gemini  → Google Gemini API
  │                           Model: GEMINI_EXTRACTION_MODEL (gemini-2.0-flash)
  │                           Requires: GEMINI_API_KEY
  │
  └─ OLLAMA_FALLBACK_TO_GEMINI=true
       If Ollama times out → automatically retry with Gemini
       Requires GEMINI_API_KEY to be set
```

**LLMGrounder:** Uses the LLM to ground extracted entity names to ontology IDs when the rule-based normalizer fails. Particularly useful for novel strain names or uncommon disease names.

| File | Role |
|------|------|
| `llm_extractor.py` | Sends paper text to LLM, parses structured JSON response |
| `llm_grounder.py` | Grounds entity names to ontology IDs via LLM prompt |
| `ollama_client.py` | HTTP client for local Ollama server with timeout handling |
| `ontology_grounder.py` | Rule-based NCBI/MeSH grounding (used by entity_normalizer) |
| `schema_inducer.py` | Induces extraction schema from examples |
| `entity_registry.py` | In-memory registry of extracted entities for deduplication |
| `candidate_store.py` | `CandidateEntity` and `CandidateRelation` data models |
| `ground_cache.py` | Persistent cache for LLM grounding results |
| `_cache.py` | Thread-safe atomic JSON file cache (base class) |
| `cache/` | Cached LLM extraction results (keyed by text MD5) |
| `ontology/` | Local NCBI Taxonomy and MeSH reference files |

---

### Entity Resolution Module (`entity_resolution/`)

**Purpose:** A production-grade 7-strategy pipeline to normalize entity surface forms to canonical ontology IDs. More comprehensive than the `EntityNormalizer` in `graph/`.

Currently runs in **shadow mode** — processes every entity and logs results, but does not replace the primary normalizer yet. Results are compared and discrepancies logged until the shadow mode is validated.

**7 strategies (executed in order, first confident match wins):**

| # | Strategy | Example |
|---|----------|---------|
| 1 | Manual override | Curator-defined mappings always win |
| 2 | Exact match | "Bacteroides fragilis" → direct lookup |
| 3 | Normalized match | "bacteroides fragilis" → case-fold + strip punctuation |
| 4 | Abbreviation expansion | "IBD" → "Inflammatory Bowel Disease" → re-enter at step 2 |
| 5 | Synonym lookup | "gut flora" → "Gastrointestinal Microbiome" (MeSH synonym) |
| 6 | Fuzzy match | "Bacteroidess fragilis" → edit distance 1 → "Bacteroides fragilis" |
| 7 | Ontology traversal | Walks parent/child hierarchy to find matches |

All resolutions are written to an audit store (SQLite) and cached by registry version.

| File | Role |
|------|------|
| `resolution_pipeline.py` | Orchestrates all 7 strategies |
| `canonical_registry.py` | Stores canonical entity records |
| `synonym_index.py` | Maps synonyms to canonical IDs |
| `abbreviation_expander.py` | Expands common biomedical abbreviations |
| `fuzzy_matcher.py` | Edit-distance matching |
| `ontology_traverser.py` | Walks NCBI/MeSH hierarchy |
| `manual_override_manager.py` | Curator-defined override management |
| `ranking_function.py` | Ranks and selects among multiple candidate matches |
| `resolution_cache.py` | Caches results keyed by registry version |
| `audit_store.py` | Writes resolution records to SQLite for audit |
| `resolution_metrics.py` | Tracks resolution success rates |
| `entity_merger.py` | Merges duplicate entity records |
| `models.py` | Data models: `ResolutionResult`, `CandidateScore`, etc. |
| `db_schema.py` | SQLite schema for the resolution database |
| `utils.py` | `normalize_surface_form()` utility |

---

### Scheduler Module (`scheduler/`)

**Purpose:** Runs the collection pipeline automatically on a schedule without manual intervention.

- **Daily at 2:00 AM:** Fetches papers added in the last 24 hours from all 4 sources
- **Weekly on Sunday at 4:00 AM:** Full re-scan of all sources for updated metadata (citation counts, open access status changes)

`ChangeDetector` compares content hashes to find papers that were updated since last processed. `HashTracker` persists a set of already-processed paper hashes to avoid re-running NLP on unchanged papers.

| File | Role |
|------|------|
| `scheduler.py` | Main scheduler (APScheduler-based) — starts and stops jobs |
| `jobs.py` | Job definitions: `daily_update()`, `weekly_full_scan()` |
| `change_detector.py` | Detects updated papers by comparing content hashes |
| `hash_tracker.py` | Persists set of processed paper hashes to disk |
| `config.py` | Scheduler-specific configuration (cron expressions) |

---

### Scripts (`scripts/`)

**Purpose:** Database management utilities — backup, migration, and rollback.

| File | Role |
|------|------|
| `backup_neo4j.py` | Creates timestamped Neo4j database dumps |
| `rollback_neo4j.py` | Restores Neo4j from a backup dump |
| `migrate_to_enhanced_schema.py` | Migrates data from legacy flat schema to current enhanced schema |
| `MIGRATION_README.md` | Step-by-step migration guide |
| `ROLLBACK_GUIDE.md` | How to roll back if migration causes problems |

The enhanced schema (current) vs legacy schema comparison:

| Feature | Legacy (deprecated) | Enhanced (current) |
|---------|--------------------|--------------------|
| Relationships | Flat `HAS_TAXON` edge | 3 semantic types with rich properties |
| Provenance | None | Full: source sentence, method, confidence, timestamp |
| Evidence | Single paper | Aggregated across papers (ScientificClaim nodes) |
| Entity grounding | String names only | NCBI Taxonomy IDs, MeSH IDs |
| Direction | None | increased / decreased / no_change |
| Statistics | None | p-values, effect sizes, sample sizes |
| Querying | Manual Cypher only | 5 research query methods + REST API |


---

## 12. Data Flow End-to-End

A complete trace of one paper through the entire system:

```
────────────────────────────────────────────────────────────────────────────────
STEP 1: Collection (Layer 1)
────────────────────────────────────────────────────────────────────────────────

PubMedCollector searches:
  Query: "human microbiome" [MeSH: Gastrointestinal Microbiome, Metagenomics]
  Date: 2024-01-01 to 2026-12-31

Returns from PubMed XML:
  PaperRecord {
    doi: "10.1038/s41586-024-07999-z",
    pmid: "38765432",
    title: "Gut microbiome composition in Type 2 Diabetes...",
    mesh_terms: ["Microbiota", "Diabetes Mellitus, Type 2"]
  }

RelevanceFilter:
  Stage 1 (MeSH): "Microbiota" present AND "Diabetes" present → KEEP
  → Written to data/audit/kept.json

CollectionOrchestrator saves:
  data/processed/collected_20260602_015012.json  [21 papers]

────────────────────────────────────────────────────────────────────────────────
STEP 2: NLP Enrichment (Layer 2)
────────────────────────────────────────────────────────────────────────────────

NLPPipeline.process_one(paper):

  ArticleClassifier:
    PubMed type "Randomized Controlled Trial" found → article_type: "original_research"

  JournalClassifier:
    "Nature" → quartile: "Q1", impact_factor: 50.5

  SectionParser:
    Parses structured abstract → sections: {background, methods, results, discussion}

  NERExtractor (Tier 1 regex):
    results section → taxa: ["Bacteroides fragilis", "Faecalibacterium prausnitzii"]
    results section → diseases: ["Type 2 Diabetes", "T2D"]
    methods section → methods: ["16S rRNA sequencing", "QIIME2"]

  NERExtractor (Tier 2 BioBERT, USE_NER_MODEL=true):
    Adds: taxa: ["Roseburia intestinalis"]  ← missed by regex

  NERExtractor (Tier 3 LLM, USE_LLM=true):
    Adds: treatments: ["metformin"]
    Adds: relations: [{"subject": "Bacteroides fragilis", "predicate": "increased", "object": "T2D"}]

  DataAvailabilityExtractor:
    Found "SRP123456" in methods → accessions: ["SRP123456"], status: "open"

  StudyDesignExtractor:
    "randomized" + "120 patients" → design: "RCT", sample_size: 120

  EvidenceExtractor:
    Pre-registered trial found → evidence_score: 0.92

  QualityScorer:
    Q1 journal + RCT + n=120 + open data → quality_score: 0.91

NLPPipeline saves:
  data/processed/enriched_20260602_070900.json  [21 enriched records]

────────────────────────────────────────────────────────────────────────────────
STEP 3: Knowledge Graph (Layer 3)
────────────────────────────────────────────────────────────────────────────────

SemanticRelationshipExtractor:
  Parses results section sentence:
    "Bacteroides fragilis was significantly increased in T2D patients
     compared to healthy controls (LDA score 3.2, p=0.001)"

  Extracts:
    SemanticRelationship {
      type:       REPORTS_ASSOCIATION
      subject:    "Bacteroides fragilis"
      predicate:  "increased"
      object:     "Type 2 Diabetes"
      effect_size: 3.2
      p_value:    0.001
      comparison: "T2D patients vs healthy controls"
    }

ProvenanceEncoder:
  Attaches:
    ProvenanceMetadata {
      paper_id:           "doi:10.1038/...",
      section:            "results",
      source_sentence:    "Bacteroides fragilis was significantly increased...",
      extraction_method:  "regex_ner_v1.0",
      confidence:         0.92,  (base 0.7 + p_value boost + results boost + effect_size boost)
      validation_status:  "unvalidated"
    }

RelationshipReifier (after processing all 21 papers):
  Finds 4 other papers with the same (Bacteroides fragilis, increased, T2D) claim
  Creates ScientificClaim {
    claim_id:              "uuid-abc123",
    supporting_papers:     [5 paper IDs],
    consensus_confidence:  0.87,
    direction_consistency: 0.80,
    evidence_strength:     "strong",
    total_sample_size:     450
  }

EntityNormalizer:
  "Bacteroides fragilis" → NCBI:817  (exact match)
  "Type 2 Diabetes"      → MeSH:D003924  (exact match)

EnhancedNeo4jLoader:
  MERGE (t:Taxon {ncbi_id: "817", name: "Bacteroides fragilis"})
  MERGE (d:Disease {mesh_id: "D003924", name: "Type 2 Diabetes"})
  MERGE (p:Paper {doi: "10.1038/..."})
  CREATE (p)-[:REPORTS_ASSOCIATION {direction: "increased", p_value: 0.001,
    effect_size: 3.2, confidence: 0.92, section: "results",
    source_sentence: "...", extraction_method: "regex_ner_v1.0"}]->(t)
  SET t.disease_associations = t.disease_associations + "D003924"
  MERGE (c:ScientificClaim {claim_id: "uuid-abc123"})
  CREATE (c)-[:SUPPORTED_BY]->(p)

────────────────────────────────────────────────────────────────────────────────
STEP 4: Research Query
────────────────────────────────────────────────────────────────────────────────

HTTP POST /query/cross-study-associations
  Body: {"disease": "Type 2 Diabetes", "study_type": "RCT", "min_papers": 3}

InputValidator: All fields valid ✓
RateLimiter: Under 10 req/min ✓
QueryCache: Miss → execute

Cypher (parameterized):
  MATCH (t:Taxon)<-[:REPORTS_ASSOCIATION]-(p:Paper)
  WHERE p.disease_associations CONTAINS $disease_mesh_id
    AND p.article_type = "original_research"
    AND p.study_design = "RCT"
  WITH t, count(p) as paper_count, avg(r.confidence) as avg_confidence
  WHERE paper_count >= $min_papers
  RETURN t.name, paper_count, avg_confidence, ...

Result:
  [{taxon_name: "Bacteroides fragilis", paper_count: 5, consensus_confidence: 0.87,
    consensus_direction: "increased", direction_consistency: 0.80}, ...]

QueryCache: Store result (TTL 24h)
Response: 200 OK, JSON
```

---

## 13. Graph Schema

### Node Types

```cypher
// Research paper
(:Paper {
  doi:                String,      // Primary ID — "10.1038/s41586-024-07999-z"
  pmid:               String,      // PubMed ID
  pmcid:              String,      // PubMed Central ID
  title:              String,
  year:               Integer,
  article_type:       String,      // original_research | review | meta_analysis | ...
  study_design:       String,      // RCT | cohort | case_control | cross_sectional
  sample_size:        Integer,
  data_availability:  String,      // open | restricted | not_stated
  accession_numbers:  [String],    // ["SRP123456", "ERP789012"]
  quality_score:      Float,       // 0.0–1.0
  journal_quartile:   String       // Q1 | Q2 | Q3 | Q4 | unknown
})

// Gut bacterium, microorganism
(:Taxon {
  name:             String,        // Surface form: "Bacteroides fragilis"
  ncbi_taxonomy_id: String,        // "817"
  canonical_name:   String,        // NCBI canonical name
  grounded:         Boolean        // false if normalization failed
})

// Medical condition
(:Disease {
  name:           String,          // Surface form: "Type 2 Diabetes"
  mesh_id:        String,          // "D003924"
  canonical_name: String,          // MeSH canonical name
  grounded:       Boolean
})

// Sequencing or analysis method
(:Method {
  name:     String,                // "16S rRNA sequencing"
  category: String                 // sequencing | analysis | statistical
})

// Aggregated evidence node (created by RelationshipReifier)
(:ScientificClaim {
  claim_id:               String,  // UUID
  claim_type:             String,  // association | intervention_effect
  subject_entity:         String,
  predicate:              String,
  object_entity:          String,
  supporting_papers:      [String],
  contradicting_papers:   [String],
  consensus_confidence:   Float,
  direction_consistency:  Float,
  evidence_strength:      String,  // strong | moderate | weak | conflicting
  total_sample_size:      Integer,
  first_reported:         Integer, // year
  last_updated:           Integer  // year
})
```

### Relationship Types

```cypher
// Taxon associated with a disease (bidirectional claim)
(:Paper)-[:REPORTS_ASSOCIATION {
  // Scientific semantics
  direction:            String,    // increased | decreased | no_change
  comparison:           String,    // "T2D patients vs healthy controls"
  statistical_measure:  String,    // LDA score | fold change | relative abundance
  effect_size:          Float,
  p_value:              Float,
  adjusted_p_value:     Float,
  disease:              String,    // Denormalized MeSH ID (for query performance)

  // Provenance (on every single edge)
  section:              String,    // abstract | methods | results | discussion
  source_sentence:      String,    // Exact sentence from paper
  extraction_method:    String,    // regex_ner_v1.0 | biobert_ner | llm_extractor_v1.2
  extraction_timestamp: DateTime,
  confidence:           Float,     // 0.0–1.0
  evidence_strength:    String     // strong | moderate | weak
}]->(:Taxon)

// Intervention modifies a taxon
(:Paper)-[:REPORTS_INTERVENTION_EFFECT {
  intervention_type:  String,      // probiotic | FMT | diet | antibiotic
  effect_direction:   String,      // increased | decreased
  duration:           String,      // "8 weeks"
  dosage:             String,      // "10^9 CFU/day"
  sample_size:        Integer,

  // Provenance
  section:            String,
  source_sentence:    String,
  extraction_method:  String,
  confidence:         Float,
  evidence_strength:  String
}]->(:Taxon)

// Paper uses a sequencing method
(:Paper)-[:USES_METHODOLOGY {
  method_name:          String,
  sequencing_platform:  String,    // Illumina MiSeq | Illumina HiSeq | Nanopore | PacBio
  sample_size:          Integer,
  data_availability:    String,    // open | restricted | none

  // Provenance
  section:              String,
  extraction_method:    String,
  confidence:           Float
}]->(:Method)

// ScientificClaim supported by specific papers
(:ScientificClaim)-[:SUPPORTED_BY]->(:Paper)
(:ScientificClaim)-[:CONTRADICTED_BY]->(:Paper)
```

### Neo4j Indexes

```cypher
-- Paper indexes (for common filter combinations)
CREATE INDEX paper_year          FOR (p:Paper) ON (p.year)
CREATE INDEX paper_article_type  FOR (p:Paper) ON (p.article_type)
CREATE INDEX paper_data_avail    FOR (p:Paper) ON (p.data_availability)
CREATE INDEX paper_year_type     FOR (p:Paper) ON (p.year, p.article_type)

-- Entity indexes (for entity lookups)
CREATE INDEX taxon_name          FOR (t:Taxon) ON (t.name)
CREATE INDEX taxon_ncbi_id       FOR (t:Taxon) ON (t.ncbi_taxonomy_id)
CREATE INDEX disease_name        FOR (d:Disease) ON (d.name)
CREATE INDEX disease_mesh_id     FOR (d:Disease) ON (d.mesh_id)

-- Relationship property indexes (for query filters)
CREATE INDEX rel_confidence      FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.confidence)
CREATE INDEX rel_p_value         FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.p_value)
CREATE INDEX rel_disease         FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.disease)
```


---

## 14. Query Architecture

### How queries work internally

`ResearchQueryEngine` holds a Neo4j driver and a `QueryCache` instance. Every query method follows the same pattern:

```python
def query_cross_study_associations(self, disease, study_type, min_papers, ...):
    cache_key = self._make_cache_key("cross_study", disease, study_type, min_papers, ...)
    
    cached = self.cache.get(cache_key)
    if cached:
        return cached  # ~50ms, avoids Neo4j round-trip
    
    cypher = """
        MATCH (t:Taxon)<-[r:REPORTS_ASSOCIATION]-(p:Paper)
        WHERE r.disease = $disease_mesh_id
          AND p.article_type = $study_type
          AND r.confidence >= $confidence_threshold
        WITH t, r, p
        WHERE ($require_open_data = false OR p.data_availability = 'open')
        WITH t.name AS taxon_name,
             count(DISTINCT p) AS paper_count,
             avg(r.confidence) AS consensus_confidence,
             ...
        WHERE paper_count >= $min_papers
        RETURN taxon_name, paper_count, ...
        ORDER BY paper_count DESC, consensus_confidence DESC
    """
    
    with self.driver.session() as session:
        result = session.run(cypher, disease_mesh_id=..., ...)
        data = [dict(record) for record in result]
    
    query_result = QueryResult(results=data, ...)
    self.cache.set(cache_key, query_result, ttl_hours=24)
    return query_result
```

All queries use **parameterized Cypher** — user input is passed as parameters, never string-concatenated. This prevents Cypher injection attacks and also enables Neo4j's query plan cache.

### Query optimization strategies

1. **Parameterized queries** — prevents injection + enables plan caching
2. **Composite indexes** — `(year, article_type)` covers the most common filter combination
3. **Denormalized disease field** — `r.disease` on the relationship avoids an expensive JOIN to the Disease node
4. **Result caching** — 24h TTL, ~75% hit rate for common research queries
5. **Batch loading** — 100 nodes/edges per transaction during ingestion (avoids lock contention)
6. **Incremental processing** — only processes new papers on re-run, does not re-load existing data

### QueryCache (`graph/query_cache.py`)

In-memory Python dict with TTL expiry. Not Redis — simple and sufficient for single-process deployment.

```python
cache.set("cross_study:T2D:RCT:3:0.7:True", result, ttl_hours=24)
cache.get("cross_study:T2D:RCT:3:0.7:True")  # Returns result or None if expired
cache.invalidate()  # Clears all entries (useful after re-running Layer 3)
cache.get_stats()   # Returns {hit_rate, total_queries, cache_hits, cache_misses}
```

---

## 15. Data Directory Structure

```
data/
├── processed/
│   ├── collected_20260602_011657.json     ← Layer 1 output (first run)
│   ├── collected_20260602_015012.json     ← Layer 1 output (second run, current)
│   ├── rejected_20260602_011657.json      ← Papers rejected by relevance filter
│   ├── rejected_20260602_015012.json      ← Rejection log for current run
│   ├── enriched_20260602_070900.json      ← Layer 2 output (current)
│   └── llm_cache.json                     ← LLM extraction results cache (keyed by text MD5)
│
├── audit/
│   ├── kept.json                          ← Papers kept by relevance filter
│   ├── rejected.json                      ← Papers rejected (detailed reasons)
│   ├── review.json                        ← Borderline papers flagged for human review
│   └── llm_verified.json                  ← Papers that reached LLM stage
│
├── raw/                                   ← Cached raw API responses (not committed)
│
├── training/                              ← ML classifier training data
│
├── audit_log.db                           ← SQLite: graph modification audit log
├── conflicting_statistics_log.json        ← Papers with contradictory statistical claims
├── curator_review_queue.json              ← Entities that failed normalization (for human curation)
├── incomplete_extraction_queue.json       ← Papers where extraction was incomplete
├── validation_queue.json                  ← Relationships pending provenance validation
└── query_timeout_log.json                 ← Queries that exceeded timeout threshold
```

**File naming convention:** `collected_YYYYMMDD_HHMMSS.json` where the timestamp is UTC. `CollectionOrchestrator.load_latest()` and `NLPPipeline.load_latest()` automatically find the most recent file by sorting filenames.

---

## 16. Configuration Reference

All configuration lives in `config.py`, which reads from `.env`. Never hard-code secrets.

### Search parameters

| Config key | Default | Purpose |
|------------|---------|---------|
| `SEARCH_QUERY` | `"human microbiome"` | Core search term for all sources |
| `DATE_FROM` | `"2024/01/01"` | Earliest publication date |
| `DATE_TO` | `"2026/12/31"` | Latest publication date |
| `MAX_RESULTS_PER_SOURCE` | `500` | Max papers per source per run |
| `PUBMED_MESH_TERMS` | see config.py | MeSH terms appended to search |

### Neo4j (Enhanced — current system)

| Config key | Env var | Default | Purpose |
|------------|---------|---------|---------|
| `NEO4J_ENHANCED_URI` | `NEO4J_ENHANCED_URI` | `bolt://localhost:7687` | Connection string |
| `NEO4J_ENHANCED_USER` | `NEO4J_ENHANCED_USER` | `neo4j` | Username |
| `NEO4J_ENHANCED_PASSWORD` | `NEO4J_ENHANCED_PASSWORD` | `password` | Password |
| `NEO4J_ENHANCED_DATABASE` | `NEO4J_ENHANCED_DATABASE` | `neo4j_enhanced` | Database name |

### Pipeline settings

| Config key | Env var | Default | Purpose |
|------------|---------|---------|---------|
| `ENHANCED_PIPELINE_ENABLED` | `ENHANCED_PIPELINE_ENABLED` | `true` | Enable/disable Layer 3 |
| `ENHANCED_BATCH_SIZE` | `ENHANCED_BATCH_SIZE` | `100` | Nodes per Neo4j transaction |
| `ENHANCED_NUM_WORKERS` | `ENHANCED_NUM_WORKERS` | `8` | Parallel workers |
| `REIFICATION_ENABLED` | `REIFICATION_ENABLED` | `true` | Enable ScientificClaim creation |
| `MIN_CONFIDENCE_THRESHOLD` | `MIN_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to load |

### Query settings

| Config key | Env var | Default | Purpose |
|------------|---------|---------|---------|
| `QUERY_CACHE_ENABLED` | `QUERY_CACHE_ENABLED` | `true` | Enable result caching |
| `QUERY_CACHE_TTL_HOURS` | `QUERY_CACHE_TTL_HOURS` | `24` | Cache expiry in hours |
| `QUERY_TIMEOUT_SECONDS` | `QUERY_TIMEOUT_SECONDS` | `30` | Neo4j query timeout |

### Entity normalization

| Config key | Env var | Default | Purpose |
|------------|---------|---------|---------|
| `ENTITY_NORMALIZATION_ENABLED` | `ENTITY_NORMALIZATION_ENABLED` | `true` | Enable ontology grounding |
| `ENTITY_FUZZY_MATCH_THRESHOLD` | `ENTITY_FUZZY_MATCH_THRESHOLD` | `2` | Max edit distance for fuzzy match |
| `PROVENANCE_CONTEXT_SENTENCES` | `PROVENANCE_CONTEXT_SENTENCES` | `2` | ±N context sentences to store |
| `PROVENANCE_VALIDATION_STRICT` | `PROVENANCE_VALIDATION_STRICT` | `true` | Reject edges without provenance |

### LLM backend

| Config key | Env var | Default | Purpose |
|------------|---------|---------|---------|
| `BACKEND_CONFIG.llm_backend` | `LLM_BACKEND` | `ollama` | `ollama` or `gemini` |
| `BACKEND_CONFIG.ollama_base_url` | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `BACKEND_CONFIG.ollama_extraction_model` | `OLLAMA_EXTRACTION_MODEL` | `llama3` | Model for NER |
| `BACKEND_CONFIG.ollama_timeout_seconds` | `OLLAMA_TIMEOUT_SECONDS` | `30` | Request timeout |
| `BACKEND_CONFIG.ollama_max_retries` | `OLLAMA_MAX_RETRIES` | `3` | Retries before giving up |
| `BACKEND_CONFIG.ollama_fallback_to_gemini` | `OLLAMA_FALLBACK_TO_GEMINI` | `false` | Auto-fallback to Gemini |

### Rate limits (seconds between requests)

| Source | Default | Notes |
|--------|---------|-------|
| PubMed | 0.4s | = 2.5 req/s. With API key: 10 req/s |
| Europe PMC | 0.5s | = 2 req/s |
| Semantic Scholar | 1.0s | = 1 req/s conservative |
| bioRxiv | 0.5s | = 2 req/s |


---

## 17. Testing

The project has a comprehensive test suite using both standard unit tests and property-based tests (via the Hypothesis library).

### Test types

**Unit tests** — test individual components in isolation with mocked dependencies.

**Integration tests** — test components working together (e.g., `test_integration_graph_construction.py` tests the full Layer 3 pipeline with a real in-memory Neo4j instance).

**Property-based tests** — use Hypothesis to generate hundreds of random inputs and check that invariants always hold. These are the most important tests for catching edge cases.

Key property tests and the invariants they check:

| Test file | Invariant |
|-----------|-----------|
| `graph/test_provenance_properties.py` | Every relationship has non-null provenance; confidence is always 0.0–1.0 |
| `graph/test_provenance_traceability_properties.py` | Given a relationship in Neo4j, you can always trace back to the exact source paper and sentence |
| `graph/test_reified_claims_properties.py` | ScientificClaim consensus confidence ≥ any individual paper confidence only if all papers agree |
| `graph/test_query_threshold_properties.py` | Raising the `confidence_threshold` never returns MORE results |
| `graph/test_scalability_10k_papers.py` | Layer 3 pipeline finishes within 5 minutes for 10,000 papers |
| `graph/test_batch_processing_throughput.py` | Batch size changes do not affect output, only speed |

### Running tests

```bash
# Run all tests
pytest

# Run only fast unit tests (skip integration and scalability)
pytest -m "not integration and not slow"

# Run a specific module's tests
pytest graph/test_provenance_properties.py -v

# Run with coverage report
python run_full_coverage.py

# Run end-to-end pipeline test
python test_e2e_workflow.py

# Quick health check (no pytest needed)
python quick_system_check.py
```

### Test files at the root level

| File | Tests |
|------|-------|
| `test_e2e_workflow.py` | Full pipeline from Layer 1 through query API |
| `test_layer2.py` | All NLP modules |
| `test_main_layer3_wiring.py` | Layer 3 component wiring (SemanticExtractor → ... → Loader) |
| `test_neo4j_connection.py` | Neo4j connectivity check |
| `test_queries.py` | All 5 research queries with real Neo4j |
| `test_semantic.py` | Semantic module (LLM extractor, grounder) |
| `test_filter.py` | Relevance filter stages |
| `quick_system_check.py` | Health check for all components without running pipeline |
| `validate_migration_completeness.py` | Validates schema migration completeness |
| `validate_phase3_checkpoint.py` | Phase 3 checkpoint validation |
| `run_coverage.py` | Basic coverage report |
| `run_full_coverage.py` | Full coverage report with HTML output |
| `final_coverage_report.py` | Summary coverage report |

---

## 18. Performance

### Query performance benchmarks (on 10,000 papers, 50,000+ relationships)

| Query | Target | Typical actual |
|-------|--------|----------------|
| Simple entity lookup | < 50ms | ~35ms |
| Cross-study associations | < 2s | ~1.2s |
| Intervention evidence | < 2s | ~1.5s |
| Methodology landscape | < 2s | ~1.8s |
| Conflicting evidence | < 5s | ~3.2s |
| Cache hit (any query) | < 100ms | ~50ms |

Cache hit rate: ~75% for common queries (24-hour TTL).

### Pipeline performance

| Stage | Throughput |
|-------|-----------|
| Layer 1 collection | ~5–10 papers/second (API rate-limited) |
| Layer 2 NLP (no models) | ~10–20 papers/second |
| Layer 2 NLP (BioBERT, CPU) | ~2 papers/second |
| Layer 2 NLP (BioBERT + LLM) | ~0.5–1 paper/second (LLM is the bottleneck) |
| Layer 3 relationship extraction | ~50 papers/second |
| Layer 3 Neo4j loading | ~100 relationships/second (batch mode) |

### Tuning tips

- Development: `MAX_PER_SOURCE=5`, `USE_NER_MODEL=false`, `USE_LLM=false` → full run takes ~2 minutes
- Production: `MAX_PER_SOURCE=500`, `USE_NER_MODEL=true`, `USE_LLM=true` → full run takes 4–8 hours
- LLM timeouts: if you see frequent Ollama timeouts, either increase `OLLAMA_TIMEOUT_SECONDS` or use a smaller model
- Neo4j memory: if Layer 3 is slow, increase Neo4j heap in `neo4j.conf`: `server.memory.heap.initial_size=2g`

---

## 19. Every File Explained

### Root-level files

| File | Role |
|------|------|
| `main.py` | Entry point. Reads `RUN_LAYER` env var and dispatches to the correct layer function. |
| `models.py` | `PaperRecord` dataclass — the shared data model used by all layers. |
| `config.py` | All configuration: paths, API keys, Neo4j settings, rate limits, LLM config. Validates on import. |
| `requirements.txt` | All Python dependencies with pinned versions. |
| `docker-compose.neo4j-dual.yml` | Starts two Neo4j instances: enhanced (port 7687) and legacy (port 7688). |
| `.env` | Environment variables — API keys, passwords, flags. Never committed. |
| `RUN_ME.sh` | Convenience shell script: activates venv, runs layers 1→2→3 in sequence. |
| `start_api.sh` | Starts the FastAPI query server. |
| `README.md` | Project overview with quick start, API examples, and query examples. |
| `ARCHITECTURE.md` | This file — comprehensive system documentation. |
| `MIGRATION_GUIDE.md` | Guide for migrating from the legacy flat-relationship schema to the current enhanced schema. |
| `QUERY_EXAMPLES.md` | Detailed examples for all 5 research queries with sample outputs. |

### Root-level test and utility files

| File | Role |
|------|------|
| `test_e2e_workflow.py` | End-to-end pipeline test (all 3 layers + API) |
| `test_layer2.py` | NLP pipeline module tests |
| `test_main_layer3_wiring.py` | Layer 3 component wiring tests |
| `test_neo4j_connection.py` | Neo4j connectivity test |
| `test_queries.py` | All 5 research query tests |
| `test_semantic.py` | Semantic module (LLM extractor/grounder) tests |
| `test_filter.py` | Relevance filter stage tests |
| `quick_system_check.py` | Fast health check for all components |
| `validate_migration_completeness.py` | Validates schema migration: ≥90% entity coverage, all queries work |
| `validate_phase3_checkpoint.py` | Phase 3 milestone checkpoint validation |
| `run_coverage.py` | Basic pytest coverage run |
| `run_full_coverage.py` | Full coverage with HTML report |
| `final_coverage_report.py` | Generates coverage summary |

### `collectors/` — Layer 1

| File | Role |
|------|------|
| `orchestrator.py` | Central coordinator: runs collectors, merges, deduplicates, saves JSON. Provides `load_latest()`. |
| `pubmed_collector.py` | PubMed E-utilities (esearch/efetch) XML parser |
| `europepmc_collector.py` | Europe PMC REST API JSON fetcher |
| `semantic_scholar_collector.py` | Semantic Scholar Graph API fetcher |
| `biorxiv_collector.py` | bioRxiv REST API preprint fetcher |
| `base_collector.py` | Abstract base: rate limiting (via `time.sleep`), retry with exponential backoff, logging |
| `relevance_filter.py` | 4-stage filter pipeline: MeSH → rules → ML → LLM |
| `metadata_filter.py` | Stage 1: MeSH-based relevance check (PubMed papers only) |
| `ml_classifier.py` | Stage 3: sentence-transformers + LogisticRegression, loads/saves `config/relevance_model.pkl` |
| `llm_verifier.py` | Stage 4: calls Ollama or Gemini to make final keep/reject decision |
| `audit_logger.py` | Writes every filter decision with reason to `data/audit/*.json` |

### `nlp/` — Layer 2

| File | Role |
|------|------|
| `pipeline.py` | NLP orchestrator: loops over papers, runs all modules, saves `enriched_*.json` |
| `enriched_record.py` | `EnrichedPaperRecord` dataclass: all PaperRecord fields + NLP annotations |
| `ner.py` | 3-tier NER: Tier 1 regex dict, Tier 2 BioBERT, Tier 3 LLM |
| `article_classifier.py` | Classifies article type from pub type tags and keyword rules |
| `journal_classifier.py` | Journal quartile and impact factor lookup |
| `section_parser.py` | Splits text into named sections (abstract, methods, results, etc.) |
| `data_availability.py` | Extracts SRA/ENA/GEO accession numbers and data availability status |
| `study_design.py` | Extracts study design (RCT, cohort, etc.) and sample size |
| `evidence_extractor.py` | Extracts named datasets and evidence quality signals |
| `quality_scorer.py` | Computes composite quality score (0.0–1.0) |
| `fulltext/` | Subdirectory: orchestrates full-text PDF/XML download from PMC, Unpaywall, etc. |

### `graph/` — Layer 3

| File | Role |
|------|------|
| `enhanced_kg_pipeline.py` | Main Layer 3 orchestrator. Wires all 5 components. Reads config from env. |
| `semantic_extractor.py` | Regex-based extraction of 3 relationship types with rich properties |
| `semantic_relationships.py` | Data models: `SemanticRelationship`, `AssociationRelationship`, `InterventionRelationship`, `MethodologyRelationship` |
| `provenance.py` | `ProvenanceMetadata` model + `ProvenanceEncoder` class |
| `relationship_reifier.py` | Groups identical claims, creates `ScientificClaim` aggregation nodes |
| `reified_claims.py` | Data models: `ScientificClaim`, `EvidenceAggregation` |
| `entity_normalizer.py` | Grounds entity names to NCBI Taxonomy and MeSH |
| `enhanced_neo4j_loader.py` | Batch-loads nodes and edges to Neo4j with MERGE semantics |
| `research_query_engine.py` | 5 research query methods with caching |
| `query_cache.py` | 24-hour TTL in-memory cache keyed by query parameters |
| `predicate_registry.py` | Validates relationship predicate names |
| `extractor_registry.py` | Registry of valid extraction method identifiers |
| `enhanced_graph_builder.py` | Higher-level graph construction utilities |
| `incremental_processor.py` | Tracks processed paper DOIs to skip on re-run |
| `create_paper_indexes.py` | Creates Neo4j indexes (run once after first load) |
| `audit_log.py` | Logs all graph modifications to SQLite for rollback |
| `rollback_manager.py` | Restores graph to a previous checkpoint |
| `error_handler.py` | Centralized error handling with recovery strategies |
| `data_validator.py` | Validates data quality (required fields, ranges) before loading |
| `evidence_ranker.py` | Ranks relationships by composite evidence quality score |
| `llm_triple_extractor.py` | LLM-based triple extraction (alternative to regex semantic_extractor) |
| `kg_pipeline.py` | Legacy pipeline (deprecated, kept for rollback compatibility) |
| `neo4j_loader.py` | Legacy loader (deprecated) |
| `query_engine.py` | Legacy query engine (deprecated) |
| `schemas/` | JSON schemas for validating extracted relationship structure |
| `test_*.py` | Unit and property-based tests for every component |

### `api/` — Query Layer

| File | Role |
|------|------|
| `query_api.py` | FastAPI app: 5 POST endpoints + Neo4j startup/shutdown lifecycle |
| `input_validator.py` | Pydantic request models for all 5 endpoints + response models |
| `rate_limiter.py` | Token bucket: 10 req/min per IP |
| `query_complexity_limiter.py` | Estimates and limits query cost |
| `example_client.py` | Python usage examples for all 5 endpoints |
| `README.md` | API-specific documentation and curl examples |
| `test_*.py` | FastAPI TestClient tests for all endpoints |

### `semantic/` — LLM extraction support

| File | Role |
|------|------|
| `llm_extractor.py` | Sends paper text to LLM, parses JSON response into entities and relations |
| `llm_grounder.py` | Grounds entity names to ontology IDs via LLM prompt |
| `ollama_client.py` | HTTP client for local Ollama with timeout, retry, and fallback |
| `ontology_grounder.py` | Rule-based NCBI/MeSH grounding |
| `schema_inducer.py` | Induces extraction schema from labeled examples |
| `entity_registry.py` | In-memory entity deduplication registry |
| `candidate_store.py` | `CandidateEntity` and `CandidateRelation` data models |
| `ground_cache.py` | Persistent cache for grounding results |
| `_cache.py` | Thread-safe atomic JSON file cache base class |

### `entity_resolution/` — Advanced entity normalization (shadow mode)

All 7-strategy resolution pipeline files — see [Supporting Modules](#11-supporting-modules) for details.

### `scheduler/` — Automated updates

| File | Role |
|------|------|
| `scheduler.py` | APScheduler main entry point |
| `jobs.py` | `daily_update()` and `weekly_full_scan()` job definitions |
| `change_detector.py` | Detects updated papers by comparing content hashes |
| `hash_tracker.py` | Persists set of processed paper hashes |
| `config.py` | Scheduler-specific cron configuration |

### `scripts/` — Database management

| File | Role |
|------|------|
| `backup_neo4j.py` | Creates timestamped Neo4j database dumps |
| `rollback_neo4j.py` | Restores Neo4j from a backup |
| `migrate_to_enhanced_schema.py` | One-time migration from legacy to enhanced schema |
| `MIGRATION_README.md` | Step-by-step migration instructions |
| `ROLLBACK_GUIDE.md` | Emergency rollback procedure |

### `config/` — Static configuration files

| File | Role |
|------|------|
| `organisms.yaml` | Weighted keyword lists for Stage 2 relevance scoring (taxa names, sequencing terms) |
| `relevance_model.pkl` | Trained ML classifier for Stage 3 (generated by `train_relevance_model()`) |

---

*Last updated: June 2, 2026 — reflects Layer 1 and Layer 2 completion; Layer 3 ready to run.*
