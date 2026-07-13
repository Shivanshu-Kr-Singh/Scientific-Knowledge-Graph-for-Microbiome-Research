# Layer 2: NLP Enrichment Pipeline — Complete Documentation

> **What is Layer 2?**
> Layer 2 is the second stage of a four-layer scientific literature pipeline for the Human Microbiome Research Knowledge Graph. It takes the raw paper records collected by Layer 1 and transforms them into richly annotated scientific documents — extracting every biomedical entity mentioned, classifying the paper type, scoring journal quality, parsing section structure, identifying study design, and grounding each entity to an authoritative ontology. The output feeds directly into Layer 3 (the Knowledge Graph).

---

## Table of Contents

1. [Big Picture — Where Layer 2 Fits](#1-big-picture)
2. [How to Run Layer 2](#2-how-to-run-layer-2)
3. [Complete Data Flow](#3-complete-data-flow)
4. [Phase 0 — Batch PMCID Resolution](#4-phase-0--batch-pmcid-resolution)
5. [Phase 0.5 — PMC Full-Text Pre-Enrichment](#5-phase-05--pmc-full-text-pre-enrichment)
6. [Phase 1 — Full-Text Acquisition](#6-phase-1--full-text-acquisition)
7. [Phase 2 — NLP Processing](#7-phase-2--nlp-processing)
8. [NLP Component Deep Dives](#8-nlp-component-deep-dives)
9. [Entity Extraction: The 3-Tier System](#9-entity-extraction-the-3-tier-system)
10. [Entity Grouping](#10-entity-grouping)
11. [Entity Normalization](#11-entity-normalization)
12. [Chunked Output and Incremental Processing](#12-chunked-output-and-incremental-processing)
13. [Output Format — EnrichedPaperRecord](#13-output-format--enrichedpaperrecord)
14. [Configuration Reference](#14-configuration-reference)
15. [File Reference Map](#15-file-reference-map)

---

## 1. Big Picture

This project has four layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Data Collection                                              │
│  Fetches papers from 6 APIs, deduplicates, filters for relevance        │
│  Output: data/processed/collected_YYYYMMDD_HHMMSS.json                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  List[PaperRecord]
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — NLP Enrichment (YOU ARE HERE)                                │
│  Extracts entities, classifies articles, parses sections, normalizes    │
│  Output: data/processed/enriched_batch_NNNN.json                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  List[EnrichedPaperRecord]
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Enhanced Knowledge Graph                                     │
│  Semantic relationships → Neo4j database                                │
└─────────────────────────────────────────────────────────────────────────┘
```

**Layer 2's job:** Take a `PaperRecord` (title, abstract, authors, metadata) and return an `EnrichedPaperRecord` — the same paper but now annotated with 30+ additional scientific fields covering every entity, classification, and quality signal the knowledge graph needs.

**Layer 2's responsibility ends** when it saves `enriched_batch_NNNN.json` and returns `List[EnrichedPaperRecord]` to Layer 3.

---

## 2. How to Run Layer 2

```bash
# Standard run — processes all papers from latest Layer 1 output
RUN_LAYER=2 python main.py

# With BioBERT NER enabled (recommended, needs transformers + torch)
USE_NER_MODEL=true RUN_LAYER=2 python main.py

# With Ollama LLM extraction enabled (needs Ollama running locally)
USE_LLM=true RUN_LAYER=2 python main.py

# Full run with both NER and LLM
USE_NER_MODEL=true USE_LLM=true RUN_LAYER=2 python main.py

# Debug run — limit to first N papers only
NLP_PAPER_LIMIT=100 USE_NER_MODEL=true USE_LLM=true RUN_LAYER=2 python main.py

# Skip full-text fetching for speed (abstract-only NLP)
NLP_SKIP_FULLTEXT=true RUN_LAYER=2 python main.py
```

**Key environment variables:**

| Variable | Default | Effect |
|---|---|---|
| `USE_NER_MODEL` | `true` | Enable BioBERT Tier 2 NER |
| `USE_LLM` | `true` | Enable Ollama Tier 3 NER |
| `NLP_WORKERS` | `4` | Number of parallel worker processes for Phase 2 |
| `NLP_IO_WORKERS` | `64` | Number of threads for Phase 1 full-text fetching |
| `NLP_PAPER_LIMIT` | `0` (all) | Cap the number of papers processed (for testing) |
| `NLP_SKIP_FULLTEXT` | `false` | Skip Phase 1 entirely, use abstract only |
| `NLP_CHUNK_SIZE` | `5000` | Records per output batch file |

---

## 3. Complete Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  INPUT: collected_YYYYMMDD.json                                              │
│  List[PaperRecord] — title, abstract, authors, doi, pmid, pmcid, metadata   │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: Batch PMCID Resolution                                             │
│  For every paper with DOI but no PMCID, resolve in bulk (200 DOIs/request): │
│    1. NCBI ID Converter API  (primary, up to 200 DOIs per request)           │
│    2. Europe PMC Search API  (fallback for DOIs NCBI missed)                 │
│  Result: pmcid_cache.json populated — per-paper resolve() is now cache hit  │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0.5: PMC Full-Text Pre-Enrichment                                     │
│  For papers with a PMCID not yet in fetch_cache.json:                        │
│    → Fetch structured XML from PMC API (sections: abstract/methods/results)  │
│  Result: up to 43 papers get Tier 1 quality text before Phase 1 runs        │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Full-Text Acquisition (64 parallel threads, I/O-bound)            │
│  For each paper, FullTextOrchestrator tries in order:                        │
│    → Cache hit?  Serve instantly from fetch_cache.json                       │
│    → Has PMCID?  EuropePMC XML → NCBI PMC XML   (Tier 1, structured)        │
│    → Has pdf_url? PDF Parser  (Tier 2, pymupdf4llm + OCR fallback)          │
│    → Has url?    Web Scraper  (Tier 2, trafilatura)                          │
│    → Has DOI?    Unpaywall → OpenAIRE             (Tier 2, OA aggregators)   │
│    → Has PMID?   NCBI Abstract Fetcher            (Tier 3, structured abs)   │
│    → All fail:   Mark exhausted (retry after 90 days via TTL)               │
│  Result: each paper_dict gets _full_text field; cache flushed to disk        │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: NLP Processing (4 parallel worker processes, CPU-bound)           │
│  Each paper goes through 8 modules in sequence inside its worker process:   │
│                                                                              │
│  1. ArticleClassifier   → normalized article type + confidence              │
│  2. JournalClassifier   → quartile, impact factor, open access status       │
│  3. SectionParser       → splits text into methods/results/discussion etc.  │
│  4. NERExtractor        → 3-tier entity extraction (rules → BioBERT → LLM) │
│  5. NERExtractor.group  → groups entities into 18 typed buckets             │
│  6. DataAvailability    → SRA/GEO accession numbers, open data status       │
│  7. StudyDesign         → RCT / cohort / cross-sectional classification     │
│  8. EvidenceExtractor   → sample sizes, sequencing methods, dataset IDs     │
│  9. QualityScorer       → composite 0.0–1.0 score                           │
│ 10. FullTextStore.save  → stores full text to data/fulltext/{hash}.txt      │
│                                                                              │
│  → EntityNormalizer grounds each entity to authoritative ontology (inline)  │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  CHUNKED OUTPUT: Every CHUNK_SIZE (5000) records → enriched_batch_NNNN.json │
│  + enriched_manifest.json  (tracks all batch files)                          │
│  + enriched_hashes.txt     (incremental skip index for future runs)          │
│  + data/fulltext/{hash}.txt (full text stored separately — not inline JSON)  │
└──────────────────────────────────────────────────────────────────────────────┘
```
