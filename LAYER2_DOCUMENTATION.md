# Layer 2: NLP Enrichment Pipeline — Complete Documentation

> **What is Layer 2?**
> Layer 2 is the second stage of a four-layer scientific literature pipeline for the Human Microbiome Research Knowledge Graph. It takes raw paper records collected by Layer 1 and transforms them into richly annotated scientific documents — extracting every biomedical entity mentioned, classifying the paper type, scoring journal quality, parsing section structure, identifying study design, and grounding each entity to an authoritative ontology. The output feeds directly into Layer 3 (the Knowledge Graph).

---

## Table of Contents

1. [Big Picture — Where Layer 2 Fits](#1-big-picture)
2. [How to Run Layer 2](#2-how-to-run-layer-2)
3. [Complete Data Flow](#3-complete-data-flow)
4. [Phase 0 — Batch PMCID Resolution](#4-phase-0--batch-pmcid-resolution)
5. [Phase 0.5 — PMC Full-Text Pre-Enrichment](#5-phase-05--pmc-full-text-pre-enrichment)
6. [Phase 1 — Full-Text Acquisition](#6-phase-1--full-text-acquisition)
7. [Phase 2 — NLP Processing (The 10-Module Pipeline)](#7-phase-2--nlp-processing)
8. [Module 1: Article Classifier](#8-module-1-article-classifier)
9. [Module 2: Journal Classifier](#9-module-2-journal-classifier)
10. [Module 3: Section Parser](#10-module-3-section-parser)
11. [Module 4: NER Extractor — 3-Tier Entity Extraction](#11-module-4-ner-extractor)
12. [The 18 Entity Categories](#12-the-18-entity-categories)
13. [Entity Grouping](#13-entity-grouping)
14. [Entity Normalization (Inline Grounding)](#14-entity-normalization-inline-grounding)
15. [Module 5: Data Availability Extractor](#15-module-5-data-availability-extractor)
16. [Module 6: Study Design Extractor](#16-module-6-study-design-extractor)
17. [Module 7: Evidence Extractor](#17-module-7-evidence-extractor)
18. [Module 8: Quality Scorer](#18-module-8-quality-scorer)
19. [Parallelism Strategy — CPU vs GPU](#19-parallelism-strategy)
20. [Chunked Output and Incremental Processing](#20-chunked-output-and-incremental-processing)
21. [Output Format — EnrichedPaperRecord](#21-output-format--enrichedpaperrecord)
22. [Configuration Reference](#22-configuration-reference)
23. [File Reference Map](#23-file-reference-map)
24. [Bottlenecks and Known Limitations](#24-bottlenecks-and-known-limitations)

---


**Layer 2's job:** Take a `PaperRecord` (title, abstract, authors, metadata) and return an `EnrichedPaperRecord` — the sam30+ additional e paper but now annotated with scientific fields covering every entity, classification, and quality signal the knowledge graph needs.

**Why does Layer 2 exist?** Layer 1 can tell you a paper exists and what it's about in general terms. But Layer 3 needs to build a graph of scientific facts — *which bacteria are associated with which diseases, with what confidence, from which study type, with open data or not.* None of that is in the raw record. Layer 2 reads the actual paper text and extracts it.

**Layer 2's responsibility ends** when it saves `enriched_batch_NNNN.json` and returns `List[EnrichedPaperRecord]` to the orchestrator.

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
| `NLP_WORKERS` | `min(cpu_count, 8)` | Parallel worker processes for Phase 2 |
| `NLP_IO_WORKERS` | `64` | Parallel threads for Phase 1 full-text fetching |
| `NLP_PAPER_LIMIT` | `0` (all) | Cap papers processed (for testing) |
| `NLP_SKIP_FULLTEXT` | `false` | Skip Phase 1 entirely, use abstract only |
| `NLP_CHUNK_SIZE` | `5000` | Records per output batch file |

---

## 3. Complete Data Flow

The entire pipeline flows through four numbered phases, each with a distinct purpose:

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
│  WHY: Crossref, OpenAlex, CORE never give PMCIDs. Without this step,        │
│  those papers fall through to Tier 2/3 strategies and get PDFs or abstracts │
│  instead of structured PMC XML — 10x worse quality for NLP.                 │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0.5: PMC Full-Text Pre-Enrichment                                     │
│  For papers with a PMCID not yet in fetch_cache.json:                        │
│    → Fetch structured XML from PMC API (sections: abstract/methods/results)  │
│  Result: papers with PMCIDs get structured full text BEFORE Phase 1 runs    │
│  WHY: This avoids Phase 1 re-fetching the same PMC papers via separate       │
│  EuropePMC/NCBI threads. One batch pre-warming is faster and shares results. │
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
│  PHASE 2: NLP Processing (4–8 parallel worker processes, CPU-bound)         │
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

---

## 4. Phase 0 — Batch PMCID Resolution

**File:** `nlp/fulltext/pmcid_resolver.py`  
**Class:** `PMCIDResolver`  
**Called by:** `NLPPipeline._batch_resolve_pmcids()`

### The Problem It Solves

The six Layer 1 collectors vary widely in what identifiers they populate. PubMed and Europe PMC always provide `pmcid` when the paper is in PubMed Central. But Crossref, OpenAlex, CORE, and sometimes Semantic Scholar never populate `pmcid` — they return only the DOI.

A paper's PMCID determines whether it gets Tier 1 quality text (structured XML with separate sections) vs Tier 2 (PDF parsing or web scraping) or Tier 3 (abstract only). This is a 10x difference in NLP quality.

Without Phase 0, every Crossref or OpenAlex paper would fall through to PDF parsing or web scraping, producing noisy unstructured text. With Phase 0, most of them get PMCIDs resolved and receive clean structured XML instead.

### How It Works

```
INPUT: List of DOIs that have no PMCID in the collected record

STEP 1: NCBI ID Converter API (batch of up to 200 DOIs per request)
  POST https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/
  Body: {ids: "10.1038/...,10.1016/...", format: "json"}
  Returns: [{doi: "10.1038/...", pmcid: "PMC11234567"}, ...]

STEP 2: Europe PMC Search API (for DOIs NCBI missed)
  GET https://www.ebi.ac.uk/europepmc/webservices/rest/search
  ?query=DOI:"10.1038/..."&resultType=idlist
  Returns: {pmcid: "PMC11234567"}

RESULT: pmcid_cache.json is populated with all resolved DOI → PMCID mappings.
        Per-paper resolve() calls in Phase 1 become instant cache hits.
```

### Why Batch Instead of Per-Paper

At 10,000 papers with ~40% lacking a PMCID, that's ~4,000 NCBI API calls if done per-paper in Phase 1. Done serially before Phase 1 starts, those 4,000 DOIs go out in 20 batch requests of 200 each — ~20 API calls total, completing in seconds instead of hours.

---

## 5. Phase 0.5 — PMC Full-Text Pre-Enrichment

**File:** `collectors/pmc_enricher.py`  
**Class:** `PMCEnricher`  
**Called by:** `NLPPipeline._pmc_enrich()`

### What It Does

After Phase 0 has resolved all possible PMCIDs, Phase 0.5 pre-fetches structured full text from PubMed Central for every paper that now has a PMCID but hasn't been fetched before (not in `fetch_cache.json`).

This step is NOT redundant with Phase 1. Its purpose is to pre-warm the fetch cache before Phase 1's 64 threads start competing for the same PMC API endpoints. Instead of 64 threads all trying to fetch PMC simultaneously (and triggering rate limits), Phase 0.5 does it cleanly in sequence, marks all these papers as `status=success` in the fetch cache, and then Phase 1 threads simply see cache hits and return instantly.

### The PMC XML Structure

PMC returns structured JATS XML. The enricher extracts these sections and stores them as labeled text:

```
## Introduction
[text content]

## Methods
[text content — most valuable for NLP: sequencing methods, sample sizes]

## Results
[text content — contains statistical findings, associations]

## Discussion
[text content]

## Data Availability
[accession numbers, repository links — critical for this project]

## Funding
[grant information]
```

### Why Full Text Matters for NLP

Abstract-only NLP misses approximately 90% of extractable scientific content:

| Content type | Found in abstract? | Found in full text? |
|---|---|---|
| Sequencing method (e.g., "QIIME2 v2023.5") | Rarely | Always (Methods) |
| Sample size ("n = 245 patients") | Sometimes | Always (Methods) |
| Bacterial species with p-values | Rarely | Always (Results) |
| SRA/GEO accession numbers | Never | Always (Data Availability) |
| Bioinformatics pipeline parameters | Never | Always (Methods/Bioinformatics) |
| Effect sizes and confidence intervals | Rarely | Always (Results) |

---

## 6. Phase 1 — Full-Text Acquisition

**File:** `nlp/fulltext/fulltext_orchestrator.py`  
**Class:** `FullTextOrchestrator`  
**Execution:** `ThreadPoolExecutor` with up to 64 threads (configurable via `NLP_IO_WORKERS`)

Phase 1 is a parallelized I/O stage. Its job is simple: for every paper, get the best available full text. It uses a tiered strategy that always tries the highest-quality source first.

### Why Threads (Not Processes)?

Phase 1 spends 99% of its time waiting for HTTP responses. Python threads release the GIL during socket waits, so 64 threads can all be waiting on different remote servers simultaneously — true network-level parallelism. Processes would add process-spawn overhead and shared-memory complexity for zero benefit on I/O-bound work.

### The Fetch Cache

Every result is cached in `data/fulltext/fetch_cache.json` by `content_hash` (MD5 of title+abstract):

```json
{
  "{content_hash}": {
    "status": "success",
    "fetch_source": "europepmc",
    "fetch_tier": 1
  },
  "{content_hash_2}": {
    "status": "exhausted",
    "tried": ["europepmc", "ncbi_pmc", "unpaywall"],
    "exhausted_at": "2026-04-01T12:00:00"
  }
}
```

**On re-runs:** Successful papers return instantly from cache. Exhausted papers (all strategies failed) are skipped for 90 days (`FULLTEXT_EXHAUSTED_TTL_DAYS`), then retried — because OA embargoes lapse and Unpaywall's index grows over time. A paper with no full text today may be fully open in 3 months.

### The Shared Orchestrator Singleton

A critical performance fix: all 64 threads share one `FullTextOrchestrator` instance. Without this, each thread creates its own orchestrator, loads the cache from disk, adds entries, then gets garbage collected without saving — so no thread ever sees another thread's results. With the singleton, cache hits compound within a run: if thread 1 fetches paper A via EuropePMC, thread 2 processing a related paper (same PMCID) sees the cache and returns instantly.

```python
# Lazily created once, shared by all threads
_SHARED_ORCHESTRATOR = None
_ORCHESTRATOR_LOCK   = threading.Lock()
```

### Strategy Routing

The orchestrator uses smart routing — it doesn't blindly try all 8 strategies on every paper. It reads what identifiers the paper has and routes to the most likely source first:

```
Paper has PMCID?
  → Try EuropePMC XML   (Tier 1 — fastest, fully structured)
  → Try NCBI PMC XML    (Tier 1 — backup)

Paper has pdf_url?
  → Try PDF Parser      (Tier 2 — slower, unstructured text)

Paper has full_text_url?
  → Try Web Scraper     (Tier 2 — HTML extraction)

Paper has DOI?
  → Try Unpaywall       (Tier 2 — OA PDF/HTML finder)
  → Try OpenAIRE        (Tier 2 — European OA aggregator)

Paper has PMID?
  → Try NCBI Abstract   (Tier 3 — complete structured abstract)

Nothing worked?
  → Mark exhausted, use collector's abstract (Tier 4)
```

### Individual Fetchers

**EuropePMCFullText** (`europepmc_fulltext.py`)  
Fetches XML from `https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML`. Returns a dict with keys `abstract`, `methods`, `results`, `discussion`, `data_availability`, `funding`. Fastest and highest quality because the text is already section-labeled in JATS XML.

**NCBIPMCFetcher** (`ncbi_pmc_fetcher.py`)  
Backup to EuropePMC. Uses NCBI efetch API (`db=pmc&rettype=xml`). Same JATS XML parsing, slightly slower due to NCBI rate limits.

**PDFParser** (`pdf_parser.py`)  
Downloads the PDF from `pdf_url` and parses it using `pymupdf4llm` (produces Markdown-style structured text preserving heading hierarchy). Falls back to OCR via `pytesseract` for scanned PDFs. Returns unstructured `full_text` as a single string — the section parser in Phase 2 later attempts to split it.

**WebScraper** (`web_scraper.py`)  
Fetches the HTML page at `full_text_url` and extracts article body text using `trafilatura` (strips navigation, ads, sidebars). Handles CORS redirect and same-host redirects automatically. Slower than XML fetchers because it must download and parse full HTML pages.

**UnpaywallFetcher** (`unpaywall_fetcher.py`)  
Queries `https://api.unpaywall.org/v2/{DOI}?email=...` to find the legal OA landing page or PDF URL, then downloads it. Unpaywall covers ~50% of all papers published since 2020.

**OpenAIREFetcher** (`openaire_fetcher.py`)  
Queries `https://api.openaire.eu/search/publications?doi={DOI}` as a fallback to Unpaywall. OpenAIRE aggregates hundreds of European institutional repositories and has strong coverage for EU-funded research that Unpaywall may miss.

**NCBIAbstractFetcher** (`ncbi_abstract_fetcher.py`)  
Last resort before giving up entirely. Calls the NCBI efetch API with the paper's PMID to retrieve a clean, complete structured abstract (Background / Methods / Results / Conclusions labeled). This is higher quality than the abstract already in the PaperRecord because NCBI preserves section labels that got stripped during collection.

---

## 7. Phase 2 — NLP Processing

**File:** `nlp/pipeline.py`  
**Function:** `_process_one_worker()`  
**Execution:** `ProcessPoolExecutor` with 4–8 worker processes (CPU-bound)

Phase 2 is where all the intelligence lives. For each paper, a worker process runs 8 NLP modules in sequence. Every module is described in detail in the sections that follow.

### Why Processes (Not Threads)?

Phase 2 is CPU-bound — regex matching, ML model inference, classification logic. Python's GIL prevents true CPU parallelism with threads. Separate processes bypass the GIL entirely: 8 processes can each pin a full CPU core and run NLP logic simultaneously, achieving near-linear speedup up to the number of physical cores.

### Per-Process Module Caching

A critical performance optimization: each worker process loads NLP modules only ONCE on its first paper, then caches them for all subsequent papers it handles. Without caching, BioBERT (440MB model) would reload from disk on every paper — ~1–2 seconds of overhead per paper at 3,000+ papers.

```python
_WORKER_MODULES: dict = {}   # Process-local global cache

def _get_worker_modules(use_ner_model, use_llm) -> dict:
    global _WORKER_MODULES
    if _WORKER_MODULES:               # Already loaded? Return instantly.
        return _WORKER_MODULES
    # First paper: load everything once, cache forever for this process
    _WORKER_MODULES = {
        "article_classifier": ArticleClassifier(),
        "journal_classifier": JournalClassifier(),
        "ner": NERExtractor(use_model=use_ner_model, use_llm=use_llm),
        "section_parser": SectionParser(),
        "data_availability": DataAvailabilityExtractor(),
    }
    return _WORKER_MODULES
```

### The Processing Sequence for One Paper

For a paper that already has its full text attached from Phase 1:

```
paper_dict  (PaperRecord + _full_text from Phase 1)
    │
    ├── 1. article_classifier.classify(article_types_raw, title, abstract)
    │         → (article_type_normalized, confidence)
    │         e.g. ("meta_analysis", 0.95)
    │
    ├── 2. journal_classifier.classify(journal_name, issn)
    │         → JournalInfo(impact_factor, quartile, field, is_open_access)
    │         e.g. JournalInfo(if=13.8, q="Q1", field="Microbiology", oa=True)
    │
    ├── 3. section_parser.parse_abstract(abstract)
    │         → List[ParsedSection]  (from abstract)
    │   section_parser.parse_full_text(full_text)
    │         → List[ParsedSection]  (from full text, appended to above)
    │         e.g. [Section("methods", ...), Section("results", ...)]
    │
    ├── 4. ner.extract(title, abstract, sections, full_text)
    │         → List[NamedEntity]  (Tier 1 rules + Tier 2 BioBERT + Tier 3 LLM)
    │         e.g. [Entity("Bacteroides fragilis","taxon"), Entity("IBD","disease")]
    │
    ├── 5. ner.group_entities(entities)
    │         → dict of 18 category lists
    │         e.g. {"taxon": ["Bacteroides fragilis"], "disease": ["IBD"], ...}
    │
    ├── 6. data_availability.extract(sections, abstract)
    │         → DataAvailabilityInfo(status, accession_numbers, repositories)
    │
    ├── 7. extract_design(full_text or abstract)
    │         → {"type": "rct", "confidence": 0.90, "is_rct": True, ...}
    │
    ├── 8. extract_evidence(full_text or abstract)
    │         → {"sample_size": 245, "sequencing_methods": ["16S rRNA", ...], ...}
    │
    ├── 9. quality_score({journal_info, article_type, data_availability, ...})
    │         → 0.73  (float 0.0–1.0)
    │
    └── 10. fulltext_store.save(content_hash, full_text)
              → "data/fulltext/a3f8b2c1.txt"
              (full text stored on disk, not inside JSON — keeps output files small)
```

All results are assembled into an `EnrichedPaperRecord` and returned. Errors are caught per-paper and logged with a title snippet — one failing paper never stops the batch.

---

## 8. Module 1: Article Classifier

**File:** `nlp/article_classifier.py`  
**Class:** `ArticleClassifier`

### What It Does

Classifies each paper into a standardized article type. This matters because the knowledge graph needs to treat evidence differently depending on study design — an RCT is stronger evidence than a narrative review; a case report is anecdote, not population data.

### Output: Normalized Article Types

| Type | Meaning |
|---|---|
| `original_research` | Has primary data: patients enrolled, samples collected, results reported |
| `systematic_review` | Structured PRISMA-style search of all existing evidence |
| `meta_analysis` | Pools statistics across multiple studies (I², forest plots) |
| `narrative_review` | Broad review of existing literature, no systematic search protocol |
| `case_report` | Single patient or small case series |
| `letter` | Short correspondence |
| `commentary` | Opinion or editorial |
| `protocol` | Pre-registered study protocol — no results yet |
| `dataset` | Data paper (describes a released dataset) |
| `unknown` | Could not classify confidently |

### The 4-Tier Classification Logic

**Tier 1 — Specific PubMed tags (confidence 1.0)**

If the paper carries a specific PubMed publication type, use it directly:

```
"Randomized Controlled Trial" → original_research  (1.0)
"Meta-Analysis"               → meta_analysis       (1.0)
"Systematic Review"           → systematic_review   (1.0)
"Case Reports"                → case_report         (1.0)
```

**Critical fix:** "Journal Article" alone does NOT short-circuit to `original_research`. ~90% of all PubMed papers carry this generic tag. Previously, Tier 1 always fired on "Journal Article" and Tier 2/3 never ran — a paper tagged `["Journal Article", "Meta-Analysis"]` was incorrectly classified as `original_research`. Now "Journal Article" is treated as a generic fallback; only specific tags drive Tier 1 classification.

**Tier 2 — Title patterns (confidence 0.80–0.95)**

If Tier 1 only found a generic tag (or found nothing), scan the title:

```
"meta-analysis" in title           → meta_analysis    (0.90)
"systematic review" in title       → systematic_review (0.90)
"network meta-analysis" in title   → meta_analysis    (0.95)
"study protocol" in title          → protocol         (0.95)
"letter to the editor" in title    → letter           (0.95)
"randomized controlled trial"      → original_research (0.85)
"cohort study"                     → original_research (0.80)
```

**Tier 3 — Abstract patterns (per-pattern confidence)**

Scan the abstract for highly specific signals:

```
"forest plot"               → meta_analysis    (0.92)
"I² heterogeneity"          → meta_analysis    (0.92)
"pooled odds ratio"         → meta_analysis    (0.90)
"random-effects model"      → meta_analysis    (0.88)
"PRISMA"                    → systematic_review (0.88)
"we enrolled N patients"    → original_research (0.80)
"16S rRNA gene sequencing"  → original_research (0.78)
"registered at ClinicalTrials" → protocol      (0.88)
```

**Tier 4 — IMRAD structure detection (confidence 0.55–0.60)**

If the abstract contains 3+ IMRAD keywords (background, methods, results, conclusions, objective, findings), classify as `original_research` at 0.55. This catches unstructured clinical papers that don't match any specific pattern.

---

## 9. Module 2: Journal Classifier

**File:** `nlp/journal_classifier.py`  
**Class:** `JournalClassifier`

### What It Does

Looks up journal metadata and returns a `JournalInfo` object with impact factor, SCImago quartile (Q1–Q4), scientific field, and open access status.

### Why This Matters for the Knowledge Graph

Journal quality is a key evidence signal for Layer 3. A finding from a Q1 journal like *Cell Host & Microbe* (IF 30.3) carries more weight than the same finding from an unknown journal. The quality scorer (Module 8) uses this directly.

### Lookup Order

**Step 1 — Exact match in curated local database (~200 journals)**

The system ships with a curated database of ~200 high-volume microbiome, gastroenterology, microbiology, and general science journals with manually verified impact factors and quartiles. Exact match on lowercased journal name:

```python
JOURNAL_DB = {
    "microbiome":         {"if": 13.8, "q": "Q1", "field": "Microbiology", "oa": True},
    "gut microbes":       {"if": 12.2, "q": "Q1", "field": "Microbiology"},
    "cell host & microbe":{"if": 30.3, "q": "Q1", "field": "Microbiology"},
    "gut":                {"if": 24.5, "q": "Q1", "field": "Gastroenterology"},
    "nature":             {"if": 64.8, "q": "Q1", "field": "Multidisciplinary"},
    "jama":               {"if": 120.7,"q": "Q1", "field": "Medicine"},
    # ... ~200 total
}
```

**Step 2 — Word-set Jaccard similarity match (threshold ≥ 0.80)**

Handles abbreviations and minor naming variants. `"the isme journal"` matches `"isme journal"` (Jaccard = 0.86). Only triggered for multi-word names — prevents `"gut"` from matching `"gut microbes"`.

**Step 3 — CrossRef API lookup by ISSN or name**

For journals not in the local database, queries the CrossRef `/journals` endpoint. Returns OA status and subject field. CrossRef does NOT provide impact factors (those are proprietary to Clarivate) — only OA flag and publisher metadata.

**Step 4 — Heuristic fallbacks**

If all lookups fail, infer open access from name patterns (`"plos"`, `"bmc"`, `"frontiers"`, `"elife"`) and check for predatory publisher signals.

### Predatory Journal Detection

Known predatory publisher names are flagged: `"omics international"`, `"longdom"`, `"imedpub"`, `"scitechnol"`, `"gavin publishers"`, `"hilaris"`. Papers from flagged journals receive `is_predatory=True` in `JournalInfo`, which reduces quality score contribution to 0.

---

## 10. Module 3: Section Parser

**File:** `nlp/section_parser.py`  
**Class:** `SectionParser`

### What It Does

Splits paper text into logically labeled sections. Instead of treating the entire paper as one block of text, NER and downstream modules can work section-specifically — extracting taxa only from the Results section, or looking for accession numbers only in Data Availability.

### Why Section-Aware NLP Matters

Consider this sentence: *"Previous studies used Lactobacillus acidophilus as a probiotic intervention."*

- In a **Methods** section → this paper used L. acidophilus
- In a **Introduction** section → a different paper used L. acidophilus; this paper is citing it

Without section awareness, NER would extract L. acidophilus for both — but they mean completely different things. Section-tagged entities let Layer 3 ask: *"Which taxa appear in the Results or Methods section?"* rather than trusting mentions anywhere in the text.

### 26 Standardized Section Types

```
abstract          introduction       background
methods           study_population   bioinformatics
statistical_analysis  results        discussion
conclusion        limitations        strengths
future_directions  data_availability  supplementary
ethics            trial_registration  conflict_of_interest
funding           acknowledgements   references
glossary          other
```

### Structured Abstract Parsing

PubMed structured abstracts look like:

```
Background: Gut microbiome dysbiosis has been associated with IBD...
Methods: We recruited 200 adult IBD patients and 200 healthy controls...
Results: Significant differences in Bacteroides abundance were observed...
Conclusions: Our findings suggest a causal role for gut dysbiosis in IBD...
```

The parser detects this format and splits it on label boundaries, mapping each label to a standardized section type. Labels are sorted longest-first to prevent `"Methods"` from shadowing `"Materials and Methods"`.

### Full-Text Section Parsing

For full papers (from PDF or web scraping), the parser scans each line and checks if it matches any of 70+ header patterns:

```
"Materials and Methods"  → methods
"2.1 Study Population"   → study_population
"IV. Results"            → results
"Data Availability Statement" → data_availability
"References"             → references  (then filtered out from NER)
```

Numeric prefixes (`"1."`, `"2.1"`, `"IV."`), Roman numerals, and `"SECTION N:"` prefixes are stripped before pattern matching so they don't interfere with recognition.

### Section Priority for NER

The NER module processes sections in priority order rather than document order, focusing on the highest-value text first:

```
Priority 1 (highest): results, discussion
Priority 2: data_availability, supplementary, statistical_analysis, bioinformatics
Priority 3: methods, abstract, conclusion
Priority 4: introduction, background, other
Priority 5 (skipped): references, acknowledgements, funding, ethics, glossary
```

References and acknowledgements are excluded entirely — they mention entities from other papers, and extracting them would be false positives.

---

## 11. Module 4: NER Extractor

**File:** `nlp/ner.py`  
**Class:** `NERExtractor`

The NER (Named Entity Recognition) extractor is the most complex module in Layer 2. It uses a three-tier cascade to extract biomedical named entities from paper text, where each tier builds on the previous one.

### The 3-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 1: Rule-Based Regex (ALWAYS ON)                               │
│  18 curated entity category dictionaries                            │
│  ~300 compiled regex patterns                                       │
│  Coverage: all known microbiome taxa, diseases, methods, etc.       │
│  Speed: ~5ms per paper | Precision: 95%+ | Recall: 70-80%          │
│                                                                     │
│  Runs on: title, abstract, each section separately                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │  known entities passed to Tier 2
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 2: BioBERT NER Model (use_model=True)                         │
│  Model: d4data/biomedical-ner-all                                   │
│  ~440MB, pre-trained on CRAFT + BioNLP biomedical corpora           │
│  Coverage: novel entities not in Tier 1 dictionaries               │
│  Speed: ~500ms/paper CPU, ~50ms/paper GPU                           │
│                                                                     │
│  Runs on: sections in priority order (results first), chunked       │
│  Chunk size: 400 words with 50-word overlap                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │  known entities passed to Tier 3
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3: LLM Extraction via Ollama (use_llm=True)                   │
│  Uses the same Ollama instance as Layer 1's LLM verifier            │
│  Serialized via file lock (cross-process safe)                      │
│  Focuses on: entities Tier 1+2 missed, novel/rare entities          │
│  Speed: ~30–120s per paper (depends on Ollama model)               │
│                                                                     │
│  Runs on: top-priority sections up to 3000 chars                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Rule-Based Regex — The Dictionary System

Tier 1 uses 18 per-category dictionaries of regex patterns, compiled at module load time into one combined regex per category for performance:

```python
# WHY PRE-COMPILE:
# At 500K papers, calling re.finditer() on each of 300+ individual pattern
# strings per paper = 150M+ regex operations. Combining into one compiled
# regex per category reduces this to 18 regex operations per paper — 15-20× speedup.

def _compile_patterns(patterns: list) -> re.Pattern:
    sorted_pats = sorted(patterns, key=len, reverse=True)  # longest first
    combined    = "|".join(f"(?:{p})" for p in sorted_pats)
    return re.compile(combined, re.IGNORECASE)
```

Patterns are sorted **longest-first** so multi-word species names (`"akkermansia muciniphila"`) match before the genus alone (`"akkermansia"`). This prevents the short match from shadowing the species match.

### Abbreviation Confirmation

Short disease abbreviations (IBS, RA, MS, etc.) are frequently ambiguous. `"MS"` could be multiple sclerosis, mass spectrometry, or manuscript. The extractor confirms ambiguous abbreviations by checking a ±80 character context window for confirming words:

```python
_ABBREV_CONTEXT = {
    "ms":  ["multiple sclerosis", "sclerosis", "demyelinating", "neurological"],
    "ra":  ["arthritis", "rheumatoid", "joint", "synovial", "autoimmune"],
    "ibs": ["irritable", "bowel", "syndrome", "functional", "gastrointestinal"],
    ...
}
```

If none of the confirming words appear in the window, the abbreviation is **silently discarded** — not stored with low confidence, but completely dropped to avoid polluting the entity list.

### Tier 2: BioBERT — How It Works

BioBERT receives text in 400-word chunks (with 50-word overlap to handle entities that span chunk boundaries) and returns entity spans with confidence scores. Labels from BioBERT's output schema are mapped to our internal label vocabulary:

```
"Disease_or_Phenotypic_Feature" → disease
"Gene_or_gene_product"          → gene
"Species"                       → taxon
"Chemical"                      → metabolite
"Cell_type"                     → immune_cell
"Tissue"                        → body_site
```

WordPiece subword tokens starting with `"##"` are filtered out — BioBERT occasionally produces these as artifacts from tokenization.

### Tier 3: LLM — Known Entities Gap-Filling

The LLM receives the top-priority sections (results + discussion first, up to 3,000 characters) and a list of **entities already found by Tiers 1 and 2**. This lets the prompt focus on: *"here's what we already know about this paper — what important entities did we miss?"*

**Cross-process file lock:** Phase 2 uses 8 separate processes. The LLM (Ollama) can only run one inference at a time. Without a cross-process lock, all 8 workers call Ollama simultaneously, the requests queue up, and the ones at the back of the queue time out after 300 seconds. The solution uses `filelock.FileLock` (filesystem-based) rather than `threading.Lock` (process-local) — the file lock correctly serializes LLM calls across all 8 worker processes.

```python
_OLLAMA_NER_LOCK = FileLock(str(DATA_DIR / ".ollama_ner.lock"), timeout=900)

with _OLLAMA_NER_LOCK:   # Only one process at a time
    with _GPU_SEMAPHORE: # Only one GPU op at a time (within-process)
        candidate_entities, _ = self._llm_extractor.extract(extraction_text, ...)
```

---

## 12. The 18 Entity Categories

Layer 2 extracts entities in 18 typed categories. The original 6 categories were expanded to 18 to support the richer knowledge graph Layer 3 builds. Here is every category with examples:

### Original 6 Categories

**1. taxon** — Microbial taxa at any level (phylum → species, fungi, archaea, viruses)
> `Firmicutes`, `Lactobacillus`, `Faecalibacterium prausnitzii`, `Akkermansia muciniphila`, `Candida albicans`, `bacteriophage`
>
> The taxa dictionary is the largest — 300+ patterns covering all recognized human gut taxa, including the full 2020 Lactobacillaceae reclassification (Ligilactobacillus, Lacticaseibacillus, Limosilactobacillus, etc.) and new NCBI phylum names (Bacillota for Firmicutes, Pseudomonadota for Proteobacteria).

**2. disease** — Medical conditions, disorders, syndromes
> `IBD`, `Crohn's disease`, `type 2 diabetes`, `autism spectrum disorder`, `COVID-19`, `colorectal cancer`

**3. method** — Research techniques, bioinformatics pipelines, study designs
> `16S rRNA sequencing`, `shotgun metagenomics`, `QIIME2`, `DADA2`, `MetaPhlAn4`, `LEfSe`, `random forest`

**4. body_site** — Anatomical locations
> `gut`, `colon`, `oral cavity`, `skin`, `vagina`, `fecal sample`, `blood`, `brain`

**5. treatment** — Interventions, therapies, drugs
> `probiotics`, `FMT`, `metformin`, `rifaximin`, `Mediterranean diet`, `high-fiber diet`, `antibiotics`

**6. dataset** — Public datasets and biorepositories
> `Human Microbiome Project`, `HMP`, `PRJNA123456`, `SRA`, `curatedMetagenomicData`

### New 12 Categories (added in v2.0)

**7. metabolite** — Biochemical compounds produced by microbes
> `butyrate`, `propionate`, `acetate`, `SCFA`, `secondary bile acids`, `TMAO`, `indole`, `LPS`

**8. gene** — Genes and receptors relevant to host-microbe interactions
> `TLR4`, `NOD2`, `NF-κB`, `IL-6`, `TNF-alpha`, `FoxP3`, `MUC2`, `NLRP3`

**9. protein** — Structural proteins, cytokines, biomarker proteins
> `zonulin`, `calprotectin`, `tight junction protein`, `CRP`, `secretory IgA`, `mucin`

**10. biomarker** — Measurable indicators of disease or microbiome state
> `Shannon diversity`, `alpha diversity`, `Chao1`, `Bray-Curtis dissimilarity`, `fecal calprotectin`

**11. pathway** — Biological pathways and signaling cascades
> `NF-κB pathway`, `TLR signaling`, `butyrate metabolism`, `kynurenine pathway`, `JAK-STAT`

**12. population** — Study populations and subject groups
> `healthy adults`, `IBD patients`, `germ-free mice`, `premature infants`, `elderly cohort`

**13. dietary_component** — Foods, nutrients, dietary patterns
> `dietary fiber`, `inulin`, `polyphenols`, `resistant starch`, `Mediterranean diet`, `omega-3`

**14. immune_cell** — Immune cell types
> `Treg`, `Th17`, `dendritic cells`, `macrophages`, `ILC3`, `CD4+ T cells`, `Paneth cells`

**15. clinical_outcome** — Disease outcomes and clinical endpoints
> `remission`, `relapse`, `dysbiosis`, `intestinal permeability`, `mucosal healing`, `glycemic control`

**16. environmental_factor** — Exposures that shape the microbiome
> `antibiotic exposure`, `cesarean section`, `breastfeeding`, `birth mode`, `smoking`, `geographic location`

**17. sequencing_platform** — Instruments used for sequencing
> `Illumina MiSeq`, `NovaSeq`, `Oxford Nanopore MinION`, `PacBio Sequel II`, `Ion Torrent`

**18. omics_feature** — Bioinformatics units of analysis
> `OTU`, `ASV`, `MAG`, `KEGG pathway`, `16S amplicon`, `relative abundance`, `gene catalog`

### Unknown/Novel Entity Store

Entities discovered by BioBERT or the LLM that don't fit any of the 18 categories are stored in `other_entities` — a dict keyed by entity type string. This allows open-world entity discovery without requiring schema changes:

```json
"other_entities": {
  "receptor": ["ACE2", "GPR41", "FXR"],
  "therapeutic": ["phage therapy", "bacteriocin"],
  "biological_process": ["colonization resistance", "quorum sensing"]
}
```

---

## 13. Entity Grouping

After the 3-tier extraction produces a flat `List[NamedEntity]`, `ner.group_entities()` organizes them into the 18 category lists for easy access:

```python
def group_entities(self, entities: List[NamedEntity]) -> dict:
    groups = {label: [] for label in ENTITY_PATTERNS}  # 18 known categories
    other  = {}  # Open-world bucket for unknown types

    for ent in entities:
        if ent.label in groups:
            groups[ent.label].append(ent.text)
        else:
            # Unknown entity type — store in open-world bucket
            label_key = ent.label.lower().strip().replace(" ", "_")
            other.setdefault(label_key, []).append(ent.text)

    # Deduplicate within each group (preserving order)
    result = {label: list(dict.fromkeys(items)) for label, items in groups.items()}
    result["other_entities"] = {k: list(dict.fromkeys(v)) for k, v in other.items()}
    return result
```

The final `EnrichedPaperRecord` gets both the raw flat `entities` list (with `source_section`, `confidence`, and grounding fields per entity) and the grouped convenience lists (`taxa`, `diseases`, `methods`, etc.) for fast access.

---

## 14. Entity Normalization (Inline Grounding)

**File:** `graph/entity_normalizer.py`  
**Class:** `EntityNormalizer`  
**Called by:** `NLPPipeline.process_one()` (sequential path)

### The Problem

Different papers write the same bacterium differently:
- `"Bacteroides fragilis"`, `"B. fragilis"`, `"ATCC 25285"`, `"B.fragilis"` (no space)

Without normalization, these 4 strings become 4 separate graph nodes instead of 1. Queries for "Bacteroides fragilis" miss the abbreviation form.

### How It Works

Every extracted entity goes through a 7-step decision tree:

```
1. Empty or ≤2 characters?  → return ungrounded immediately
   (avoids noise like "dr", "co" from NER mistakes)

2. SQLite cache hit?         → return cached result instantly
   (every successful lookup is remembered — grounding_cache.db)

3. Known abbreviation?       → expand via YAML maps
   ("B. fragilis" → "Bacteroides fragilis", then continue)

4. Route to authoritative API by entity type:
   taxon     → NCBI Taxonomy  (ncbi:816 for Bacteroides fragilis)
   disease   → NCBI MeSH      (mesh:D003015 for Crohn's Disease)
   gene      → NCBI Gene
   protein   → UniProt REST
   metabolite → EMBL-EBI OLS  (ChEBI ontology)
   pathway   → OLS (GO, PW)
   method    → OLS (OBI, EFO)
   unknown   → OLS cross-search

5. OLS cross-search fallback  → if primary API failed

6. LLM fallback               → temperature=0, deterministic
   (marks result as grounded=False — no authoritative ID found)

7. All failed                 → ungrounded:{text} ID + logged to failure DB
```

### Confidence Levels

| Confidence | Source |
|---|---|
| 1.0 | Exact match in authoritative ontology (NCBI, MeSH, UniProt) |
| 0.8 | Fuzzy match in authoritative ontology |
| 0.6 | LLM suggestion (not authoritative) |
| 0.0 | All lookups failed |

### Grounding Fields on NamedEntity

Each entity carries these grounding fields after normalization:

```python
class NamedEntity(BaseModel):
    text:                 str           # Original text span: "B. fragilis"
    label:                str           # Entity type: "taxon"
    canonical_name:       Optional[str] # "Bacteroides fragilis"
    ontology_id:          Optional[str] # "ncbi:816"
    ontology_name:        Optional[str] # "NCBI Taxonomy"
    grounded:             bool          # True if authoritative ID found
    grounding_confidence: Optional[float]
    grounding_source:     Optional[str] # "ncbi" | "ols" | "uniprot" | "llm" | "none"
```

### Note on Sequential vs Parallel Path

Entity normalization runs inline in the sequential path (`process_one()`). In the parallel path (`_process_one_worker()`), the `EntityNormalizer` is not currently called per-entity — normalization in the parallel path happens later in Layer 3 (which has its own normalizer with a shared SQLite cache pre-warmed before parallel batches start). This is a known bottleneck documented in Section 24.

---

## 15. Module 5: Data Availability Extractor

**File:** `nlp/data_availability.py`  
**Class:** `DataAvailabilityExtractor`

### Why This Is a Dedicated Module

Data availability is a core requirement of this entire project. The pipeline specifically targets papers with publicly accessible sequencing data. Knowing whether data is deposited in SRA or GEO determines whether a study can be reproduced or built upon.

### What It Extracts

For each paper, this module produces a `DataAvailabilityInfo` with:
- **status**: `open` | `restricted` | `not_stated` | `accession_linked`
- **accession_numbers**: List of database IDs found (e.g. `["PRJNA123456", "SRR7654321"]`)
- **repositories**: Named repositories (e.g. `["NCBI SRA", "GEO"]`)
- **urls**: Direct links to data (e.g. `["https://zenodo.org/record/1234567"]`)
- **notes**: Context like `"Contact: j.smith@uni.ac.uk"` for restricted data

### Accession Number Patterns (by repository)

| Repository | Pattern | Example |
|---|---|---|
| NCBI SRA (Study) | `SRP\d{6,9}` | SRP123456 |
| NCBI SRA (Run) | `SRR\d{6,9}` | SRR7654321 |
| NCBI BioProject | `PRJNA\d{6,9}` | PRJNA789012 |
| NCBI GEO Series | `GSE\d{4,8}` | GSE98765 |
| ENA (Europe) | `ERP\d{6,9}` | ERP012345 |
| ArrayExpress | `E-MTAB-\d{3,6}` | E-MTAB-1234 |
| DDBJ (Japan) | `DRP\d{6,9}` | DRP001234 |
| MGnify | `MGP\d{4,7}` | MGP01234 |

### Search Strategy

```
1. Look for dedicated data availability section (section_type="data_availability")
   → Most accurate: authors wrote this specifically to describe their data

2. Search full text for "data availability" phrase + surrounding 500 chars
   → Catches unstructured papers where it's in the discussion

3. Search full text for accession number patterns directly
   → Some papers embed accession IDs in the methods without a dedicated section

4. Search abstract as last resort
```

### Status Classification Logic

```
Accession numbers found anywhere in text           → accession_linked
                                                     (highest score in quality scorer)
Direct URL to data found (Zenodo, Figshare, etc.)  → open
"data are publicly available" / "deposited at"     → open
"available upon reasonable request"                → restricted
"due to privacy" / "ethical restrictions"          → restricted
"no new data were generated" / ISNA signal         → not_stated
Data availability section exists but unclear       → not_stated
No data availability language found                → not_stated
```

---

## 16. Module 6: Study Design Extractor

**File:** `nlp/study_design.py`  
**Class:** `StudyDesignExtractor`

### What It Does

Classifies the study design of a paper and assigns a calibrated confidence score to that classification. Study design is critical for the evidence hierarchy — an RCT finding is stronger than an observational cohort finding.

### Output Structure

```python
{
    "type":          "rct",      # primary design label
    "confidence":    0.90,       # how confident we are
    "all_designs":   [           # ALL matched designs
        {"type": "rct", "confidence": 0.90},
        {"type": "clinical_trial", "confidence": 0.80}
    ],
    "is_rct":        True,       # True only for genuinely randomized trials
    "is_prospective": True       # True for prospective/longitudinal signals
}
```

### Study Design Types and Confidence Calibration

| Type | Trigger phrase | Confidence |
|---|---|---|
| `rct` | "double-blind placebo-controlled" | 0.95 |
| `rct` | "randomized controlled trial" | 0.90 |
| `rct` | "placebo-controlled randomized" | 0.85 |
| `rct` | "open-label randomized" | 0.75 |
| `meta_analysis` | "forest plot" / "I² heterogeneity" | 0.95 |
| `systematic_review` | "PRISMA" / database search + inclusion criteria | 0.90 |
| `cohort` | "prospective longitudinal cohort" | 0.85 |
| `cohort` | "cohort study" / "prospective study" | 0.75 |
| `case_control` | "case-control study" | 0.90 |
| `cross_sectional` | "cross-sectional study" | 0.90 |
| `protocol` | "study protocol" / "will be randomized" | 0.85 |
| `pilot` | "pilot study" / "feasibility study" | 0.60 |
| `mendelian_randomization` | "Mendelian randomization" | 0.95 |

### Key Fix: RCT Requires Co-occurrence

The original version triggered on `r"randomized|trial"` alone — firing at 0.9 on phrases like "trial version", "trial period", or "during a trial run". The fix requires that RCT-level evidence have co-occurring terms:

- `"randomized"` + `"controlled"` together
- OR `"double-blind"`
- OR `"placebo-controlled"`

This prevents non-clinical papers from being falsely elevated to RCT quality.

---

## 17. Module 7: Evidence Extractor

**File:** `nlp/evidence_extractor.py`  
**Function:** `extract(text)`

### What It Extracts

```python
{
    "sample_size":          245,     # Number of participants
    "sample_size_raw":      "n = 245 patients",  # Matched text for audit
    "sample_size_priority": 1,       # 1=most reliable, 7=least
    "sequencing_methods":   ["16S rRNA sequencing", "shotgun metagenomics"],
    "datasets":             ["PRJNA789012"],
    "study_confidence":     0.9,     # Based on study design signals
    "is_meta_analysis":     False
}
```

### Sample Size Extraction — Priority Hierarchy

The original version just took `max(all_number_matches)` across the entire paper — catastrophically wrong for systematic reviews where the largest number is the "database records screened" (e.g., "we identified 12,450 records"), not the participant count.

The fix uses a strict priority hierarchy, stopping as soon as a reliable match is found:

| Priority | Pattern Type | Example | Why |
|---|---|---|---|
| 1 | `n = X` notation | `(n = 245)` | Author explicitly stated participant count |
| 2 | Enrollment verb | `"enrolled 245 patients"` | Strong active-voice signal |
| 3 | Subject noun | `"245 participants"` | Direct subject count |
| 4 | Preamble phrase | `"cohort of 245"` | Indirect but clear |
| 5 | Adjective noun | `"245 obese individuals"` | Less direct |
| 6 | Non-English | `"245例"`, `"245 Patienten"` | May be imprecise |
| 7 | Proxy (sample count) | `"samples from 245"` | Least reliable |

**Exclusion set:** Numbers preceded by database-search language (`"identified 12,450 records"`, `"screened 3,200 abstracts"`) are explicitly excluded. For meta-analyses (detected by forest plot / I² patterns), priority-7 proxy patterns are disabled entirely.

### Sequencing Method Detection

```
"16s rrna" / "16s"        → 16S rRNA sequencing
"shotgun metagenomic"     → shotgun metagenomics
"whole metagenome seq"    → whole metagenome sequencing
"metatranscriptom"        → metatranscriptomics
"nanopore"                → nanopore
"pacbio"                  → PacBio
"rna-seq" / "rnaseq"      → RNA-seq
"its1/its2 sequencing"    → ITS sequencing
```

### Dataset / Accession Detection

Scans for common database accession ID patterns directly:
`PRJNA`, `PRJEB`, `SRP`, `SRR`, `SRX`, `ERR`, `ERX`, `GSE`, `GSM`, `E-MTAB-`, `SAMN`, `MGP`

---

## 18. Module 8: Quality Scorer

**File:** `nlp/quality_scorer.py`  
**Function:** `score(paper_dict) → float`

### What It Does

Produces a single composite quality score (0.0–1.0) for each paper. This score is used by Layer 3 to rank evidence — a high-quality RCT in a Q1 journal with open data scores near 1.0; a letter in an unknown journal with no data scores near 0.1.

### Scoring Components (Total max = 1.0)

**Journal Quartile (max 0.30)**

| Quartile | Score |
|---|---|
| Q1 | 0.30 |
| Q2 | 0.20 |
| Q3 | 0.10 |
| Q4 | 0.05 |
| unknown | 0.00 |

**Study Design (max 0.30)**

Uses the calibrated confidence from the Study Design Extractor, scaled linearly:
`design_score = study_conf × 0.30`

Special article-type overrides:
- `meta_analysis` → 0.22 (pools evidence, no primary data)
- `systematic_review` → 0.20
- `protocol` → 0.05 (no results yet)
- `case_report` → 0.05
- `letter` / `commentary` → 0.00

Example: double-blind RCT (confidence=0.95) → `0.95 × 0.30 = 0.285`

**Sample Size (max 0.20)**

| Sample size | Score |
|---|---|
| ≥ 1,000 | 0.20 |
| ≥ 500 | 0.15 |
| ≥ 100 | 0.10 |
| ≥ 10 | 0.05 |
| < 10 or unknown | 0.00 |

**Data Availability (max 0.15)**

Granular scoring based on how reproducible the data is:

| Status | Score | Reason |
|---|---|---|
| `accession_linked` | 0.15 | Has SRA/GEO ID — fully reproducible |
| `open` | 0.10 | Data available but no structured ID |
| `restricted` | 0.05 | Data exists, behind access control |
| `not_stated` | 0.00 | No data availability information |

**Sequencing Method Depth (max 0.05)**

| Method | Score |
|---|---|
| Shotgun / WGS / metatranscriptomics | 0.05 |
| 16S rRNA / amplicon | 0.03 |
| Any other method | 0.01 |

### Score Examples

| Paper type | Score breakdown | Total |
|---|---|---|
| Q1 RCT, n=500, open SRA data, shotgun | 0.30+0.285+0.15+0.15+0.05 | ~0.93 |
| Q2 cohort, n=100, restricted data, 16S | 0.20+0.225+0.10+0.05+0.03 | ~0.60 |
| Q3 narrative review, no data | 0.10+0.20+0+0+0 | ~0.30 |
| Unknown journal letter, no data | 0+0+0+0+0 | 0.00 |

---

## 19. Parallelism Strategy

**File:** `nlp/pipeline.py`

Layer 2 uses two different parallelism models for the two phases, chosen based on the nature of each workload.

### Phase 1 — ThreadPoolExecutor (I/O-bound)

```
64 threads × waiting for HTTP
      ↓  ↓  ↓  ↓  ↓  ↓ ... (all waiting simultaneously)
EuropePMC  NCBI  Unpaywall  WebScraper  PDF download ...
```

Why threads: Python threads release the GIL during `socket.recv()`. All 64 threads can be blocked on different network calls at the same time — true I/O concurrency. No benefit from processes here because no CPU work is being done.

### Phase 2 — ProcessPoolExecutor (CPU-bound) on CPU

```
Process 1 → NER regex + BioBERT on papers 0..N/8
Process 2 → NER regex + BioBERT on papers N/8..2N/8
Process 3 → ...
...
Process 8 → NER regex + BioBERT on papers 7N/8..N
```

Why processes: NER regex and BioBERT inference are CPU-bound. Python's GIL prevents threads from achieving real parallelism on CPU work. Separate processes each get a full CPU core and can run simultaneously.

### Phase 2 — ThreadPoolExecutor (GPU mode)

When `USE_NER_MODEL=true` AND a CUDA GPU is detected, the pipeline automatically switches Phase 2 to ThreadPoolExecutor:

```
1 shared process (main process)
  ↳ 1 shared BioBERT instance on GPU
  ↳ 8–16 threads sharing it

Thread 1 → section parsing + article classification (CPU)
           → calls shared BioBERT.forward() (GPU, GIL released during CUDA ops)
Thread 2 → same, while thread 1 is in CUDA → runs in parallel
...
```

Why not processes for GPU: 8 processes each trying to load a 440MB BioBERT onto the same GPU would immediately OOM. One process with one GPU copy, accessed by multiple threads, is the only viable approach. CUDA operations release the GIL, so threads DO achieve real GPU concurrency.

### Fallback to Sequential

If either `ProcessPoolExecutor` or `ThreadPoolExecutor` fails (e.g., insufficient memory, OS process limit, CUDA error), the pipeline falls back to sequential single-process processing. This never crashes — it just becomes slower.

---

## 20. Chunked Output and Incremental Processing

**Files:** `nlp/pipeline.py`, `data/processed/enriched_manifest.json`, `data/processed/enriched_hashes.txt`

### Why Chunked Output?

At 10,000+ papers, writing all results to a single JSON file has two problems:
1. If the run crashes at paper 9,999, you lose everything
2. A 500MB JSON file is slow to load and parse

Layer 2 writes output in chunks of 5,000 records (configurable via `NLP_CHUNK_SIZE`). Each chunk is saved to disk immediately as `enriched_batch_NNNN.json`. Progress is never lost — a crash at paper 9,999 means 9,000 records are already safely on disk.

### The Manifest File

`enriched_manifest.json` tracks all batch files across runs:

```json
{
  "batches": [
    "enriched_batch_0000.json",
    "enriched_batch_0001.json",
    "enriched_batch_0002.json"
  ],
  "total_records": 15000,
  "last_updated": "2026-07-14T03:12:02"
}
```

`load_all()` reads the manifest and merges all batches, deduplicating by `content_hash`. `load_latest()` reads only the most recent batch.

### The Hash Index — Incremental Skip

`enriched_hashes.txt` is a flat text file, one `content_hash` per line. When a new Layer 2 run starts, it loads this file and skips any paper whose `content_hash` is already in it:

```python
done_hashes = self._load_processed_hashes()
new_papers  = [p for p in papers if (p.content_hash or "") not in done_hashes]
```

This makes re-runs after a crash or after adding new papers from Layer 1 fast — only genuinely new papers are processed. At 500K papers, the hash index is ~20MB vs reading 100 JSON batch files to find all existing content_hashes.

### Full Text Storage

Full text is NOT stored inside the enriched JSON records. Storing 50,000 full papers inline would create 50GB+ JSON files. Instead:

```python
ft_path = fulltext_store.save(paper.content_hash, full_text)
# → writes to data/fulltext/a3f8b2c1d4e5f6a7.txt
# → returns "data/fulltext/a3f8b2c1d4e5f6a7.txt"
```

The `EnrichedPaperRecord` stores only `fulltext_path` — a pointer to the file. `full_text` is set to `None` before serialization to JSON.

---

## 21. Output Format — EnrichedPaperRecord

**File:** `nlp/enriched_record.py`  
**Class:** `EnrichedPaperRecord`

`EnrichedPaperRecord` extends `PaperRecord` (which has all the original Layer 1 fields) and adds 30+ new fields produced by the 8 NLP modules.

### Full Schema

```python
class EnrichedPaperRecord(PaperRecord):
    # ── Inherited from PaperRecord (Layer 1 fields) ───────────────────────
    doi:              Optional[str]
    pmid:             Optional[str]
    pmcid:            Optional[str]
    title:            str
    abstract:         Optional[str]
    authors:          List[str]
    keywords:         List[str]
    journal:          Optional[str]
    publication_date: Optional[str]
    publication_year: Optional[int]
    article_types:    List[str]       # raw tags from source
    is_open_access:   bool
    citation_count:   Optional[int]
    mesh_terms:       List[str]
    content_hash:     Optional[str]
    is_preprint:      bool

    # ── Module 1: Article classifier output ──────────────────────────────
    article_type_normalized: str           # "original_research", "meta_analysis" etc.
    article_type_confidence: Optional[float]  # 0.0–1.0

    # ── Module 2: Journal classifier output ──────────────────────────────
    journal_info: Optional[JournalInfo]    # {impact_factor, quartile, field, is_oa}

    # ── Module 3: Section parser output ──────────────────────────────────
    sections: List[ParsedSection]         # [{section_type, header, content}, ...]

    # ── Module 4: NER output — flat list ─────────────────────────────────
    entities: List[NamedEntity]           # All entities with confidence + grounding

    # ── Module 4: NER output — grouped convenience lists ─────────────────
    taxa:                  List[str]   # Microbial taxa
    diseases:              List[str]   # Diseases / conditions
    methods:               List[str]   # Research methods
    body_sites:            List[str]   # Anatomical locations
    treatments:            List[str]   # Interventions / drugs
    datasets:              List[str]   # Dataset accession IDs
    metabolites:           List[str]   # Metabolites / biochemicals
    genes:                 List[str]   # Genes / receptors
    proteins:              List[str]   # Proteins / cytokines
    biomarkers:            List[str]   # Measurable biomarkers
    pathways:              List[str]   # Biological pathways
    populations:           List[str]   # Study populations
    dietary_components:    List[str]   # Foods / nutrients
    immune_cells:          List[str]   # Immune cell types
    clinical_outcomes:     List[str]   # Clinical endpoints
    environmental_factors: List[str]   # Environmental exposures
    sequencing_platforms:  List[str]   # Instruments (MiSeq, etc.)
    omics_features:        List[str]   # OTU, ASV, MAG, etc.
    other_entities:        Dict[str, List[str]]  # Open-world bucket

    # ── Module 5: Data availability output ───────────────────────────────
    data_availability: Optional[DataAvailabilityInfo]
    # {status, accession_numbers, repositories, urls, notes}

    # ── Module 6: Study design output ────────────────────────────────────
    study_design: Optional[dict]
    # {type, confidence, all_designs, is_rct, is_prospective}

    # ── Module 7: Evidence extractor output ──────────────────────────────
    evidence_score:   float        # = sample_size (0 if not found)
    datasets:         List[str]    # Accession IDs from evidence extractor

    # ── Module 8: Quality scorer output ──────────────────────────────────
    quality_score:    float        # Composite 0.0–1.0

    # ── Full-text retrieval metadata ──────────────────────────────────────
    fetch_source:     Optional[str]   # "europepmc" | "ncbi_pmc" | "pdf" | ...
    fetch_status:     Optional[str]   # "success" | "cached" | "exhausted"
    fulltext_path:    str             # "data/fulltext/a3f8b2c1.txt" or ""
    full_text:        None            # Always None — stored in fulltext_path

    # ── Pipeline metadata ─────────────────────────────────────────────────
    nlp_processed_at: Optional[str]   # ISO timestamp
    nlp_version:      str             # "2.0"
```

### Example Output Record (truncated)

```json
{
  "doi": "10.1038/s41586-024-07999-z",
  "pmid": "38765432",
  "pmcid": "PMC11234567",
  "title": "Gut microbiome composition predicts IBD remission following FMT",

  "article_type_normalized": "original_research",
  "article_type_confidence": 0.90,

  "journal_info": {
    "name": "Nature",
    "impact_factor": 64.8,
    "quartile": "Q1",
    "field": "Multidisciplinary",
    "is_open_access": false
  },

  "taxa": ["Bacteroides fragilis", "Faecalibacterium prausnitzii", "gut microbiota"],
  "diseases": ["inflammatory bowel disease", "IBD", "ulcerative colitis"],
  "methods": ["16S rRNA sequencing", "QIIME2", "shotgun metagenomics"],
  "treatments": ["fecal microbiota transplantation", "FMT"],
  "metabolites": ["butyrate", "SCFA"],
  "biomarkers": ["Shannon diversity", "alpha diversity"],
  "populations": ["IBD patients", "healthy controls"],
  "clinical_outcomes": ["remission", "dysbiosis"],

  "data_availability": {
    "status": "accession_linked",
    "accession_numbers": ["PRJNA789012"],
    "repositories": ["NCBI SRA"]
  },

  "study_design": {
    "type": "rct",
    "confidence": 0.90,
    "is_rct": true,
    "is_prospective": true
  },

  "evidence_score": 245,
  "quality_score": 0.875,
  "fetch_source": "europepmc",
  "fulltext_path": "data/fulltext/a3f8b2c1d4e5f6a7.txt",
  "nlp_processed_at": "2026-07-14T03:12:02",
  "nlp_version": "2.0"
}
```

---

## 22. Configuration Reference

| Variable | Source | Default | Effect |
|---|---|---|---|
| `USE_NER_MODEL` | env | `true` | Enable BioBERT Tier 2 NER model |
| `USE_LLM` | env | `true` | Enable Ollama Tier 3 LLM extraction |
| `NLP_WORKERS` | env | `min(cpu_count, 8)` | CPU worker processes for Phase 2 |
| `NLP_IO_WORKERS` | env | `64` | I/O threads for Phase 1 full-text fetch |
| `NLP_PAPER_LIMIT` | env | `0` (all) | Cap number of papers (testing) |
| `NLP_SKIP_FULLTEXT` | env | `false` | Skip Phase 1 — abstract-only mode |
| `NLP_CHUNK_SIZE` | env | `5000` | Records per output JSON batch file |
| `FULLTEXT_EXHAUSTED_TTL_DAYS` | config | `90` | Days before retrying exhausted papers |
| `NCBI_EMAIL` | .env | (required) | NCBI polite pool — required for all NCBI calls |
| `NCBI_API_KEY` | .env | "" | Upgrades NCBI rate limit to 10 req/sec |
| `OLLAMA_BASE_URL` | .env | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VERIFIER_MODEL` | .env | `"llama3"` | LLM model for NER Tier 3 |

---

## 23. File Reference Map

```
Layer 2 files and their roles:

nlp/
  pipeline.py                    ← Main orchestrator: 2-phase parallel processing
  enriched_record.py             ← EnrichedPaperRecord schema (Layer 2 output contract)
  article_classifier.py          ← Module 1: 4-tier article type classification
  journal_classifier.py          ← Module 2: impact factor + quartile + OA status
  section_parser.py              ← Module 3: structured section detection + parsing
  ner.py                         ← Module 4: 3-tier NER (rules → BioBERT → LLM)
                                              + 18-category entity dictionaries
  data_availability.py           ← Module 5: accession numbers, repos, status
  study_design.py                ← Module 6: RCT/cohort/cross-sectional classifier
  evidence_extractor.py          ← Module 7: sample size, sequencing methods, datasets
  quality_scorer.py              ← Module 8: composite 0.0–1.0 quality score

nlp/fulltext/
  fulltext_orchestrator.py       ← Smart fetch routing + persistent fetch cache
  europepmc_fulltext.py          ← Tier 1: EuropePMC XML fetcher
  ncbi_pmc_fetcher.py            ← Tier 1: NCBI PMC efetch XML fetcher
  pdf_parser.py                  ← Tier 2: pymupdf4llm PDF → markdown
  web_scraper.py                 ← Tier 2: trafilatura HTML → article text
  unpaywall_fetcher.py           ← Tier 2: Unpaywall OA PDF/HTML finder
  openaire_fetcher.py            ← Tier 2: OpenAIRE European OA aggregator
  ncbi_abstract_fetcher.py       ← Tier 3: NCBI PubMed structured abstract
  pmcid_resolver.py              ← Phase 0: DOI → PMCID batch resolution
  fulltext_store.py              ← data/fulltext/{hash}.txt — separate full-text storage
  domain_throttle.py             ← Per-domain rate limiting for Tier 2 strategies

semantic/
  llm_extractor.py               ← Tier 3 NER: LLM entity extraction prompt
  ollama_client.py               ← Ollama HTTP client with retry

collectors/
  pmc_enricher.py                ← Phase 0.5: structured PMC full-text pre-enrichment

graph/
  entity_normalizer.py           ← Inline grounding: entity → ontology ID

config/
  (no Layer 2-specific config files — all settings in config.py and .env)

data/ (auto-created)
  processed/
    enriched_batch_NNNN.json     ← Chunked Layer 2 output (input to Layer 3)
    enriched_manifest.json       ← Tracks all batch files + record counts
    enriched_hashes.txt          ← Flat hash index for incremental skip
  fulltext/
    fetch_cache.json             ← Full-text fetch results cache
    {content_hash}.txt           ← Stored full texts (one file per paper)
```

---

## 24. Bottlenecks and Known Limitations

This section catalogs every significant bottleneck in Layer 2 — both confirmed performance problems and architectural limitations — with the root cause, current workaround, and ideal fix.

---

### Bottleneck 1: Tier 3 LLM Extraction Serialization

**Severity: Critical**  
**Location:** `nlp/ner.py` — `_llm_extract()`

**What happens:**  
Phase 2 uses 8 worker processes running NLP in parallel. Each process calls Ollama for Tier 3 NER. Ollama internally runs one inference at a time. Without cross-process serialization, all 8 processes call Ollama simultaneously, requests queue up inside Ollama, and the requests at the back of the queue time out after 300 seconds.

**Current fix:**  
A `filelock.FileLock` (filesystem-based, cross-process safe) serializes all Ollama calls:
```python
_OLLAMA_NER_LOCK = FileLock(str(DATA_DIR / ".ollama_ner.lock"), timeout=900)
with _OLLAMA_NER_LOCK:
    candidate_entities, _ = self._llm_extractor.extract(text, ...)
```

**Remaining impact:**  
All 8 workers must wait for each other's LLM calls. If each paper takes 60s of LLM time, and you have 1,000 papers, that's 1,000 × 60s = 16.7 hours of LLM time even with 8 workers. Effectively, Tier 3 NER reduces Phase 2 parallelism from 8× to 1×.

**Ideal fix:**  
Run a single dedicated LLM worker process that other workers submit to via a queue (multiprocessing.Queue or Redis). One producer/consumer per Ollama instance. Use a larger, faster model (e.g., `qwen2.5:7b` vs `qwen2.5:0.5b`) to reduce per-paper latency.

---

### Bottleneck 2: BioBERT Model Reload (Fixed in v2.0, documented for context)

**Severity: Resolved**  
**Location:** `nlp/pipeline.py` — `_get_worker_modules()`

**What was happening:**  
In v1.0, every paper created a new `NERExtractor()`, which triggered BioBERT to reload 440MB from disk (~1–2s). At 3,000 papers with 8 workers, this was 3,000 × 1.5s = 1.25 hours of pure model loading overhead before any NLP work happened.

**Fix applied:**  
Per-process module caching via `_WORKER_MODULES` global dict. BioBERT loads once per process on its first paper, then all subsequent papers in that process reuse the loaded model.

---

### Bottleneck 3: Phase 1 Fetching Duplicate PMC Papers

**Severity: Medium**  
**Location:** `nlp/pipeline.py` — `_fetch_fulltext_worker()`

**What happens:**  
There's a residual bug in `_fetch_fulltext_worker` — the function calls `_get_shared_orchestrator().fetch()` once, stores the result in `paper_dict["_full"]`, and then calls `FullTextOrchestrator().fetch()` again (creating a new unshared instance). The second call fetches the same paper again, wastes API calls, and overwrites the first result:

```python
# BUG: two separate fetch calls in _fetch_fulltext_worker
full = _get_shared_orchestrator().fetch(paper) or {}   # ← shared, cached
...
try:
    full = FullTextOrchestrator().fetch(paper) or {}    # ← NEW instance, cache miss!
except Exception:
    full = {}
```

**Current impact:**  
Every paper's full text is fetched twice during Phase 1. Second call is often a cache hit (same `content_hash`) so it's fast, but it still creates unnecessary FullTextOrchestrator instances and does redundant cache lookups.

**Fix required:**  
Remove the second `FullTextOrchestrator()` instantiation and the second `fetch()` call. Use only the shared orchestrator's result.

---

### Bottleneck 4: Entity Normalization Not Inline in Parallel Path

**Severity: Medium**  
**Location:** `nlp/pipeline.py` — `_process_one_worker()`

**What happens:**  
The sequential `process_one()` path calls `EntityNormalizer.normalize()` for every entity inline, attaching ontology IDs to each `NamedEntity` before returning. The parallel path (`_process_one_worker()`) does NOT call the EntityNormalizer — entities go into the `EnrichedPaperRecord` with `grounded=False` and no ontology IDs.

**Why it was designed this way:**  
`EntityNormalizer` uses a SQLite database (`grounding_cache.db`) and makes HTTP calls to NCBI/OLS APIs. SQLite is not process-safe by default for concurrent writes. The parallel worker processes would conflict on cache writes.

**Current impact:**  
Layer 3 has its own EntityNormalizer and pre-warms its cache before parallel work starts. This means normalization happens in Layer 3 instead of Layer 2 — the enriched records sent from Layer 2 to Layer 3 are missing ontology IDs. If you inspect `EnrichedPaperRecord.entities` after parallel Layer 2 processing, `grounded=False` and `ontology_id=None` for all entities.

**Ideal fix:**  
Use `WAL` (Write-Ahead Log) mode for SQLite to allow safe concurrent reads + one writer, or normalize using a pre-built read-only lookup dict in worker processes (no writes during NLP, flush to disk after all workers complete).

---

### Bottleneck 5: Full-Text Coverage Rate

**Severity: Medium (architectural, not a bug)**  
**Location:** `nlp/fulltext/fulltext_orchestrator.py`

**What happens:**  
Not all papers have full text available. The typical breakdown:
- ~25–30% of papers have PMCIDs → Tier 1 (XML, best quality)
- ~20–25% can be found via Unpaywall/OpenAIRE → Tier 2 (PDF/HTML)
- ~15–20% get NCBI structured abstracts → Tier 3
- ~25–35% fall through to abstract-only → Tier 4 (lowest NLP quality)

**Impact:**  
Papers without full text have NER run only on title + abstract. This means:
- No accession numbers (never in abstract)
- No sample size with high confidence (often only in methods)
- Fewer taxa/disease entities (methods and results have the most)
- No sequencing platform details

**Workarounds:**  
- Phase 0 PMCID resolution upgrades Crossref/OpenAlex papers to Tier 1 (major improvement)
- Phase 0.5 pre-enrichment ensures all PMC papers are fetched cleanly
- Unpaywall covers ~50% of papers published since 2020

**No complete fix** — some papers are behind paywalls that no legal OA aggregator covers. The 90-day TTL retry means papers that become OA (embargo lifting) will eventually get full text.

---

### Bottleneck 6: PDF Parsing Quality

**Severity: Medium**  
**Location:** `nlp/fulltext/pdf_parser.py`

**What happens:**  
PDFs from Tier 2 strategies are significantly harder to process than structured XML:
- Multi-column PDFs produce garbled text when linearized (words from column A and column B interleaved)
- Figures and tables produce garbage text
- Footnotes mix into body text
- Running headers/footers repeat on every page
- Section boundaries are invisible — the section parser must infer them from formatting cues

**Impact:**  
NER on PDF text has more false positives than on PMC XML. Section detection is less reliable. Sample size and accession number extraction are less accurate.

**Current mitigation:**  
`pymupdf4llm` produces Markdown-formatted output that partially preserves heading structure. The section parser can detect headings in this Markdown. OCR fallback handles scanned PDFs.

**No complete fix** — PDF parsing is an inherently lossy process. The information hierarchy (headers, paragraphs, figures, tables) encoded visually in a PDF is difficult to recover programmatically.

---

### Bottleneck 7: Ollama Model Size for NER Quality

**Severity: Medium**  
**Location:** `semantic/llm_extractor.py`

**What happens:**  
The configured Tier 3 model is `qwen2.5:0.5b` (500M parameters). At this size, the model:
- Misses subtle entities (e.g., rare bacterial species, novel metabolites)
- Occasionally hallucinates entities not in the text
- Produces non-JSON output requiring repair logic

**Impact:**  
Tier 3 NER extracts 2–5 additional entities per paper on average (vs 20–40 from Tier 1 rules). At `qwen2.5:0.5b`, many of those are low quality. A larger model would dramatically improve recall and precision.

**Ideal fix:**  
Use `qwen2.5:7b` or `llama3.1:8b` — ~10× more parameters → much higher NER quality. Requires either a GPU with 8GB+ VRAM or accepting ~3–5 min per paper on CPU.

---

### Bottleneck 8: Journal Database Coverage

**Severity: Low**  
**Location:** `nlp/journal_classifier.py`

**What happens:**  
The curated `JOURNAL_DB` contains ~200 journals. The microbiome literature spans thousands of journals. Papers from journals not in the database (and not found by CrossRef) get `quartile="unknown"`, contributing 0 to the quality score.

**Current fix:**  
CrossRef API lookup catches most standard journals. But CrossRef doesn't provide impact factors or quartile rankings (proprietary Clarivate data), so even CrossRef-found journals get `quartile="unknown"`.

**Impact:**  
Papers in smaller specialty journals get penalized in quality score simply because we don't have their quartile. This biases quality scores toward high-profile journals in the curated database.

**Ideal fix:**  
Integrate the SCImago Journal Rank public dataset (freely available) to supplement the curated database with quartile data for thousands of additional journals.

---

### Bottleneck 9: Section Parser Accuracy on Non-Standard Papers

**Severity: Low**  
**Location:** `nlp/section_parser.py`

**What happens:**  
The section parser relies on header pattern matching. Papers with non-standard section naming — conference papers, preprints, some MDPI journals — use unusual headers (`"Experimental Section"`, `"Main Text"`, `"Experimental Work"`) that don't match the 70+ patterns. These sections fall through as `section_type="other"`.

**Impact:**  
NER on these sections still runs, but the results are tagged with `source_section="other"` rather than `"methods"` or `"results"`. Layer 3 queries that filter by section type miss these entities.

**No immediate fix needed** — the section type tagging is a convenience feature for Layer 3, not a correctness requirement. All entities are still extracted; they just have less precise provenance.

---

*This document covers Layer 2 of the Human Microbiome Research Knowledge Graph pipeline.*  
*For Layer 1 (Data Collection), see LAYER1_DOCUMENTATION.md.*  
*For Layer 3 (Knowledge Graph), see LAYER3_DOCUMENTATION.md.*
