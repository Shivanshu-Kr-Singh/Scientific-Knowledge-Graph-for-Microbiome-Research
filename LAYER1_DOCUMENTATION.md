# Layer 1: Data Collection — Complete Documentation

> **What is Layer 1?**
> Layer 1 is the first stage of a four-layer scientific literature pipeline for the Human Microbiome Research Knowledge Graph. Its job is simple to state but complex in execution: **collect research papers from multiple academic APIs, filter them for relevance, and hand a clean, deduplicated list to Layer 2 (NLP Enrichment).**

---

## Table of Contents

1. [Big Picture — Where Layer 1 Fits](#1-big-picture)
2. [How to Run Layer 1](#2-how-to-run-layer-1)
3. [Project Configuration (`config.py`)](#3-project-configuration)
4. [The Data Model (`models.py` — PaperRecord)](#4-the-data-model)
5. [BaseCollector — Shared Infrastructure](#5-basecollector)
6. [The Six Data Source Collectors](#6-the-six-data-source-collectors)
   - [PubMed Collector](#61-pubmed-collector)
   - [Europe PMC Collector](#62-europe-pmc-collector)
   - [Semantic Scholar Collector](#63-semantic-scholar-collector)
   - [OpenAlex Collector](#64-openalex-collector)
   - [Crossref Collector](#65-crossref-collector)
   - [CORE Collector](#66-core-collector)
7. [The Orchestrator — Coordinating All Collectors](#7-the-orchestrator)
8. [The 4-Stage Relevance Filter Pipeline](#8-the-4-stage-relevance-filter-pipeline)
   - [Stage 1 — MeSH Metadata Filter](#81-stage-1--mesh-metadata-filter)
   - [Metagenomics Gate](#82-metagenomics-gate)
   - [Stage 2 — Weighted Keyword Scorer](#83-stage-2--weighted-keyword-scorer)
   - [Stage 3 — ML Classifier](#84-stage-3--ml-classifier)
   - [Stage 3.5 — Embedding Similarity Filter](#85-stage-35--embedding-similarity-filter)
   - [Stage 4 — LLM Verifier](#86-stage-4--llm-verifier)
9. [Audit & Observability](#9-audit--observability)
10. [PMC Full-Text Enricher](#10-pmc-full-text-enricher)
11. [Complete Data Flow Diagram](#11-complete-data-flow-diagram)
12. [Output Format](#12-output-format)
13. [Cross-Run Incrementality (Cursors)](#13-cross-run-incrementality-cursors)
14. [Configuration Reference Table](#14-configuration-reference-table)
15. [File Reference Map](#15-file-reference-map)

---

## 1. Big Picture

This project has four layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Data Collection (YOU ARE HERE)                               │
│  Fetches papers from 6 APIs, deduplicates, filters for relevance        │
│  Output: data/processed/collected_YYYYMMDD_HHMMSS.json                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  List[PaperRecord]
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — NLP Enrichment                                               │
│  Extracts entities, classifies articles, parses sections                │
│  Output: data/processed/enriched_YYYYMMDD_HHMMSS.json                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Enhanced Knowledge Graph                                     │
│  Semantic relationships → Neo4j database                                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  QUERY LAYER — Research Query Engine                                    │
│  REST API + Query Cache → Answers 5 core research questions             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Layer 1's responsibility ends** when it returns `List[PaperRecord]` — a clean, deduplicated,
relevance-filtered list of papers ready for NLP processing.

---

## 2. How to Run Layer 1

```bash
# Basic run (fetches up to MAX_RESULTS_PER_SOURCE papers per source)
RUN_LAYER=1 python main.py

# Development run (smaller batch — faster for testing)
MAX_PER_SOURCE=20 RUN_LAYER=1 python main.py

# Production run (full 500 per source)
MAX_PER_SOURCE=500 RUN_LAYER=1 python main.py
```

**What happens when you run it:**
1. All 6 collectors start in parallel (each hits a different API host)
2. Results are merged and deduplicated by DOI → PMID → title
3. The relevance filter pipeline runs on every paper (4 stages)
4. The final filtered list is saved to `data/processed/collected_YYYYMMDD_HHMMSS.json`
5. Cursors are saved so the next run picks up from where this one stopped

**Output location:** `data/processed/collected_YYYYMMDD_HHMMSS.json`

---

## 3. Project Configuration

**File:** `config.py`

This is the single source of truth for all settings. Every other file imports from here.
If you ever need to change an API key, date range, or threshold — you change it in **one place**.

### What `config.py` controls

```python
# What we're searching for
SEARCH_QUERY = "human microbiome"
DATE_FROM    = "2024/01/01"
DATE_TO      = "2026/12/31"

# How many papers per source per run
MAX_RESULTS_PER_SOURCE = 500

# PubMed MeSH terms for the query
PUBMED_MESH_TERMS = [
    "Microbiota",
    "Gastrointestinal Microbiome",
    "RNA, Ribosomal, 16S",
    "Metagenomics",
    "Bacteria",
]
```

### API Credentials (from `.env` file — never hardcoded)

```bash
# .env file (never commit this to git)
NCBI_EMAIL=you@example.com
NCBI_API_KEY=abc123...           # PubMed: 3 req/sec without, 10 req/sec with
SEMANTIC_SCHOLAR_API_KEY=xyz...  # Semantic Scholar rate upgrade

# Neo4j (for Layer 3)
NEO4J_ENHANCED_URI=bolt://localhost:7687
NEO4J_ENHANCED_USER=neo4j
NEO4J_ENHANCED_PASSWORD=password
```

### Directory Structure (auto-created)

```
data/
  raw/          # Cached raw API responses (per source)
    pubmed/
    europepmc/
    semantic_scholar/
    openalex/
    crossref/
    core/
  processed/    # NLP-processed records + cursors + LLM cache
  embeddings/   # Embedding store for Stage 3.5 + semantic LLM cache
logs/           # Rotating log files (10 MB rotation, 30-day retention)
```

---

## 4. The Data Model

**File:** `models.py`

Every collector, regardless of source, converts its raw API response into a single shared
object: `PaperRecord`. This is the **contract between all layers** of the system.

> Think of `PaperRecord` like a standardized form that every API must fill out. PubMed speaks
> XML, Semantic Scholar speaks JSON, OpenAlex speaks a different JSON — but after parsing,
> they all produce the same form. Layer 2 never needs to know which source a paper came from.

```python
class PaperRecord(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────
    doi:      Optional[str]   # e.g. "10.1038/s41586-024-07999-z"
    pmid:     Optional[str]   # e.g. "38765432"  (PubMed ID)
    pmcid:    Optional[str]   # e.g. "PMC11234567" (PubMed Central ID)
    arxiv_id: Optional[str]   # e.g. "2024.12345"
    source:   str             # "pubmed" | "europepmc" | "semantic_scholar" | ...

    # ── Core Content ──────────────────────────────────────────────────────
    title:    str
    abstract: Optional[str]
    authors:  List[str]
    keywords: List[str]

    # ── Publication Info ──────────────────────────────────────────────────
    journal:          Optional[str]
    journal_abbrev:   Optional[str]
    issn:             Optional[str]
    publication_date: Optional[str]   # ISO: "2024-03-15"
    publication_year: Optional[int]
    volume:           Optional[str]
    issue:            Optional[str]
    pages:            Optional[str]

    # ── Article Classification ─────────────────────────────────────────────
    article_types: List[str]   # e.g. ["Journal Article", "Review"]

    # ── Access & Full Text ─────────────────────────────────────────────────
    is_open_access: bool
    full_text_url:  Optional[str]
    pdf_url:        Optional[str]
    full_text:      Optional[str]   # Added by PMCEnricher

    # ── Citations ─────────────────────────────────────────────────────────
    citation_count:   Optional[int]
    reference_count:  Optional[int]

    # ── PubMed-specific ────────────────────────────────────────────────────
    mesh_terms: List[str]   # Medical Subject Headings — key for Stage 1 filter

    # ── Pipeline Metadata ─────────────────────────────────────────────────
    content_hash: Optional[str]   # MD5 of title+abstract (detects corrections)
    fetched_at:   Optional[str]   # ISO timestamp of when we fetched this
    is_preprint:  bool
```

### Deduplication Key

When the same paper appears in multiple sources, the system deduplicates using the best
available identifier:

```python
def get_dedup_key(self) -> str:
    if self.doi:   return f"doi:{self.doi.lower().strip()}"   # Most reliable
    if self.pmid:  return f"pmid:{self.pmid}"
    return f"title:{self.title.lower()[:80]}"                  # Fuzzy fallback
```

---

## 5. BaseCollector

**File:** `collectors/base_collector.py`

`BaseCollector` is the **abstract base class** that all 6 collectors inherit from.
It provides shared infrastructure so that each collector only needs to implement 3 methods:
`build_query()`, `fetch_page()`, and `parse_record()`.

### What BaseCollector provides

#### Rate Limiting
Every source has a different allowed request rate. Without rate limiting, APIs block you.
The base class enforces a configurable minimum gap between requests:

```
Source         Rate Limit
───────────────────────────────
PubMed         0.4s (2.5 req/sec without key, 0.1s with key)
Europe PMC     0.5s
Semantic Scholar  1.0s
OpenAlex       0.1s (polite pool: 100 req/sec with email)
Crossref       0.02s (polite pool: 50 req/sec with email)
CORE           0.6s
```

```python
def _wait_for_rate_limit(self):
    elapsed = time.time() - self._last_request_time
    wait = self._rate_limit_seconds - elapsed
    if wait > 0:
        time.sleep(wait)
    self._last_request_time = time.time()
```

#### Automatic Retry on Network Failure
Uses `tenacity` to retry failed HTTP requests with exponential backoff:

```
Attempt 1 → fails → wait 2s
Attempt 2 → fails → wait 4s
Attempt 3 → fails → wait 8s
Attempt 4 → fail  → give up, log error
```

This handles transient network errors, temporary API downtime, and brief rate limit bursts
without crashing the whole collection run.

#### Content Hash Checking
Each paper gets an MD5 hash of its `title + abstract`. When a paper is re-fetched in a
future run, the hash is compared. If the hash hasn't changed, the paper hasn't been
corrected or updated — no need to reprocess it in Layer 2.

#### Raw Response Caching
Every raw API response is saved to `data/raw/<source>/` as JSON. This serves as a local
cache — if something goes wrong downstream, you don't need to re-hit the API.

#### User-Agent Identification
All requests identify the tool to the API:
```
User-Agent: MicrobiomeMiner/1.0 (Academic research; contact@example.com)
```
Academic APIs require this. Anonymous scrapers get blocked faster.

---

## 6. The Six Data Source Collectors

The system queries 6 different academic databases in parallel. Each one covers different
strengths, so together they maximize recall (finding more relevant papers) while the
relevance filter maximizes precision (keeping only the right ones).

### Why 6 sources?

The same paper can appear in multiple databases — but each database also has **unique papers**:

| Source | Best At | Unique Advantage |
|---|---|---|
| PubMed | Precision, MeSH terms | Human-curated vocabulary, Humans filter |
| Europe PMC | Full-text, preprints | PMCIDs, medRxiv coverage |
| Semantic Scholar | Citation counts | Token pagination, AI field tags |
| OpenAlex | Scale (250M+), concepts | Concept IDs, funding metadata |
| Crossref | DOI authority, funding | Canonical DOIs, funder names |
| CORE | Open-access full text | Actual PDF content for NLP |

---

### 6.1 PubMed Collector

**File:** `collectors/pubmed_collector.py`  
**API:** NCBI E-utilities (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils`)  
**Rate:** 3 req/sec (no key) or 10 req/sec (with `NCBI_API_KEY`)

#### Why PubMed?

PubMed is the **gold standard** for biomedical literature. Every indexed paper is manually
reviewed by NCBI librarians who assign MeSH (Medical Subject Headings) — a controlled
vocabulary. This means:

- A paper about "gut flora" is tagged with `Gastrointestinal Microbiome` even if that exact
  phrase never appears in the text
- The `Humans` MeSH term reliably filters out animal studies
- We get higher precision from PubMed than any other source

#### The 2-Step E-utilities Process

PubMed's API uses a two-step process called **WebHistory**:

```
Step 1: esearch (register query server-side)
─────────────────────────────────────────────
POST https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
params: {term: "...", usehistory: "y", retmax: 0}

Returns: {WebEnv: "MCID_...", querykey: "1", count: 42000}
         └── Server stores your query results under WebEnv

Step 2: efetch (retrieve records using WebHistory)
────────────────────────────────────────────────────
POST https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
params: {WebEnv: "MCID_...", query_key: "1", retstart: 0, retmax: 200}

Returns: XML with up to 200 full paper records
```

**Why WebHistory?** Direct PMID pagination has a hard limit of `retstart ≤ 9,999`.
WebHistory has no such limit — you can page through all 70,000+ results by incrementing
`retstart` in steps.

#### Monthly Date-Range Strategy

PubMed limits each WebHistory session to 9,999 records. For the full 2024–2026 date range,
there are ~100,000+ microbiome papers. The solution: split the date range into **monthly
sub-ranges**, each getting its own WebHistory session.

```
2024-01 → esearch → WebEnv_A → efetch page 0...9999  
2024-02 → esearch → WebEnv_B → efetch page 0...9999  
...  
2026-12 → esearch → WebEnv_AJ → efetch page 0...9999  
```

36 months × up to 9,999 records = up to 360,000 papers total.

#### The Query Built

```
(
  "Microbiota"[MeSH Terms] OR
  "Gastrointestinal Microbiome"[MeSH Terms] OR
  "RNA, Ribosomal, 16S"[MeSH Terms] OR
  "Metagenomics"[MeSH Terms] OR
  "Bacteria"[MeSH Terms] OR
  "human microbiome"[Title/Abstract]
)
AND "2024/01/01"[PDAT]:"2026/12/31"[PDAT]
AND "Humans"[MeSH Terms]
```

The `Humans` MeSH filter is critical — it excludes mouse, zebrafish, and soil microbiome
papers at the query level before any data is transferred.

#### XML Parsing

PubMed returns XML. Each article has this structure:

```xml
<PubmedArticle>
  <MedlineCitation>
    <PMID>38765432</PMID>
    <Article>
      <ArticleTitle>Gut microbiome composition in IBD...</ArticleTitle>
      <Abstract>
        <AbstractText Label="Background">...</AbstractText>
        <AbstractText Label="Methods">...</AbstractText>
        <AbstractText Label="Results">...</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
      </AuthorList>
      <Journal>
        <Title>Nature</Title>
        <JournalIssue><Volume>625</Volume><PubDate><Year>2024</Year></PubDate></JournalIssue>
      </Journal>
      <PublicationTypeList>
        <PublicationType>Journal Article</PublicationType>
      </PublicationTypeList>
      <ELocationID EIdType="doi">10.1038/s41586-024-07999-z</ELocationID>
    </Article>
    <MeshHeadingList>
      <MeshHeading><DescriptorName>Gastrointestinal Microbiome</DescriptorName></MeshHeading>
      <MeshHeading><DescriptorName>Humans</DescriptorName></MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
</PubmedArticle>
```

The parser extracts all fields and produces a `PaperRecord`. Structured abstracts (with
Background/Methods/Results labels) are joined together with their labels preserved.

---

### 6.2 Europe PMC Collector

**File:** `collectors/europepmc_collector.py`  
**API:** `https://www.ebi.ac.uk/europepmc/webservices/rest/search`  
**Rate:** 0.5s between requests  
**Format:** Clean JSON (no XML parsing needed)

#### Why Europe PMC in addition to PubMed?

Three reasons:

1. **Full text for open-access papers.** Europe PMC serves full-text for all PMC papers.
   This means Methods and Data Availability sections — not just abstracts.
2. **Preprint coverage.** It indexes medRxiv, bioRxiv preprints that haven't reached PubMed yet.
3. **Clean JSON API.** Much easier to work with than PubMed's XML.

#### Two-Layer Query Strategy

Europe PMC uses a different query strategy because it lacks PubMed's `Humans` MeSH filter:

```
Positive layer — require human context:
  TITLE:"human microbiome" OR TITLE:"gut microbiome" OR
  MH:"Gastrointestinal Microbiome" OR
  (ABSTRACT:"human" AND ABSTRACT:"microbiome")

Negative layer — exclude known non-human topics:
  NOT TITLE:"zebrafish"
  NOT TITLE:"mouse model"
  NOT TITLE:"soil microbiome"
  NOT TITLE:"marine microbiome"
  NOT TITLE:"fermented food"
  ...
```

Both layers together achieve high precision AND recall. Positive-only would miss papers
that don't say "human" explicitly. Negative-only would miss papers where exclusion terms
aren't in the title.

#### Pagination

Europe PMC uses simple 1-indexed page numbers (`page=1, 2, 3...`) with a max of 1000
results per page. The collector loops until `max_results` is reached or results are exhausted.

---

### 6.3 Semantic Scholar Collector

**File:** `collectors/semantic_scholar_collector.py`  
**API:** `https://api.semanticscholar.org/graph/v1/paper/search/bulk`  
**Rate:** 1 req/sec (with API key)  
**Pagination:** Token-based (opaque continuation tokens)

#### Why Semantic Scholar?

The key unique advantage: **citation counts**. Semantic Scholar provides accurate, up-to-date
citation counts for every paper. High citation count is a strong signal of paper quality and
importance for the knowledge graph.

Also useful: it tags papers by field of study (`Biology`, `Medicine`) and has strong coverage
of AI/ML papers that intersect with microbiome research.

#### The Bulk Search Endpoint

This collector uses the `/paper/search/bulk` endpoint (not `/paper/search`). The difference:
- `/paper/search` — relevance-ranked, 10,000 paper limit
- `/paper/search/bulk` — chronological, no effective limit, designed for bulk retrieval

```
Query: "human microbiome metagenomics microbiota"
Year:  "2024-2026"
Fields of Study: Biology, Medicine
Sort: publicationDate:desc
```

#### Token-Based Pagination

Unlike PubMed (offset) or OpenAlex (cursor strings), Semantic Scholar uses short opaque tokens:

```
Request 1: no token → get {total: 42000, token: "abc123", data: [...1000 papers]}
Request 2: token="abc123" → get {total: 42000, token: "def456", data: [...1000 papers]}
Request 3: token="def456" → get {data: [...500 papers], no token} ← last page
```

When no `token` appears in the response, you've reached the end. The token is saved to the
cursor file so the next run can resume from exactly where it stopped.

---

### 6.4 OpenAlex Collector

**File:** `collectors/openalex_collector.py`  
**API:** `https://api.openalex.org/works`  
**Rate:** 0.1s (polite pool: 100 req/sec with email param)  
**Free to use:** No API key required

#### Why OpenAlex?

OpenAlex is the largest open academic graph (250M+ works). It's most useful for:

1. **Concept filtering.** Rather than keyword search, we filter by OpenAlex's concept IDs —
   stable identifiers in a controlled vocabulary hierarchy. This is more precise and
   catches synonyms automatically.
2. **Funding metadata.** Author affiliations, country codes, funder names — unique among
   our collectors and important for the Layer 3 knowledge graph.
3. **Scale.** Covers books, datasets, conference papers, grey literature that PubMed misses.

#### Concept-Based Filtering

Instead of keyword search, we use OpenAlex concept IDs:

```python
MICROBIOME_CONCEPTS = [
    "C2778793",    # Human microbiome (most specific)
    "C185592680",  # Gut microbiota
    "C2776943",    # Microbiota
    "C2781022",    # Metagenomics
    "C2781029",    # 16S rRNA
]
```

These are assigned by OpenAlex's ML classifier to papers — more robust than matching
keywords against free text.

#### Abstract Reconstruction (Inverted Index)

OpenAlex stores abstracts as an **inverted index** — a dictionary mapping each word to
the list of positions it appears at. This is a compression technique:

```json
"abstract_inverted_index": {
    "The":      [0],
    "gut":      [1, 15, 42],
    "microbiome": [2],
    "plays":    [3],
    ...
}
```

The collector reconstructs the original abstract by sorting all `(word, position)` pairs
and joining them:

```python
def _reconstruct_abstract(self, inverted_index: dict) -> Optional[str]:
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(word for _, word in word_positions)
```

#### Cursor-Based Pagination

OpenAlex uses opaque cursor strings. The pattern is the same as Semantic Scholar's token
pagination — start with `cursor=*` (wildcard), then use the `next_cursor` from each
response. The cursor is saved for cross-run resumption.

---

### 6.5 Crossref Collector

**File:** `collectors/crossref_collector.py`  
**API:** `https://api.crossref.org/works`  
**Rate:** 0.02s (polite pool: 50 req/sec with email in User-Agent)  
**Free to use:** No API key required

#### Why Crossref?

Crossref is where **DOIs are registered**. Every paper with a DOI has a Crossref record,
making it the most authoritative source for:

- **DOI completeness.** Papers that PubMed indexed without a DOI often have one in Crossref.
  Critical for our deduplication accuracy — DOI is the primary dedup key.
- **Funding information.** Funder names and grant IDs are unique to Crossref. Essential for
  Layer 3 funding graph nodes.
- **License data.** CC-BY, CC-BY-NC — tells us reuse rights for full text.
- **Publisher metadata.** Canonical journal name, ISSN, volume, issue, pages.

#### Abstract Format: JATS XML Tags

Crossref abstracts come wrapped in JATS XML markup:

```xml
<jats:p>The gut microbiome plays a central role...</jats:p>
<jats:sec>
  <jats:title>Background</jats:title>
  <jats:p>Dysbiosis has been linked to...</jats:p>
</jats:sec>
```

The collector strips all XML tags and normalizes whitespace:

```python
def _strip_jats(self, text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # remove all tags
    return re.sub(r"\s+", " ", text).strip()  # normalize whitespace
```

#### Pagination

Crossref uses simple offset pagination (`rows` + `offset`). Sorted by `published:desc`
(publication date) because relevance sort breaks at deep offsets (>9,999).

---

### 6.6 CORE Collector

**File:** `collectors/core_collector.py`  
**API:** `https://api.core.ac.uk/v3/search/works` (POST)  
**Rate:** 0.6s between requests  
**API Key:** Free registration at `https://core.ac.uk/services/api`

#### Why CORE?

CORE is the world's largest open-access aggregator. Its **unique advantage** over all other
collectors: it returns `fullText` — the actual parsed content of the paper, not just the
abstract.

```json
{
  "title": "Gut microbiome in IBD",
  "abstract": "Background: ...",
  "fullText": "Introduction\nInflammatory bowel disease (IBD) affects...\n\nMethods\n...",
  "downloadUrl": "https://core.ac.uk/download/pdf/12345.pdf"
}
```

This is pure gold for Layer 2 NLP — full text extracts bacterial strain names, sequencing
protocols, and dataset accession numbers from Methods sections that would otherwise be
invisible in abstract-only analysis.

#### POST-Based Search

Unlike all other collectors (which use GET), CORE's search endpoint uses POST with a JSON
body:

```json
POST /v3/search/works
{
    "q": "(title:\"microbiome\" OR abstract:\"human microbiome\") AND yearPublished>=2024",
    "limit": 100,
    "offset": 0,
    "sort": "yearPublished:desc"
}
```

#### Rate Limits

Without an API key: only 100 tokens/day (extremely limited).
With a free registered key: 1,000 tokens/day, 25/min — sufficient for collection runs.
Get a free key at `https://core.ac.uk/services/api`.

---

## 7. The Orchestrator

**File:** `collectors/orchestrator.py`  
**Class:** `CollectionOrchestrator`

The orchestrator is the **single entry point** for Layer 1. It coordinates all 6 collectors,
merges their results, runs deduplication, and triggers the relevance filter.

### Parallel Collection

All 6 collectors run simultaneously using Python's `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=6) as executor:
    future_to_collector = {
        executor.submit(self._run_collector, collector, ...): collector
        for collector in self.collectors
    }
    for future in as_completed(future_to_collector):
        records, cursor_updates = future.result()
        all_records.extend(records)
```

**Why threads and not processes?** Each collector spends most of its time waiting for
HTTP responses. Python threads release the GIL during I/O waits, so 6 collectors can
all be waiting on different network responses simultaneously. This is the same principle
as using I/O threads in any web server. We're not increasing pressure on any single API —
just no longer waiting for source A to finish before starting source B.

### Deduplication and Merging

The same paper appears in multiple sources routinely. Nature publishes in PubMed, Europe
PMC, Semantic Scholar, OpenAlex, and Crossref simultaneously.

**Deduplication strategy:**

```
Priority: DOI → PMID → normalized title

For each unique dedup key:
  If paper only appears once → use it as-is
  If paper appears in multiple sources → MERGE:
    - MeSH terms, publication dates → from PubMed (most authoritative)
    - Citation count               → from Semantic Scholar (most up-to-date)
    - PMCID, full_text_url         → from Europe PMC (most complete)
    - DOI, journal metadata        → from Crossref (canonical)
    - source field                 → "merged:pubmed+europepmc+semantic_scholar"
```

Merging gives us the **best available value for every field** regardless of which source
it comes from.

### Cursor Persistence

After each collector finishes, its current position (offset/token/cursor) is saved to
`data/processed/collector_cursors.json`. The next run loads this file and each collector
resumes from where it stopped:

```json
{
  "pubmed": 15000,
  "europepmc": 3000,
  "semantic_scholar_token": "eyJhbGci...",
  "openalex_cursor": "IljDJLqPk...",
  "crossref": 2000,
  "core": 500
}
```

This is critical for large collections — you can stop and restart without losing progress.

---

## 8. The 4-Stage Relevance Filter Pipeline

**File:** `collectors/relevance_filter.py`  
**Class:** `RelevanceFilter`

After all papers are collected and deduplicated, every paper runs through a **4-stage
relevance filter** designed on a single principle: **use the cheapest stage first.**

The filter works like a funnel:

```
All collected papers
        │
        ▼
Stage 1 (MeSH — instant, free)
  → Confident KEEP or REJECT → done
  → Borderline → Stage 2
        │
        ▼
Stage 2 (Keyword rules — fast)
  → score ≥ 0.70 → KEEP → Metagenomics Gate
  → score < 0.40 → REJECT
  → 0.40–0.69 → Stage 3
        │
        ▼
[Metagenomics Gate — applied to Stage 2 confident keeps]
        │
        ▼
Stage 3 (ML Classifier — medium cost)
  → Confident → done
  → Borderline → Stage 3.5
        │
        ▼
Stage 3.5 (Embedding similarity — medium cost)
  → Confident → done
  → Disagreement/uncertain → Stage 4
        │
        ▼
Stage 4 (LLM — expensive, only for true borderline papers)
  → Final keep/reject verdict
```

### Routing Logic by Source

PubMed papers with MeSH → Stage 1 first, then Stage 2 if borderline.  
All other sources → Stage 2 directly, then Stage 3 if borderline.

---

### 8.1 Stage 1 — MeSH Metadata Filter

**File:** `collectors/metadata_filter.py`  
**Class:** `MetadataFilter`  
**Cost:** Zero — pure dictionary lookups, no model inference

#### What is MeSH?

MeSH (Medical Subject Headings) is a controlled vocabulary maintained by NCBI.
Every PubMed paper is manually reviewed by a librarian who assigns standardized tags.
This is the highest-quality signal we have because it's human-curated, not algorithmic.

Three lists are loaded from `config/stage1_mesh.yaml`:

```yaml
mesh_keep:         # Microbiome-specific terms
  - gastrointestinal microbiome
  - microbiota
  - metagenomics
  - dysbiosis
  - fecal microbiota transplantation

mesh_human_signal: # Human subject terms
  - humans
  - adult
  - aged
  - child
  - infant
  - adolescent

mesh_animal_only:  # Non-human model terms
  - animals
  - mice
  - rats
  - zebrafish
  - swine
  - cattle
```

#### Decision Logic

```
has_microbiome MeSH AND has_human MeSH → KEEP (score 0.90)
has_microbiome AND has_human AND has_animal → KEEP (score 0.70, human wins)
has_animal MeSH AND NOT has_human → REJECT (score 0.05)
has_microbiome MeSH, no human/animal → UNKNOWN (score 0.45, pass to Stage 2)
no microbiome MeSH → UNKNOWN (score 0.10, pass to Stage 2)
no MeSH at all → UNKNOWN (score 0.0, pass to Stage 2)
```

Papers without MeSH (all non-PubMed papers, and PubMed papers not yet indexed) get
`UNKNOWN` and proceed to Stage 2.

---

### 8.2 Metagenomics Gate

**File:** `config/metagenomics_gate.yaml`  
**Controlled by:** `METAGENOMICS_GATE_ENABLED` in `.env` (default: `true`)

The metagenomics gate is a **project-specific requirement** applied after Stage 2's
confident-keep decisions. Even a paper confidently identified as human microbiome research
must mention at least one of the 200+ terms in this list to pass.

The gate enforces the project's core requirement: only papers that involve actual
sequencing data or data availability are included.

**Categories of terms in the gate:**

- Sequencing methods: `metagenom`, `shotgun`, `16s`, `amplicon`, `nanopore`, `illumina`...
- Bioinformatics tools: `qiime`, `dada2`, `metaphlan`, `kraken`, `humann`...
- Quality control: `fastqc`, `trimmomatic`, `fastp`, `denoising`, `chimera removal`...
- Taxonomic analysis: `alpha diversity`, `beta diversity`, `shannon`, `otu`, `asv`...
- Public repositories: `prjna`, `srr`, `sra`, `mgnify`, `bioproject`, `data availab`...
- Study design: `cohort study`, `fecal sample`, `fecal microbiota transplant`...

**Why a separate gate?** The relevance filter stages focus on "is this a human microbiome
paper?" The metagenomics gate focuses on "does this paper actually involve microbiome data
generation or analysis?" A paper about gut health without any sequencing is excluded.

---

### 8.3 Stage 2 — Weighted Keyword Scorer

**File:** `config/stage2_rules.yaml` (rules)  
**Used by:** `collectors/relevance_filter.py`  
**Cost:** Very fast — string matching

Stage 2 computes a weighted score by scanning the concatenated text of:
`title + abstract + mesh_terms + keywords + article_types + journal`

#### How the Score is Computed

```
text = (title + abstract + mesh_terms + keywords + article_types + journal).lower()

For each matched positive term: score += weight
For each matched negative term: score += weight (negative)

raw_score = sum of all matched weights
clamped   = max(-1.0, min(1.5, raw_score))
normalized = (clamped - (-1.0)) / (1.5 - (-1.0))  → [0.0, 1.0]
```

#### Weight Guide for Positive Terms

```
0.70–0.80 → Very strong signal
  "human gut microbiome": 0.80
  "human gut microbiota": 0.80
  "gut metagenomics": 0.75

0.40–0.59 → Strong signal
  "fecal transplant": 0.55
  "clinical trial microbiome": 0.50
  "inflammatory bowel disease": 0.45

0.20–0.39 → Supporting signal
  "gut bacteria": 0.30
  "intestinal flora": 0.25

0.10–0.19 → Weak positive
  "gut": 0.10
  "stool": 0.12
```

#### Weight Guide for Negative Terms

```
-0.80 to -1.00 → Confident off-topic
  "soil microbiome": -0.90
  "zebrafish": -0.85
  "gnotobiotic mouse": -0.80

-0.50 to -0.79 → Likely off-topic
  "murine": -0.65
  "marine microbiome": -0.60

-0.20 to -0.49 → Contextual penalty
  "in vitro": -0.30
  "cell line": -0.35
```

#### Score Thresholds

```
score ≥ 0.70 → Confident KEEP → proceed to Metagenomics Gate
score 0.40–0.69 → Borderline → proceed to Stage 3 ML
score < 0.40 → Confident REJECT
```

---

### 8.4 Stage 3 — ML Classifier

**File:** `collectors/ml_classifier.py`  
**Class:** `MLClassifier`  
**Model:** `sentence-transformers/all-MiniLM-L6-v2` + `LogisticRegression`  
**Status:** Inactive until trained (needs 500+ papers first)

#### Architecture: Why this approach?

For 300–1,000 training samples, a fine-tuned BERT is overkill and slow to train.
`LogisticRegression` on sentence-transformer embeddings achieves F1 > 0.90 and is
100x faster. It's the right tool at this scale.

#### How it works

1. Paper's `title + abstract` is encoded into a 384-dimensional vector by `all-MiniLM-L6-v2`
2. The trained `LogisticRegression` classifier maps the vector to a probability (0.0–1.0)
3. The probability is compared to auto-computed thresholds

```
prob ≥ keep_threshold   → KEEP
prob ≤ reject_threshold → REJECT
between both           → BORDERLINE → Stage 3.5
```

#### Self-Supervised Bootstrap Training

Training uses Stage 2's rule scores as pseudo-labels — no manual annotation needed:

```
Stage 2 score > 0.85 → label as relevant (1)
Stage 2 score < 0.15 → label as off-topic (0)
0.15–0.85 → excluded from training (uncertain label)
```

Then:
1. Encode all pseudo-labeled papers with `all-MiniLM-L6-v2`
2. Cross-validate `LogisticRegression` (5-fold stratified CV)
3. Train final model on full pseudo-labeled dataset
4. Auto-compute optimal thresholds from the probability distribution

#### Auto-Computed Thresholds

Instead of hardcoding 0.85/0.15, the system finds the optimal thresholds from the
actual training data distribution:

- **KEEP threshold** = lowest probability where precision ≥ 90%
  ("Only say KEEP when we're at least 90% sure")
- **REJECT threshold** = highest probability where recall of negatives ≥ 90%
  ("Only say REJECT when we're at least 90% sure it's irrelevant")

This adapts to your data — if 80% of papers are relevant, the natural boundary is
much lower than 0.85.

To train: `RUN_LAYER=train_filter python main.py`

---

### 8.5 Stage 3.5 — Embedding Similarity Filter

**File:** `collectors/embedding_filter.py`  
**Class:** `EmbeddingFilter`  
**Model:** `allenai/specter2` (768-dim, scientific papers) with `all-MiniLM-L6-v2` fallback  
**Cost:** Medium — requires encoding + cosine similarity against stored embeddings

Stage 3.5 compares a new paper's embedding against two partitions of previously-seen papers:

- **Positive partition:** Papers that were kept as relevant
- **Negative partition:** Papers that were rejected as off-topic

#### Embedding Model

**File:** `collectors/embedding_model.py`

The primary model is `allenai/specter2`, a 768-dimensional model pre-trained specifically
on scientific papers. It understands that "16S rRNA" and "amplicon sequencing" are closely
related in a way a general-purpose model might not.

The model handles OOM gracefully by halving the batch size and retrying:

```
batch_size=64 → OOM → batch_size=32 → OOM → batch_size=16 → success
```

Papers are encoded as `"title [SEP] abstract"` for a combined representation.

#### Embedding Store

**File:** `collectors/embedding_store.py`

Stores embeddings as numpy arrays on disk (`.npy` files) with JSON metadata:

```
data/embeddings/
  positive_embeddings.npy    # (N, 768) float32 matrix
  positive_metadata.json     # list of {doi, title, decision, ...}
  negative_embeddings.npy    # (M, 768) float32 matrix
  negative_metadata.json
```

#### Decision Logic

```
pos_sim ≥ 0.85 AND neg_sim < CROSS_CEILING → KEEP
neg_sim ≥ 0.85 AND pos_sim < CROSS_CEILING → REJECT
else → BORDERLINE → Stage 4
INSUFFICIENT_DATA → if < MIN_PARTITION_SIZE embeddings in either partition
```

This stage only activates once enough positive and negative examples have accumulated.
`HYBRID_MIN_STORE_SIZE = 2000` papers is the threshold for the full hybrid classifier.

---

### 8.6 Stage 4 — LLM Verifier

**File:** `collectors/llm_verifier.py`  
**Class:** `LLMVerifier`  
**Backend:** Local Ollama instance (`llama3` by default)  
**Cost:** High — full LLM inference per paper  
**Used for:** Only the most borderline papers that all previous stages couldn't decide on

#### Why a local LLM?

Borderline papers are genuinely ambiguous — they mention "microbiome" and "humans" but it's
unclear if the study design qualifies. A local Ollama LLM gives the system the ability to
reason over the full title and abstract and make a judgment call.

Using a local model (not an API) means:
- No per-call cost
- No rate limits
- Full reproducibility (same prompt → same answer)
- No data leaves your machine

#### The Prompt

The system prompt gives the LLM highly specific criteria:

**Include (keep: true) if:**
- Human subjects (patients, cohorts, clinical trials, RCTs)
- Microbiome/microbiota analysis
- At least one of: sequencing method, bioinformatics tool, or clinical outcome

**Exclude (keep: false) if:**
- Animal models (mouse, zebrafish, rat) as primary focus
- Environmental microbiomes (soil, marine, wastewater)
- Food fermentation (kombucha, kefir, cheese)
- Pure pathogen study with no microbiome community analysis
- In vitro only (no human subjects)

#### Few-Shot Examples

The prompt includes 8 pre-labeled examples to anchor the LLM's decision-making:

```
Title: "Gut microbiome dysbiosis in IBD: 16S rRNA profiling of 200 patients"
→ keep: true, confidence: 0.98

Title: "Zebrafish gut microbiome response to antibiotics"
→ keep: false, confidence: 0.99

Title: "Soil microbiome diversity in agricultural fields"
→ keep: false, confidence: 0.99
```

#### Output Format

```json
{
  "keep": true,
  "confidence": 0.95,
  "reason": "human IBD cohort with 16S sequencing and clinical outcome",
  "human": true,
  "microbiome": true,
  "metagenomics": true,
  "animal": false,
  "environmental": false,
  "article_type": "case-control"
}
```

#### Caching

Every LLM response is cached in `data/processed/llm_cache.json` by MD5 hash of the
`title + abstract`. Re-runs never call the LLM twice for the same paper — critical for
reproducibility (same verdict every time, important for research).

#### Semantic Cache

An additional `SemanticCache` layer catches near-duplicate papers without calling the LLM:

```
New paper embedding → cosine similarity against all previously LLM-verified papers
similarity > 0.97 → reuse the cached verdict (these papers are essentially the same)
similarity ≤ 0.97 → call the LLM
```

#### Batched Verifier

For efficiency, `BatchedVerifier` groups up to 16 borderline papers into a single LLM
call. The LLM returns an array of verdicts, one per paper. On parse failure, the batch
is split in half and retried recursively until success or single-paper level.

#### Starting Ollama

```bash
ollama serve                  # Start Ollama
ollama pull llama3            # Download the model (first time only)
```

Set `OLLAMA_VERIFIER_MODEL=llama3` in `.env` (default is already `llama3`).
Set `OLLAMA_BASE_URL=http://localhost:11434` (default).

---

## 9. Audit & Observability

**File:** `collectors/audit_logger.py`  
**Class:** `AuditLogger`

Every keep/reject/review decision is logged to a JSON file in `data/audit/`:

```
data/audit/
  kept.json         # All kept papers with reason and stage
  rejected.json     # All rejected papers with reason and stage
  review.json       # Borderline papers flagged for human review
  llm_verified.json # Papers that went through the LLM (Stage 4)
```

Each entry includes:

```json
{
  "title": "Gut microbiome composition in IBD...",
  "source": "pubmed",
  "year": 2024,
  "decision": "keep",
  "stage": "stage1_metadata",
  "score": 0.90,
  "reason": "human+microbiome_mesh:humans,gastrointestinal microbiome",
  "doi": "10.1038/...",
  "pmid": "38765432",
  "abstract": "..."
}
```

This makes the pipeline **fully auditable** — you can always trace exactly why any paper
was kept or rejected, at which stage, and with what score.

**Metrics Logger** (`collectors/metrics_logger.py`) records per-run pipeline statistics:
how many papers each stage processed, how many it kept/rejected/flagged for review.

**Stage 2 Calibrator** (`collectors/stage2_calibrator.py`) auto-calibrates Stage 2's
keep/review thresholds based on observed decision distributions, preventing threshold
drift over time as the paper corpus grows.

---

## 10. PMC Full-Text Enricher

**File:** `collectors/pmc_enricher.py`  
**Class:** `PMCEnricher`

The PMC Enricher is not a primary collector — it doesn't find new papers. It **upgrades**
papers already collected by fetching their full structured text from PubMed Central.

### Why full text matters

Abstract-only NLP misses:
- Specific bacterial strains mentioned only in Methods
- Statistical results and effect sizes (only in Results)
- Dataset accession numbers like `PRJNA123456` (only in Data Availability)
- Sequencing protocols and parameters (only in Methods)

Full text gives Layer 2 NLP approximately 10x more extractable content per paper.

### How it works

Any paper with a `pmcid` field can have its full text fetched via the NCBI efetch API:

```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
  ?db=pmc&id=11234567&rettype=xml&retmode=xml
```

The response is structured JATS XML. The enricher extracts:

```
## Introduction
[text]

## Methods
[text]

## Results
[text]

## Discussion
[text]

## Data Availability
[accession numbers, repository links]

## Funding
[grant information]
```

The extracted text is stored in `paper.full_text` and the `full_text_url` is set to
the canonical PMC article URL.

### Rate limiting

Uses `NCBI_API_KEY` if available (10 req/sec), otherwise 3 req/sec.
A `max_enrichments` cap (default: 50 per run) prevents slow runs when many PMCIDs exist.

> **Note:** As of the current implementation, PMC enrichment is deferred to Layer 2.
> Layer 1 collects the PMCIDs; Layer 2 triggers full-text fetching as part of its
> NLP enrichment pass.

---

## 11. Complete Data Flow Diagram

```
main.py → run_layer1()
    │
    │  Load logging (stderr + file rotation)
    │  Verify sentence-transformers available
    │
    └─→ CollectionOrchestrator.collect_all(query, date_from, date_to, max_per_source)
            │
            │  Load cursor file (for cross-run resumption)
            │
            ├─→ [ThreadPoolExecutor — 6 threads, one per source]
            │       │
            │       ├── PubMedCollector.collect()
            │       │     │
            │       │     ├── build_query() → MeSH + keyword + date + Humans filter
            │       │     ├── Split 2024–2026 into 36 monthly sub-ranges
            │       │     ├── For each month:
            │       │     │     ├── esearch (get WebEnv + query_key)
            │       │     │     └── efetch (page through XML in batches of 200)
            │       │     ├── parse_record() → PaperRecord for each article
            │       │     └── Returns List[PaperRecord]
            │       │
            │       ├── EuropePMCCollector.collect()
            │       │     ├── build_query() → positive + negative title/abstract filters
            │       │     ├── fetch_page() → JSON, 1000 records/page
            │       │     ├── parse_record() → PaperRecord
            │       │     └── Returns List[PaperRecord]
            │       │
            │       ├── SemanticScholarCollector.collect()
            │       │     ├── build_query() → keyword + year + fields of study
            │       │     ├── Token-based pagination (bulk endpoint, 1000/request)
            │       │     ├── parse_record() → PaperRecord
            │       │     └── Returns List[PaperRecord]
            │       │
            │       ├── OpenAlexCollector.collect()
            │       │     ├── build_query() → concept IDs + year + type:article
            │       │     ├── Cursor-based pagination (200/page, polite pool)
            │       │     ├── _reconstruct_abstract() from inverted index
            │       │     ├── parse_record() → PaperRecord
            │       │     └── Returns List[PaperRecord]
            │       │
            │       ├── CrossrefCollector.collect()
            │       │     ├── build_query() → keyword phrase + date filter
            │       │     ├── fetch_page() → JSON, offset pagination (1000/page)
            │       │     ├── _strip_jats() from abstract
            │       │     ├── parse_record() → PaperRecord
            │       │     └── Returns List[PaperRecord]
            │       │
            │       └── CoreCollector.collect()
            │             ├── build_query() → title/abstract + year range
            │             ├── POST /search/works, offset pagination (100/page)
            │             ├── parse_record() → PaperRecord (with fullText)
            │             └── Returns List[PaperRecord]
            │
            │  [All 6 collectors complete — results pooled]
            │  all_records = concat of all 6 lists
            │
            ├─→ _deduplicate_and_merge(all_records)
            │     │
            │     ├── Group by dedup_key (DOI → PMID → title)
            │     ├── Single occurrence → keep as-is
            │     └── Multiple sources → merge (best value per field)
            │           PubMed wins:   mesh_terms, publication_date
            │           S2 wins:       citation_count
            │           EuropePMC wins: pmcid, full_text_url
            │           Crossref wins: doi, journal_abbrev
            │           source = "merged:pubmed+europepmc+..."
            │
            ├─→ RelevanceFilter.filter(deduplicated_records)
            │     │
            │     └── For each paper → _evaluate(paper):
            │
            │         PubMed papers:
            │           Stage 1 (MetadataFilter.evaluate)
            │             has_human + has_microbiome MeSH → KEEP (0.90)
            │             has_animal, no human → REJECT (0.05)
            │             borderline/no MeSH → UNKNOWN → Stage 2
            │
            │         All papers reaching Stage 2:
            │           Stage 2 (weighted keyword scorer)
            │             Compute weighted score over title+abstract+mesh+keywords
            │             score ≥ 0.70 → confident KEEP
            │               └─→ Metagenomics Gate
            │                     must mention sequencing/data term → pass/fail
            │             score 0.40–0.69 → borderline → Stage 3
            │             score < 0.40 → confident REJECT
            │
            │         Borderline papers → Stage 3:
            │           Stage 3 (MLClassifier.evaluate)
            │             Encode title+abstract → 384-dim vector
            │             LogisticRegression probability
            │             prob ≥ keep_threshold → KEEP
            │             prob ≤ reject_threshold → REJECT
            │             else → BORDERLINE → Stage 3.5
            │
            │         Stage 3.5 borderline → Stage 3.5:
            │           Stage 3.5 (EmbeddingFilter.evaluate)
            │             Encode with SPECTER2 → 768-dim vector
            │             Cosine similarity vs positive partition
            │             Cosine similarity vs negative partition
            │             pos_sim ≥ 0.85 AND neg_sim low → KEEP
            │             neg_sim ≥ 0.85 AND pos_sim low → REJECT
            │             else → BORDERLINE → Stage 4
            │             insufficient data → INSUFFICIENT_DATA → Stage 4
            │
            │         Stage 4 borderline → Stage 4:
            │           Stage 4 (LLMVerifier.verify)
            │             Check SemanticCache (similarity > 0.97 → reuse verdict)
            │             Check hash cache (exact match → reuse verdict)
            │             Call Ollama llama3 with system prompt + few-shot examples
            │             Parse JSON response
            │             keep=true AND confidence ≥ 0.70 → KEEP
            │             else → REJECT or review_queue
            │
            │     [All decisions logged to data/audit/]
            │     returns kept_papers: List[PaperRecord]
            │
            ├─→ _save_merged(kept_papers)
            │     Writes data/processed/collected_YYYYMMDD_HHMMSS.json
            │
            ├─→ _save_cursors(updated_cursors)
            │     Writes data/processed/collector_cursors.json
            │
            └─→ return kept_papers
                  └── Layer 2 reads this via CollectionOrchestrator.load_latest()
```

---

## 12. Output Format

The output of Layer 1 is a JSON array of `PaperRecord` objects:

```json
[
  {
    "doi": "10.1038/s41586-024-07999-z",
    "pmid": "38765432",
    "pmcid": "PMC11234567",
    "arxiv_id": null,
    "source": "merged:pubmed+europepmc",

    "title": "Gut microbiome composition predicts IBD remission following FMT",
    "abstract": "Background: Fecal microbiota transplantation (FMT)...\nMethods: We performed...",
    "authors": ["Smith, John A", "Jones, Kate M", "Brown, Robert T"],
    "keywords": ["gut microbiome", "IBD", "FMT", "16S rRNA", "dysbiosis"],

    "journal": "Nature",
    "journal_abbrev": "Nature",
    "issn": "0028-0836",
    "publication_date": "2024-03-15",
    "publication_year": 2024,
    "volume": "625",
    "issue": "7994",
    "pages": "1-12",

    "article_types": ["Journal Article", "Randomized Controlled Trial"],

    "is_open_access": true,
    "full_text_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11234567/",
    "pdf_url": null,
    "full_text": null,

    "citation_count": 42,
    "reference_count": 87,

    "mesh_terms": [
      "Gastrointestinal Microbiome",
      "Inflammatory Bowel Diseases",
      "Humans",
      "Fecal Microbiota Transplantation",
      "Metagenomics"
    ],

    "content_hash": "a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5",
    "fetched_at": "2024-05-21T02:00:00.123456",
    "is_preprint": false
  },
  ...
]
```

**File location:** `data/processed/collected_YYYYMMDD_HHMMSS.json`  
**Layer 2 access:** `CollectionOrchestrator.load_latest()` reads the most recent file.

---

## 13. Cross-Run Incrementality (Cursors)

**File:** `data/processed/collector_cursors.json`

Layer 1 is designed to be run repeatedly (e.g., weekly or monthly) to pick up new papers.
Each run saves its current position so the next run continues from where the last one stopped.

```json
{
  "pubmed": 15000,
  "europepmc": 3000,
  "semantic_scholar_token": "eyJhbGciOiJIUzI1NiJ9...",
  "openalex_cursor": "IljDJLqPkAAAAAAAAAABIA==",
  "crossref": 2000,
  "core": 500
}
```

**How cursors work per source:**
- `pubmed`: Encodes `month_index * 10000 + retstart_within_month`
- `europepmc`: Simple page number × page_size offset
- `semantic_scholar_token`: Opaque base64 continuation token
- `openalex_cursor`: Opaque cursor string (starts with `*` for fresh start)
- `crossref`: Simple numeric offset
- `core`: Simple numeric offset

**To reset and re-collect everything from scratch:** Delete `collector_cursors.json`.

---

## 14. Configuration Reference Table

| Setting | Source | Default | What it controls |
|---|---|---|---|
| `SEARCH_QUERY` | config.py | `"human microbiome"` | Base keyword for all sources |
| `DATE_FROM` | config.py | `"2024/01/01"` | Start of publication date range |
| `DATE_TO` | config.py | `"2026/12/31"` | End of publication date range |
| `MAX_RESULTS_PER_SOURCE` | config.py | `500` | Papers per source per run |
| `NCBI_EMAIL` | .env | (required) | PubMed polite pool identification |
| `NCBI_API_KEY` | .env | "" | PubMed 10 req/sec (vs 3 without) |
| `SEMANTIC_SCHOLAR_API_KEY` | .env | "" | S2 rate limit upgrade |
| `CORE_API_KEY` | .env | "" | CORE 1000 tokens/day (vs 100) |
| `LAYER1_SOURCE_WORKERS` | .env | `6` | Parallel collector threads |
| `METAGENOMICS_GATE_ENABLED` | .env | `"true"` | Enforce sequencing-term requirement |
| `EMBEDDING_MODEL_NAME` | config.py | `allenai/specter2` | Stage 3.5 primary embedding model |
| `EMBEDDING_FALLBACK_MODEL` | config.py | `all-MiniLM-L6-v2` | Stage 3.5 fallback model |
| `EMBEDDING_POS_KEEP_THRESHOLD` | config.py | `0.85` | Stage 3.5 confident-keep threshold |
| `EMBEDDING_NEG_REJECT_THRESHOLD` | config.py | `0.85` | Stage 3.5 confident-reject threshold |
| `EMBEDDING_CROSS_CEILING` | config.py | `0.70` | Stage 3.5 cross-similarity ceiling |
| `EMBEDDING_MIN_PARTITION_SIZE` | config.py | `10` | Min embeddings per partition |
| `BLENDED_CONFIDENCE_LOW` | config.py | `0.40` | Stage 2 borderline lower bound |
| `BLENDED_CONFIDENCE_HIGH` | config.py | `0.70` | Stage 2 confident-keep threshold |
| `HYBRID_MIN_STORE_SIZE` | config.py | `2000` | Min papers to activate HybridClassifier |
| `SEMANTIC_CACHE_THRESHOLD` | config.py | `0.97` | LLM semantic cache similarity threshold |
| `BATCH_LLM_SIZE` | config.py | `16` | Max papers per batched LLM call |
| `OLLAMA_BASE_URL` | .env | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VERIFIER_MODEL` | .env | `"llama3"` | LLM model for Stage 4 |
| `OLLAMA_TIMEOUT_SECONDS` | .env | `60` | Timeout per Ollama call |
| `EMBEDDING_STORE_DIR` | .env | `data/embeddings` | Where embedding store files live |

---

## 15. File Reference Map

```
Layer 1 files and their roles:

config.py                              ← Central configuration (all settings)
models.py                              ← PaperRecord data model (shared with all layers)
main.py                                ← Entry point (run_layer1 function)

collectors/
  orchestrator.py                      ← Coordinates all collectors + dedup + filter
  base_collector.py                    ← Abstract base: rate limiting, retry, caching
  pubmed_collector.py                  ← PubMed via NCBI E-utilities (XML, WebHistory)
  europepmc_collector.py               ← Europe PMC via REST JSON
  semantic_scholar_collector.py        ← Semantic Scholar bulk search (token pagination)
  openalex_collector.py                ← OpenAlex via concept filters (cursor pagination)
  crossref_collector.py                ← Crossref /works endpoint (offset pagination)
  core_collector.py                    ← CORE /search/works (POST, full text)
  relevance_filter.py                  ← 4-stage filter pipeline orchestrator
  metadata_filter.py                   ← Stage 1: MeSH-based fast filter
  ml_classifier.py                     ← Stage 3: sentence-transformers + LogisticRegression
  embedding_model.py                   ← SPECTER2 / MiniLM encoder with OOM retry
  embedding_filter.py                  ← Stage 3.5: cosine similarity filter
  embedding_store.py                   ← numpy-backed embedding store (positive/negative)
  hybrid_classifier.py                 ← Meta-classifier combining all signals (≥2000 papers)
  llm_verifier.py                      ← Stage 4: Ollama LLM + SemanticCache + BatchedVerifier
  audit_logger.py                      ← Logs every decision to data/audit/ JSON files
  metrics_logger.py                    ← Records per-run pipeline metrics
  stage2_calibrator.py                 ← Auto-calibrates Stage 2 thresholds
  pmc_enricher.py                      ← Fetches PMC full-text XML for papers with PMCID

config/
  stage1_mesh.yaml                     ← MeSH term lists for Stage 1 filter
  stage2_rules.yaml                    ← Weighted keyword rules for Stage 2 scorer
  metagenomics_gate.yaml               ← 200+ sequencing/data terms for the gate
  relevance_model.pkl                  ← Trained LogisticRegression model (Stage 3)

data/ (auto-created)
  raw/<source>/                        ← Cached raw API responses
  processed/
    collected_YYYYMMDD_HHMMSS.json     ← Layer 1 output (input to Layer 2)
    collector_cursors.json             ← Cross-run position cursors
    llm_cache.json                     ← LLM verdict cache (hash → verdict)
  embeddings/
    positive_embeddings.npy            ← Stage 3.5 positive partition
    negative_embeddings.npy            ← Stage 3.5 negative partition
    llm_verdict_cache.npy              ← SemanticCache vectors
    llm_verdict_cache_meta.json        ← SemanticCache metadata
  audit/
    kept.json                          ← All kept papers with decisions
    rejected.json                      ← All rejected papers with decisions
    review.json                        ← Borderline papers for human review
    llm_verified.json                  ← Papers verified by Stage 4 LLM
```

---

*This document covers Layer 1 of the Human Microbiome Research Knowledge Graph pipeline.*  
*For Layer 2 (NLP Enrichment), Layer 3 (Knowledge Graph), and the Query Layer, see the README.*
