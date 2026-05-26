# Microbiome Literature Miner

An automated pipeline for collecting, processing, and storing research literature on **human microbiome studies (2024–2026)**. Built as an independent project combining literature mining, NLP, and knowledge graph construction.

---

## What this project does

This system continuously collects published research papers from five sources, processes them using NLP to extract structured information, and stores everything in a Neo4j knowledge graph — automatically updating as new papers are published.

```
5 data sources → NLP pipeline → knowledge graph → auto-updates
```

---

## Architecture — 4 layers

### Layer 1 — Data Collection ✅ (complete)
Scrapes and normalizes research papers from PubMed, Europe PMC, Semantic Scholar, bioRxiv/medRxiv, and Google Scholar. Deduplicates across sources using DOI and content hashing.
“Relevance filtering employed a hybrid multi-stage pipeline for human microbiome literature screening. Stage 1 used PubMed MeSH metadata filtering, leveraging human-curated MeSH terms to confidently retain human microbiome studies and reject animal-only records. Stage 2 applied a weighted rule-based relevance scorer driven by an external YAML taxonomy configuration (organisms.yaml) containing positive and negative biomedical terms. A project-specific metagenomics gate then enforced the presence of sequencing- or metagenomics-related signals (e.g., 16S rRNA, shotgun sequencing, metagenomics, accession identifiers) to ensure downstream compatibility with microbiome data mining objectives. Borderline papers were subsequently verified using Gemini Flash LLM classification, where only uncertain cases were evaluated to minimize computational cost. LLM verdicts were cached to ensure reproducibility, reduce repeated API calls, and accelerate reruns. The system additionally maintained audit logs (kept, rejected, review, and LLM-verified) for full traceability and reproducibility of filtering decisions. Across N collected papers, X% were automatically resolved via metadata/rule stages, Y% required LLM verification, and Z% remained in the review queue.”

### Layer 2 — NLP Processing 🔄 (next)
Classifies each paper by type (original research, review, meta-analysis), extracts named entities using BioBERT (taxa, diseases, sequencing methods, body sites), parses sections, and extracts data availability information.

### Layer 3 — Knowledge Graph Storage 🔜
Stores all structured information in Neo4j with nodes for Papers, Authors, Journals, Entities, and Datasets. Supports Cypher queries like *"find all Q1 journal papers from 2025 using shotgun metagenomics that share data on SRA"*.

### Layer 4 — Scheduler 🔜
APScheduler jobs that run daily (new paper checks) and weekly (full metadata re-scan) to keep the database current.

---

## Project structure

```
microbiome_miner/
│
├── collectors/                      # Layer 1: data collection
│   ├── base_collector.py            # Shared rate limiting, retry, caching logic
│   ├── pubmed_collector.py          # PubMed E-utilities (MeSH queries + XML parsing)
│   ├── europepmc_collector.py       # Europe PMC REST API (full-text access)
│   ├── semantic_scholar_collector.py # Semantic Scholar (citation graphs)
│   ├── biorxiv_collector.py         # bioRxiv + medRxiv preprints
│   └── orchestrator.py             # Merges all sources, deduplicates
│
├── nlp/                             # Layer 2: NLP pipeline (coming)
├── graph/                           # Layer 3: Neo4j storage (coming)
├── scheduler/                       # Layer 4: auto-update jobs (coming)
│
├── data/
│   ├── raw/                         # Cached API responses (per source)
│   └── processed/                   # Merged, deduplicated JSON outputs
│
├── logs/                            # Structured logs (loguru)
├── models.py                        # PaperRecord — unified data schema
├── config.py                        # All settings (query, dates, rate limits)
├── main.py                          # Entry point
├── requirements.txt
└── .env.example                     # API key template
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/microbiome-miner.git
cd microbiome-miner
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
NCBI_EMAIL=your_email@example.com

# Recommended (free)
NCBI_API_KEY=...           # https://www.ncbi.nlm.nih.gov/account/
SEMANTIC_SCHOLAR_API_KEY=... # https://www.semanticscholar.org/product/api

# Google Scholar — pick one mode
GOOGLE_SCHOLAR_MODE=scholarly   # free, may get rate-limited
SCRAPER_API_KEY=...             # proxy for scholarly (free tier at scraperapi.com)
# OR
GOOGLE_SCHOLAR_MODE=serpapi
SERPAPI_KEY=...                 # https://serpapi.com/ (paid, reliable)
```

### 3. Run Layer 1

```bash
# Test run — 20 papers per source (~2 min)
MAX_PER_SOURCE=20 python main.py

# Full production run
MAX_PER_SOURCE=500 python main.py
```

Output is saved to `data/processed/collected_YYYYMMDD_HHMMSS.json`.

---

## Data sources

| Source | Type | Papers | Unique value |
|---|---|---|---|
| PubMed | Official API | up to 500 | MeSH terms, article type tags |
| Europe PMC | Official API | up to 500 | Full text for open-access papers |
| Semantic Scholar | Official API | up to 500 | Citation counts, cross-links |
| bioRxiv / medRxiv | Official API | up to 500 | Preprints, cutting-edge work |

All sources are deduplicated by DOI → PMID → normalized title. Metadata is merged (e.g. MeSH terms from PubMed + citation count from Semantic Scholar → one record).

---

## Configuration

All settings are in `config.py`:

```python
SEARCH_QUERY = "human microbiome"
DATE_FROM    = "2024/01/01"
DATE_TO      = "2026/12/31"

PUBMED_MESH_TERMS = [
    "Microbiota",
    "Gastrointestinal Microbiome",
    "RNA, Ribosomal, 16S",
    "Metagenomics",
    ...
]

RATE_LIMITS = {
    "pubmed":          0.4,   # sec between requests
    "europepmc":       0.5,
    "semantic_scholar": 1.0,
    "biorxiv":         0.5,
    "google_scholar":  8.0,   # much slower — avoids blocking
}
```

---

## Output format

Each collected paper is a `PaperRecord`:

```json
{
  "doi": "10.1038/s41586-024-07999-z",
  "pmid": "38765432",
  "pmcid": "PMC11234567",
  "title": "Gut microbiome composition and ...",
  "abstract": "Background: ...",
  "authors": ["Smith J", "Jones K", "Lee M"],
  "journal": "Nature",
  "publication_year": 2024,
  "publication_date": "2024-03-15",
  "article_types": ["Journal Article"],
  "mesh_terms": ["Microbiota", "Gastrointestinal Microbiome"],
  "keywords": ["16S rRNA", "gut bacteria"],
  "is_open_access": true,
  "full_text_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11234567/",
  "citation_count": 42,
  "content_hash": "a3f8b2c1d4e5f6a7",
  "source": "merged:pubmed+europepmc+semantic_scholar",
  "fetched_at": "2024-05-21T02:00:00"
}
```

---

## Roadmap

- [x] Layer 1 — Multi-source data collection (PubMed, Europe PMC, Semantic Scholar, bioRxiv, Google Scholar)
- [x] Unified `PaperRecord` schema with content hashing and deduplication
- [x] Rate limiting, retry with exponential backoff, raw response caching
- [ ] Layer 2 — NLP pipeline (article classifier, journal classifier, BioBERT NER, section parser, data availability extractor)
- [ ] Layer 3 — Neo4j knowledge graph with Paper/Author/Entity/Dataset nodes
- [ ] Layer 4 — APScheduler for daily/weekly auto-updates
- [ ] REST API for querying the knowledge graph
- [ ] Dashboard for visualizing the paper collection

---

## Dependencies

| Package | Purpose |
|---|---|
| `biopython` | PubMed Entrez / E-utilities |
| `requests` + `tenacity` | HTTP with retry |
| `pydantic` | Data validation and schema |
| `loguru` | Structured logging |
| `transformers` | BioBERT NER (Layer 2) |
| `neo4j` | Knowledge graph driver (Layer 3) |
| `apscheduler` | Auto-update jobs (Layer 4) |

---

## Academic context

**Topic**: Literature Mining and Metagenomic Data Preprocessing for Human Microbiome Studies

**Scope**: Research papers published 2024–2026 on human microbiome studies, including gut, oral, skin, and lung microbiome research. Covers original research articles, systematic reviews, meta-analyses, and preprints.

**Focus areas**:
- 16S rRNA amplicon sequencing studies
- Shotgun metagenomics
- Microbiome-disease associations
- Microbiome intervention studies (probiotics, FMT)
- Computational methods for microbiome analysis

---

## License

MIT License — free to use for academic and research purposes.
