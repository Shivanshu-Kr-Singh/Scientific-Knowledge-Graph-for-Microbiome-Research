# Scientific Knowledge Graph for Microbiome Research
## Pipeline Documentation — Layer 1, 2 & 3

---

# Layer 1 — Data Collection Pipeline

## How It Works

### 1. Four Data Sources

The pipeline collects papers from 4 academic sources simultaneously:

- **PubMed** — NCBI's biomedical literature database, accessed via Entrez E-utilities API with MeSH term support
- **Europe PMC** — European life sciences literature database
- **Semantic Scholar** — AI-powered research paper database with citation data
- **bioRxiv/medRxiv** — Preprint servers for biology and medical research

---

### 2. Collection Strategy per Source

**PubMed, EuropePMC, Semantic Scholar:**
- Send keyword query `"human microbiome"` with date range `2024/01/01 → 2026/12/31`
- API returns papers that match the keyword directly
- Fetch exactly `MAX_PER_SOURCE` papers (e.g. 10)
- Papers are already topically relevant because the API does keyword filtering

**bioRxiv/medRxiv:**
- API does NOT support keyword search — only date-range fetching
- Fetches 30 papers from bioRxiv + 30 from medRxiv = 60 total
- Immediately runs a relevance pre-filter on these 60 to find microbiome-related ones
- Only the relevant ones proceed forward
- This pre-filter is necessary here specifically because the API cannot filter by topic

---

### 3. Relevance Filter (4 Stages)

Every paper passes through a 4-stage relevance pipeline:

| Stage | Method | Purpose |
|---|---|---|
| Gate (Metagenomics) | Rule check | Does it mention sequencing/microbiome at all? |
| Stage 1 (Metadata) | MeSH terms, journal, article type | Is it from a relevant journal/field? |
| Stage 2 (Rules) | 60+ scoring rules | Does it score above threshold on content rules? |
| Stage 3 (ML Classifier) | Logistic Regression | Trained classifier for borderline papers (requires 500+ papers to train) |
| Stage 4 (LLM) | Ollama `qwen2.5:1.5b` | For borderline papers — LLM decides keep/reject |

- Papers scoring below 0.5 confidence → **Rejected**
- Papers scoring 0.5–0.6 → **Flagged for human review** (saved to `curator_review_queue.json`)
- Papers above 0.6 → **Kept**

> **Note:** Stage 3 (ML Classifier) is currently inactive. It requires a minimum of 500 collected papers to train. Once trained by running `RUN_LAYER=train_filter python main.py`, it activates and improves relevance filtering accuracy, reducing the load on the LLM in Stage 4.

---

### 4. Deduplication

After all 4 sources are collected and filtered, papers are merged into one pool. Duplicates are removed using:
- **DOI matching** — same DOI = same paper
- **Content hash** — same title/abstract hash = same paper even if from different sources

---

### 5. Output

Final merged, deduplicated, relevance-filtered list saved to:
`data/processed/collected_YYYYMMDD_HHMMSS.json`

Each paper record contains: DOI, PMID, title, abstract, authors, journal, publication year, article types, MeSH terms, open access status, content hash, fetch timestamp.

---

## Example — 10 Papers Per Source (Actual Run)

### Step 1: Collection

```
PubMed           → fetches 10 papers (keyword: "human microbiome", 2024-2026)
EuropePMC        → fetches 10 papers (keyword: "human microbiome", 2024-2026)
Semantic Scholar → fetches 10 papers (keyword: "human microbiome", 2024-2026)
bioRxiv          → fetches 60 papers (date range only, no keyword support)
                    └─ pre-filter → keeps 1 relevant preprint
                    └─ rejects 59 (neuroscience, surgery, genetics, etc.)
```

### Step 2: Merge Pool

```
10 (PubMed) + 10 (EuropePMC) + 10 (Semantic Scholar) + 1 (bioRxiv) = 31 papers
```

### Step 3: Deduplication

```
31 papers → remove 1 duplicate (same paper in PubMed + EuropePMC)
           → 30 unique papers
```

### Step 4: Relevance Filter (29 papers)

```
Input: 29 papers (30 minus bioRxiv's 1 already pre-filtered)

Rejected (3 papers):
  ✗ "Acute Septic Arthritis Caused by Finegoldia magna After ACL Reconstruction"
     Reason: orthopaedic surgery paper, bacteria mentioned incidentally

  ✗ "In vitro and in vivo antibacterial effects of mitochondrial..."
     Reason: antibacterial drug study, not microbiome research

  ✗ "Insights into microbiota profile of Pediculus humanus capitis"
     Reason: head lice microbiome, outside scope of human microbiome research

Flagged for review (1 paper):
  ? "In vitro and in vivo antibacterial effects..." [LLM confidence: 0.42]
     Reason: LLM was 42% confident — borderline case
     Action: saved to curator_review_queue.json for human review

Kept: 26 papers
```

### Step 5: Final Count

```
26 (from PubMed/EuropePMC/Semantic Scholar) + 1 (bioRxiv) = 27 papers
Saved → data/processed/collected_20260608_163715.json
```

### Summary Table

| Source | Fetched | Pre-filtered | After Dedup | After Relevance | Final |
|---|---|---|---|---|---|
| PubMed | 10 | — | 10 | 9 | 9 |
| EuropePMC | 10 | — | 9 | 9 | 9 |
| Semantic Scholar | 10 | — | 10 | 8 | 8 |
| bioRxiv/medRxiv | 60 | 1 | 1 | 1 | 1 |
| **Total** | **90** | **31** | **30** | **27** | **27** |

### Actual Log Output (Layer 1)

```
22:05:47 | INFO  | Starting Microbiome Literature Miner — Layer 1
22:05:51 | INFO  | [pubmed] Collection complete: 10 papers
22:05:54 | INFO  | [europepmc] Collection complete: 10 papers
22:05:57 | INFO  | [semantic_scholar] Collection complete: 10 papers
22:06:01 | INFO  | [biorxiv] Collection complete: 60 papers (pre-filter kept 1)
22:07:04 | INFO  | Total raw records before dedup: 31
22:07:04 | INFO  | Removed 1 duplicate records
22:07:04 | INFO  | After deduplication: 30 unique papers
22:07:15 | INFO  | Relevance filter: kept 27, removed 3, flagged for review: 1
22:07:16 | SUCCESS | Layer 1 complete. 27 papers ready for NLP processing.
```

---
---

# Layer 2 — NLP Enrichment Pipeline

## How It Works

### 1. Input

Reads the latest `collected_YYYYMMDD.json` from Layer 1. In this run: **27 papers**.

---

### 2. NLP Modules Initialized

Five modules load at startup:

| Module | Tool | Purpose |
|---|---|---|
| Article Classifier | Logistic Regression + rules | Classifies paper type (original research, review, meta-analysis, etc.) |
| Journal Classifier | External API + rules | Gets journal quality metrics (quartile, impact factor) |
| NER (Named Entity Recognition) | BioBERT + rules + Ollama | Extracts named entities from text |
| Section Parser | Rule-based | Splits paper into abstract, methods, results, discussion |
| Data Availability | Rule-based + regex | Finds data deposition info (SRA, GEO accessions) |

BioBERT (`d4data/biomedical-ner-all`, 440MB) downloads once on first run and loads into GPU memory (RTX 4050) for the entire pipeline run.

---

### 3. Per-Paper Processing — 3-Tier NER

For each paper, entity extraction runs through 3 tiers in sequence:

**Tier 1 — Rule-based Dictionary** (instant, runs first)
- Matches against ~500+ hardcoded regex patterns
- Covers: Taxa, Diseases, Methods, Metabolites, Genes, Proteins, Body Sites, Treatments, Datasets
- Example patterns: `\bfaecalibacterium prausnitzii\b`, `\btype 2 diabetes\b`, `\b16s rrna sequencing\b`
- High precision, catches all known standard terms instantly

**Tier 2 — BioBERT** (fast with GPU)
- HuggingFace transformer model runs on RTX 4050
- Catches novel entities not in the dictionary
- Returns entity spans with confidence scores
- Model: `d4data/biomedical-ner-all`

**Tier 3 — Ollama LLM** (slow, only for complex papers)
- `qwen2.5:1.5b` used as last resort for entities too complex for rules/BioBERT
- Times out for very long papers (300s limit) → returns empty result
- Paper still fully processed with Tier 1+2 data even if Tier 3 times out

**Entity Normalization** (after extraction)
- Each extracted entity is grounded to an authoritative ontology
- NCBI Taxonomy for taxa (e.g. "gut microbiome" → `ncbi:749906`)
- MeSH for diseases (e.g. "IBD" → `mesh:D015212`)
- OBI/EFO for methods
- UniProt for proteins
- Results cached in SQLite — same entity never looked up twice

---

### 4. Full-Text Acquisition (Unpaywall)

For open-access papers, the pipeline tries to download the full PDF via Unpaywall API for richer entity extraction. Paywalled papers fail silently and fall back to abstract-only processing.

---

### 5. Output

Saves `enriched_YYYYMMDD.json` — same papers but now each has:

| Field | Description |
|---|---|
| `article_type_normalized` | original_research / narrative_review / systematic_review / meta_analysis |
| `article_type_confidence` | Classifier confidence 0.0–1.0 |
| `journal_info` | Quartile (Q1-Q4), impact factor, open access status |
| `entities` | Full list of named entities with text, label, ontology ID, grounding source |
| `taxa` | List of microbial taxa found |
| `diseases` | List of diseases found |
| `methods` | List of methods found |
| `metabolites` | List of metabolites found |
| `genes` | List of genes found |
| `sections` | Parsed sections (abstract, methods, results, discussion) |
| `data_availability` | open/restricted/not_stated + accession numbers |

---

## Example — 27 Papers (Actual Run)

### Processing

```
Loaded:    27 papers from Layer 1
GPU:       RTX 4050 (BioBERT running on CUDA)
Duration:  ~8 minutes (17 seconds per paper average)
```

### Per-Paper Entity Extraction (Sample)

| Paper Title | Entities | Taxa | Diseases | Methods |
|---|---|---|---|---|
| A sustained living coating for infectious keratitis... | 107 | 22 | 13 | 14 |
| UMI-guided single locus sequence typing for Cutibacterium | 19 | 6 | 1 | 7 |
| Gut microbiota and the kidney-gut-skin axis in CKD | 35 | 3 | 6 | 1 |
| The role of gut microbiota in osteoporosis | 24 | 3 | 2 | 0 |
| Gut microbiota dysbiosis in type 1 diabetes mellitus | 29 | 4 | 4 | 0 |
| Gut-bone axis in rheumatoid arthritis | 235 | 18 | 77 | 18 |
| **Average across 27 papers** | **~62** | | | |

### Warnings (Non-blocking)

```
[Unpaywall] Failed for DOI 10.3389/fmicb.2026.1856738
[Unpaywall] Failed for DOI 10.38124/ijisrt/26may280
[Unpaywall] Failed for DOI 10.1101/2023.08.09.23293887
→ 3 paywalled papers. Fell back to abstract-only. Pipeline continued.

OllamaTimeoutError for 1 paper (Tier 3 timeout at 300s)
→ Tier 1+2 data still used. Paper fully processed. Pipeline continued.
```

### NLP Pipeline Summary

```
Total enriched:    27 papers (0 errors)

Article types:
  narrative_review:    15 papers
  original_research:    9 papers
  unknown:              2 papers
  systematic_review:    1 paper

Journal quartiles:
  Q1:      10 papers
  Q2:       3 papers
  unknown: 14 papers

Data availability:
  not_stated: 26 papers
  open:        1 paper

Open access: 5 papers
```

### Actual Log Output (Layer 2)

```
22:07:20 | INFO    | Starting Layer 2 — NLP Processing Pipeline
22:07:20 | INFO    | Loaded 27 papers from Layer 1
22:08:43 | INFO    | [NER] BioBERT model loaded
22:08:44 | SUCCESS | NLP pipeline ready
22:08:44 | INFO    | Starting NLP processing for 27 papers
22:11:04 | WARNING | [Unpaywall] Failed for DOI 10.3389/fmicb.2026.1856738 (paywalled)
22:16:22 | WARNING | [Unpaywall] Failed for DOI 10.38124/ijisrt/26may280 (paywalled)
22:16:41 | WARNING | [Unpaywall] Failed for DOI 10.1101/2023.08.09.23293887 (paywalled)
22:16:45 | SUCCESS | NLP complete: 27 enriched, 0 errors
22:16:45 | SUCCESS | Saved → data/processed/enriched_20260608_164645.json
22:16:45 | INFO    | Layer 2 complete. 27 enriched records ready for Layer 3.
```

---
---

# Layer 3 — Knowledge Graph Pipeline

## How It Works

### 1. Input

Reads the latest `enriched_YYYYMMDD.json` from Layer 2. In this run: **27 enriched papers**.

---

### 2. Semantic Relationship Extraction

Three fixed relationship types extracted per paper using regex + NER patterns:

| Relationship Type | Meaning | Example |
|---|---|---|
| `REPORTS_ASSOCIATION` | Paper reports a taxon–disease association | Paper → Faecalibacterium prausnitzii (decreased in IBD) |
| `REPORTS_INTERVENTION_EFFECT` | Paper reports an intervention affecting a taxon | Paper → Lactobacillus (increased by probiotic treatment) |
| `USES_METHODOLOGY` | Paper uses a specific sequencing method | Paper → 16S rRNA sequencing |

Plus **open-world triples** extracted by Ollama LLM:
- Free-form `(subject, predicate, object)` triples
- Not limited to 3 fixed types — any relationship found in text
- Example: `(butyrate, inhibits, NF-kB signaling)`
- Example: `(Akkermansia muciniphila, associated with, reduced obesity)`

---

### 3. Evidence Reification (Claim Aggregation)

When multiple papers report the same relationship, they are merged into a **ScientificClaim** node:

- Aggregates confidence scores across all papers
- Tracks consensus direction (increased / decreased / no_change)
- Records supporting and contradicting papers separately
- Stores total combined sample size across all studies
- Computes effect direction consistency percentage

This answers questions like: *"How many papers agree that Faecalibacterium prausnitzii is decreased in IBD?"*

---

### 4. Entity Normalization

Each entity in a relationship is grounded to a canonical ontology ID:
- Taxa → NCBI Taxonomy ID
- Diseases → MeSH ID
- Methods → OBI/EFO ontology ID

This ensures "gut microbiome", "gut microbiota", and "intestinal microbiota" all resolve to the same node in the graph.

---

### 5. Neo4j Knowledge Graph Loading

All edges, claims, and open-world triples loaded in batches into Neo4j:
- Paper nodes created with: title, DOI, year, article type, data availability
- Entity nodes created with: canonical name, ontology ID, grounding source
- Relationship edges created with: confidence score, evidence sentence, extraction method, timestamp
- Indexes created on key properties for fast querying

---

### 6. Output Files (saved to `data/processed/`)

| File | Contents |
|---|---|
| `enhanced_edges_*.json` | All raw relationships with full provenance |
| `enhanced_claims_*.json` | All reified scientific claims |
| `enhanced_stats_*.json` | Pipeline statistics |
| `entities_*.json` | All unique nodes — papers, taxa, methods — with full names and ontology IDs |
| `relationships_*.json` | All edges with human-readable from/to names, confidence, evidence sentence |

---

### 7. REST API (Query Layer)

After Layer 3, the knowledge graph is queryable via REST API at `http://localhost:8000`:

| Endpoint | Question Answered |
|---|---|
| `/query/cross-study-associations` | Which taxa show consistent association with a disease across multiple RCT studies? |
| `/query/intervention-evidence` | What interventions have RCT-level evidence for modifying specific taxa? |
| `/query/methodology-landscape` | Which studies deposited data on SRA/ENA and used which sequencing methods? |
| `/query/top-associations` | Top N taxa associated with a disease ranked by evidence quality |
| `/query/conflicting-evidence` | Which taxa show conflicting associations (increased vs decreased) for a disease? |

---

## Example — 27 Papers (Actual Run)

### Processing

```
Loaded:    27 enriched papers from Layer 2
Workers:   8 parallel workers
Duration:  302 seconds (~5 minutes)
LLM:       1 paper timed out → skipped gracefully, rest processed normally
```

### Relationships Extracted

```
Total relationships extracted:  103
  ├── Associations (REPORTS_ASSOCIATION):              54
  ├── Interventions (REPORTS_INTERVENTION_EFFECT):      5
  ├── Methodologies (USES_METHODOLOGY):                44
  └── Open-world triples (LLM-extracted):              71

Reified claims created:         77
  (evidence aggregated across multiple papers)
```

### Graph Structure in Neo4j

```
Node types:
  Paper           → 27 nodes  (title, DOI, year, article_type)
  Taxon           → ~40 nodes (canonical name, NCBI taxonomy ID)
  Method          → ~15 nodes (16S rRNA, shotgun metagenomics, etc.)
  ScientificClaim → 77 nodes  (aggregated evidence with consensus metrics)
  + open-world types: Disease, Gene, Metabolite, Pathway, etc.

Relationship types in graph:
  REPORTS_ASSOCIATION          Paper → Taxon
  REPORTS_INTERVENTION_EFFECT  Paper → Taxon
  USES_METHODOLOGY             Paper → Method
  SUPPORTED_BY                 ScientificClaim → Paper
  CONTRADICTED_BY              ScientificClaim → Paper
  + custom LLM-extracted types (e.g. INHIBITS, PRODUCES, ASSOCIATED_WITH)
```

### Entities File Sample (`entities_*.json`)

```json
{
  "id": "doi:10.3389/fcimb.2026.1811786",
  "type": "Paper",
  "name": "Gut microbiota and the kidney-gut-skin axis in chronic kidney disease",
  "year": 2026,
  "article_type": "narrative_review",
  "data_availability": "not_stated"
},
{
  "id": "ncbi:853",
  "type": "Taxon",
  "name": "Faecalibacterium prausnitzii",
  "ontology": "NCBI Taxonomy",
  "ontology_id": "ncbi:853",
  "grounded": true
}
```

### Relationships File Sample (`relationships_*.json`)

```json
{
  "from_id": "doi:10.3389/fcimb.2026.1811786",
  "from_name": "Gut microbiota and the kidney-gut-skin axis in chronic kidney disease",
  "relationship_type": "REPORTS_ASSOCIATION",
  "to_id": "ncbi:749906",
  "to_name": "gut metagenome",
  "confidence": 0.5,
  "evidence_strength": "weak",
  "source_sentence": "The kidney-gut-skin axis has attracted attention...",
  "extraction_method": "regex_ner",
  "year": 2026
}
```

### Actual Log Output (Layer 3)

```
22:17:01 | INFO    | Starting Layer 3 — Enhanced Knowledge Graph Pipeline
22:17:01 | INFO    | Loaded 27 enriched papers from Layer 2
22:17:01 | INFO    | [LLMTripleExtractor] Ollama client ready | model=qwen2.5:1.5b
22:22:03 | WARNING | [LLMTripleExtractor] Extraction failed for 10.1126/sciadv.aea0302: timeout
22:22:07 | SUCCESS | Layer 3 complete.
           Extracted 103 relationships, created 77 reified claims.
           Processing time: 302.45s

Pipeline Statistics:
  Total relationships:  103
  Associations:          54
  Interventions:          5
  Methodologies:         44
  Open-world triples:    71
  Reified claims:        77
  Processing time:      302s

Results loaded into Neo4j database: neo4j_enhanced
```

---

## Complete Pipeline Summary

| Layer | Input | Output | Duration |
|---|---|---|---|
| Layer 1 (Collection) | 4 API sources | 27 papers collected | ~2 minutes |
| Layer 2 (NLP) | 27 papers | 27 enriched records, ~62 entities/paper | ~8 minutes |
| Layer 3 (Graph) | 27 enriched records | 103 relationships, 77 claims in Neo4j | ~5 minutes |
| **Total** | | **Knowledge Graph ready** | **~15 minutes** |

## Viewing the Knowledge Graph

**Neo4j Browser:** `http://localhost:7474`
- Username: `neo4j`
- Password: `microbiome2024`

```cypher
-- See full graph
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100

-- Count all node types
MATCH (n) RETURN labels(n) AS type, count(n) AS count ORDER BY count DESC

-- See disease-microbiome associations
MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)
RETURN p.title, r.confidence, t.name LIMIT 20
```

**REST API:** `http://localhost:8000/docs`
- Interactive Swagger UI with all 5 research query endpoints

---

*Generated from actual pipeline run — 27 papers, June 2026*
