# Layer 3: Enhanced Knowledge Graph — Complete Documentation

## Table of Contents

1. [What is Layer 3?](#1-what-is-layer-3)
2. [Why Do We Need a Knowledge Graph?](#2-why-do-we-need-a-knowledge-graph)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Flow — Step by Step](#4-data-flow--step-by-step)
5. [Component 1: Semantic Relationship Extractor](#5-component-1-semantic-relationship-extractor)
6. [Component 2: Provenance Encoder](#6-component-2-provenance-encoder)
7. [Component 3: Entity Normalizer](#7-component-3-entity-normalizer)
8. [Component 4: Enhanced Graph Builder](#8-component-4-enhanced-graph-builder)
9. [Component 5: Relationship Reifier](#9-component-5-relationship-reifier)
10. [Component 6: Evidence Strength Classifier](#10-component-6-evidence-strength-classifier)
11. [Component 7: Triple Promoter (Open-World Pipeline)](#11-component-7-triple-promoter-open-world-pipeline)
12. [Component 8: LLM Triple Extractor](#12-component-8-llm-triple-extractor)
13. [Component 9: Predicate Registry](#13-component-9-predicate-registry)
14. [Component 10: Enhanced Neo4j Loader](#14-component-10-enhanced-neo4j-loader)
15. [Component 11: Enhanced KG Pipeline (Orchestrator)](#15-component-11-enhanced-kg-pipeline-orchestrator)
16. [Component 12: Research Query Engine](#16-component-12-research-query-engine)
17. [What Gets Stored in Neo4j](#17-what-gets-stored-in-neo4j)
18. [Neo4j Indexes](#18-neo4j-indexes)
19. [Configuration Reference](#19-configuration-reference)
20. [Bottlenecks and Known Limitations](#20-bottlenecks-and-known-limitations)
21. [Running Layer 3](#21-running-layer-3)

---

## 1. What is Layer 3?

Layer 3 is the **Enhanced Knowledge Graph** layer — the brain of the entire system. It takes the enriched papers produced by Layer 2 (which already have extracted entities, classified article types, parsed sections) and transforms them into a **queryable, semantically rich graph database** stored in Neo4j.

Think of it this way:

- **Layer 1** collects raw papers from PubMed, Europe PMC, etc.
- **Layer 2** reads each paper and labels what's in it (taxa, diseases, methods)
- **Layer 3** reads those labels and builds a *network of scientific facts* that you can query

The output of Layer 3 is not a spreadsheet or a list. It's a graph — a web of nodes (papers, bacteria, diseases, methods) connected by arrows (relationships) that carry rich scientific meaning: "this paper *reports* that *Prevotella copri* is **increased** in patients with *Type 2 Diabetes* with **85% confidence**, from the **results section**, p < 0.01."

---

## 2. Why Do We Need a Knowledge Graph?

### The problem with flat data

Imagine you have 1,000 papers, each mentioning dozens of bacteria and diseases. If you stored this as a spreadsheet:

| Paper | Taxon | Disease | Direction |
|-------|-------|---------|-----------|
| Paper_A | Bacteroides fragilis | T2D | increased |
| Paper_B | Bacteroides fragilis | T2D | increased |
| Paper_C | F. prausnitzii | IBD | decreased |

You could answer simple questions, but not complex ones like:
- *"Which bacteria are consistently increased in T2D across 5+ RCT studies?"*
- *"Do any bacteria show conflicting evidence — some papers say increased, others say decreased?"*
- *"What interventions have strong evidence for modifying gut bacteria?"*

For these questions you need to **join**, **aggregate**, and **reason across** thousands of relationships simultaneously. A graph database does this natively.

### The graph approach

In Neo4j, instead of rows and columns, you have:
- **Nodes**: circles representing things (Paper, Taxon, Disease, Method, ScientificClaim)
- **Relationships**: arrows connecting nodes, carrying properties
- **Properties**: data attached to nodes and arrows (confidence, p-value, direction, etc.)

You can then ask: *"Starting from 'Type 2 Diabetes', follow all REPORTS_ASSOCIATION arrows backwards, find the bacteria at the other end, count how many papers point to each bacterium, and return those with high confidence."* This is a single graph query.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 3 ARCHITECTURE                             │
│                                                                         │
│  INPUT: 100 enriched papers from Layer 2                                │
│         (entities, sections, article types already extracted)           │
│                          │                                              │
│                          ▼                                              │
│          ┌───────────────────────────────┐                              │
│          │    EnhancedKGPipeline         │  ← orchestrates everything   │
│          │    (enhanced_kg_pipeline.py)  │                              │
│          └───────────────┬───────────────┘                              │
│                          │                                              │
│          ┌───────────────▼───────────────┐                              │
│          │  Cache Pre-warming            │  ← fills SQLite before       │
│          │  (entity_normalizer)          │    parallel workers start    │
│          └───────────────┬───────────────┘                              │
│                          │                                              │
│          ┌───────────────▼───────────────┐                              │
│          │  ThreadPoolExecutor           │  ← 8 workers in PARALLEL     │
│          │  (8 batch workers)            │                              │
│          └──┬──┬──┬──┬──┬──┬──┬──┬──────┘                              │
│             │  │  │  │  │  │  │  │                                      │
│   Each worker runs EnhancedGraphBuilder for its 100-paper batch:        │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  Per Paper:                                                  │      │
│   │  1. SemanticRelationshipExtractor  ← regex patterns          │      │
│   │     → associations, interventions, methodology               │      │
│   │  2. EntityNormalizer               ← NCBI/OLS/cache          │      │
│   │     → canonical ontology IDs                                 │      │
│   │  3. ProvenanceEncoder              ← attached to every edge  │      │
│   │     → source sentence, section, confidence, timestamp        │      │
│   │  4. LLMTripleExtractor (optional)  ← USE_LLM_LAYER3=true     │      │
│   │     → open-world (subject, predicate, object)                │      │
│   │  5. TriplePromoter (optional)      ← enriches LLM triples    │      │
│   │     → provenance + grounding + evidence strength             │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                          │                                              │
│          ┌───────────────▼───────────────┐                              │
│          │  RelationshipReifier          │  ← SERIAL after batches      │
│          │  (aggregate across papers)    │  ← creates ScientificClaims  │
│          └───────────────┬───────────────┘                              │
│                          │                                              │
│          ┌───────────────▼───────────────┐                              │
│          │  EnhancedNeo4jLoader          │  ← SERIAL batch loading      │
│          │  (writes to Neo4j)            │  ← 10,000 ops per txn        │
│          └───────────────┬───────────────┘                              │
│                          │                                              │
│  OUTPUT: Neo4j database with:                                           │
│          - Paper nodes, Taxon nodes, Disease nodes, Method nodes        │
│          - REPORTS_ASSOCIATION relationships (892 in current run)       │
│          - USES_METHODOLOGY relationships (120 in current run)          │
│          - ScientificClaim nodes (194 in current run)                   │
│          - SUPPORTED_BY links from claims to papers                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow — Step by Step

Here is exactly what happens when you run `RUN_LAYER=3 python3 main.py`:

**Step 1: Load papers**
Layer 3 reads the latest `enriched_batch_*.json` file produced by Layer 2. Each paper already has: extracted taxa, diseases, methods, sections (abstract/methods/results), article type, data availability status.

**Step 2: Cache pre-warming**
Before any parallel work starts, every unique entity (bacteria name, disease name, method name) from all papers is sent to the EntityNormalizer. This fills the SQLite grounding cache so batch workers never need to wait for NCBI API calls.

**Step 3: Parallel batch processing**
Papers are split into batches of 100. Up to 8 batches run simultaneously, each in a separate thread, each with its own EnhancedGraphBuilder. For each paper, the builder runs three extractors and (optionally) the LLM extractor.

**Step 4: Claim merging**
After all batches finish, relationships are grouped by their (subject, predicate, object) triplet. Multiple papers reporting the same finding are merged into a single ScientificClaim with consensus metrics.

**Step 5: Neo4j loading**
All edges, claims, and open-world triples are batch-loaded into Neo4j in transactions of 10,000 operations each.

**Step 6: Save intermediate JSON**
All edges, claims, and statistics are also saved to `data/processed/enhanced_edges_*.json` for debugging and backup.

---

## 5. Component 1: Semantic Relationship Extractor

**File:** `graph/semantic_extractor.py`
**Class:** `SemanticRelationshipExtractor`

### What it does

This is the core extraction engine. It takes one enriched paper and finds all the scientific relationships in its text using **regex pattern matching** — no AI required.

It extracts three types of relationships:

---

### Type 1: Taxon-Disease Associations (REPORTS_ASSOCIATION)

**The question it answers:** *"In this paper, is bacteria X associated with disease Y, and if so, in what direction?"*

**How it works:**
1. Takes the paper's results, abstract, and discussion sections
2. Splits text into individual sentences
3. For each sentence: checks if any known taxon AND any known disease appear in it
4. If both appear, tries to determine the *direction* of the relationship

**Direction detection (priority order):**

| Direction | Example phrases detected |
|-----------|--------------------------|
| `increased` | "elevated", "enriched", "higher", "upregulated", "more abundant", "overrepresented", "colonized", "detected in", "prevalent" |
| `decreased` | "reduced", "depleted", "lower", "downregulated", "less abundant", "underrepresented", "absent", "suppressed", "eliminated" |
| `no_change` | "unchanged", "no significant difference", "similar", "no significant change" |
| `associated` | "implicated in", "role in", "biomarker", "pathobiont", "dysbiosis", "protective", "linked to", "contributor to", "correlated" |

If no direction pattern matches → the sentence is **silently discarded**. No guessing.

**Additional information extracted per association:**
- **P-value**: parsed from "p = 0.001", "p < 0.05" patterns
- **Effect size**: parsed from "fold change = 2.5", "LDA score = 3.2"
- **Statistical measure**: "LDA score", "fold change", "relative abundance", "odds ratio"
- **Comparison context**: "Type 2 Diabetes vs healthy", "pre vs post"

**Confidence scoring:**
```
direction found         → +0.5  (minimum to pass the 0.5 threshold)
p-value found           → +0.3
effect size found       → +0.15
statistical measure     → +0.05
─────────────────────────────
maximum possible        = 1.0
```
Relationships with confidence < 0.5 are discarded.

---

### Type 2: Intervention Effects (REPORTS_INTERVENTION_EFFECT)

**The question it answers:** *"In this RCT paper, did intervention X change bacteria Y?"*

**How it works:**
1. Only runs on `original_research`, `meta_analysis`, `systematic_review` papers
2. Scans the methods section for intervention keywords
3. Scans results for sentences mentioning both an intervention and a taxon
4. Only keeps results with p < 0.05 OR an explicit significance statement

**Intervention types detected:** probiotic, FMT (fecal microbiota transplant), diet, antibiotic, prebiotic, synbiotic

**Additional information extracted:** duration ("4 weeks"), dosage ("10^9 CFU"), sample size (n=50)

---

### Type 3: Methodology Usage (USES_METHODOLOGY)

**The question it answers:** *"What sequencing method did this paper use, and with what sample size?"*

**How it works:**
1. Reads the methods section
2. For each method name already extracted by Layer 2, confirms it appears in the methods text
3. Extracts: sequencing platform (Illumina, PacBio), sample size, data availability status

**Fixed confidence:** 0.8 (methodology is either mentioned or not — no ambiguity)

---

## 6. Component 2: Provenance Encoder

**File:** `graph/provenance.py`
**Class:** `ProvenanceMetadata`

### What it does

Every single relationship extracted — every arrow in the graph — carries a **complete audit trail** so you can always trace *why* an edge exists.

### Why this matters

In scientific research, you need to know: *"Where did this claim come from? Which sentence? Which section? How confident is the extractor? When was it extracted?"* Without this, you can't validate or trust the graph.

### What gets recorded for every edge

| Field | What it stores | Example |
|-------|---------------|---------|
| `paper_id` | DOI or PMID of source paper | `doi:10.3389/fcimb.2024.1394873` |
| `section_type` | Which section it came from | `results` |
| `source_sentence` | The exact sentence | *"Prevotella copri was significantly increased..."* |
| `sentence_offset` | Character position in section | `847` |
| `extraction_method` | Which extractor was used | `regex_ner` |
| `extraction_timestamp` | When extraction happened (UTC) | `2025-07-14T03:12:02Z` |
| `extractor_version` | Version of the code | `1.0` |
| `confidence_score` | How confident the extractor is | `0.85` |
| `validation_status` | Has a human verified it? | `unvalidated` |
| `surrounding_context` | ±2 sentences around the evidence | *"...previous studies showed..."* |

### Validation rules

The ProvenanceMetadata model rejects records that:
- Have an empty source sentence
- Have confidence outside [0.0, 1.0]
- Use an unregistered extraction method
- Have a timestamp more than 1 minute in the future

---

## 7. Component 3: Entity Normalizer

**File:** `graph/entity_normalizer.py`
**Class:** `EntityNormalizer`

### The problem it solves

Different papers write the same bacterium differently:
- "Bacteroides fragilis"
- "B. fragilis"
- "ATCC 25285"

Without normalization, these would create 3 separate graph nodes instead of 1. When you query "show me all papers about Bacteroides fragilis", you'd miss the other two.

### How it works — the decision tree

For every entity name, the normalizer goes through these steps in order:

```
1. Empty or ≤2 characters?  → return ungrounded immediately (no API call)
   (catches noise like "dr", "co" from NER mistakes)

2. Already in SQLite cache? → return cached result (no API call)
   (every successful lookup is remembered forever)

3. Known abbreviation?      → check YAML maps (e.g., "B. fragilis" → "Bacteroides fragilis")

4. Route to authoritative API based on entity type:
   taxon     → NCBI Taxonomy  (ncbi:816 for Bacteroides fragilis)
   disease   → NCBI MeSH      (mesh:D003015 for Crohn's Disease)
   gene      → NCBI Gene
   protein   → UniProt REST
   metabolite → EMBL-EBI OLS (ChEBI ontology)
   pathway   → OLS (GO, PW)
   method    → OLS (OBI, EFO)
   unknown   → OLS cross-search (all ontologies)

5. OLS cross-search fallback (if primary API failed)

6. LLM fallback (temperature=0, deterministic) — marks result as grounded=False

7. All failed → return ungrounded:{text} ID + log to failure DB
```

### Confidence levels

| Confidence | Meaning |
|-----------|---------|
| 1.0 | Exact match in authoritative ontology |
| 0.8 | Fuzzy match |
| 0.6 | LLM suggestion (not authoritative) |
| 0.0 | Failed all attempts |

### The SQLite cache

The cache (`grounding_cache.db`) means:
- The first time "Bacteroides fragilis" is normalized: NCBI API call, ~0.5 seconds
- Every subsequent time: SQLite lookup, ~0.001 seconds

This is why cache pre-warming (Step 2 in the pipeline) is critical — it fills the cache before 8 workers all compete for the same NCBI endpoints.

---

## 8. Component 4: Enhanced Graph Builder

**File:** `graph/enhanced_graph_builder.py`
**Class:** `EnhancedGraphBuilder`

### What it does

The EnhancedGraphBuilder is the **per-batch coordinator**. Each parallel worker gets one instance of this class to process its batch of 100 papers. It wires together the extractor, normalizer, LLM extractor, and triple promoter into a single paper-processing loop.

### Processing flow for one paper

```
process_paper(paper)
  │
  ├── 1. SemanticRelationshipExtractor.extract_associations(paper)
  │       → list of SemanticRelationship objects (REPORTS_ASSOCIATION)
  │
  ├── 2. SemanticRelationshipExtractor.extract_intervention_effects(paper)
  │       → list of SemanticRelationship objects (REPORTS_INTERVENTION_EFFECT)
  │
  ├── 3. SemanticRelationshipExtractor.extract_methodology_usage(paper)
  │       → list of SemanticRelationship objects (USES_METHODOLOGY)
  │
  │   For each relationship above:
  │   ├── EntityNormalizer.normalize(target_entity, entity_type)
  │   │     → canonical ontology ID (or checks Layer 2 pre-grounded cache first)
  │   └── Build EnhancedGraphEdge with embedded provenance
  │
  └── 4. LLMTripleExtractor.extract_triples(section_text)  [if USE_LLM_LAYER3=true]
          → raw (subject, predicate, object) dicts
          └── TriplePromoter.promote_batch(triples, paper_metadata)
                → PromotedTriple objects with provenance + grounding + evidence strength
```

### EnhancedGraphEdge

The final representation of each relationship before it goes to Neo4j. It combines:
- `source` (paper DOI/PMID)
- `target` (canonical ontology ID of the entity)
- `relation` (REPORTS_ASSOCIATION, REPORTS_INTERVENTION_EFFECT, USES_METHODOLOGY)
- All semantic properties (direction, p-value, confidence, etc.)
- All provenance fields (source sentence, section, timestamp, etc.)
- Paper metadata (year, article_type, data_availability, accession_numbers)

### Relationship index

As papers are processed, the builder maintains a dictionary keyed by `(source_entity, predicate, canonical_target_id)`. This index is used later by the RelationshipReifier to group evidence from multiple papers for the same claim.

---

## 9. Component 5: Relationship Reifier

**File:** `graph/relationship_reifier.py`
**Class:** `RelationshipReifier`

### The core insight: from edges to claims

After processing all 100 papers in a batch, you might have extracted:
- Paper A says: "Prevotella copri is increased in T2D" (confidence 0.85)
- Paper B says: "Prevotella copri is increased in T2D" (confidence 0.80)
- Paper C says: "Prevotella copri is increased in T2D" (confidence 0.75)

You now have 3 separate edges in Neo4j. But scientifically, these represent **one claim** supported by **3 independent studies**. The RelationshipReifier converts those 3 edges into one **ScientificClaim node** that says: *"This claim has 3 supporting papers, consensus confidence 0.80, direction consistent across all papers."*

This is called **reification** — making a relationship into a first-class entity that can itself have properties and relationships.

### What a ScientificClaim contains

```
ScientificClaim {
  claim_id:                   "uuid-1234..."
  claim_type:                 "association"
  subject_entity:             "doi:10.3389/fcimb.2024.1394873"
  predicate:                  "associated_with_increased"
  object_entity:              "prevotella copri"
  supporting_papers:          ["doi:paper_a", "doi:paper_b", "doi:paper_c"]
  contradicting_papers:       []
  consensus_confidence:       0.80   ← arithmetic mean of all confidence scores
  effect_direction_consistency: 1.0  ← all 3 papers agree
  evidence_strength:          "moderate"
  supporting_paper_count:     3
  first_reported:             "2024-01-15T..."
  last_updated:               "2024-03-20T..."
}
```

### Evidence strength classification

| Strength | Criteria |
|----------|---------|
| `strong` | p < 0.01 AND original_research or meta_analysis |
| `moderate` | p < 0.05 |
| `weak` | p ≥ 0.05 or no p-value |
| `conflicting` | Both supporting AND contradicting evidence exists |

### Conflict detection

The reifier also detects when two claims about the same entities have **opposite predicates**:
- "associated_with_increased" vs "associated_with_decreased" → **conflicting**
- "associated_with_increased" vs "associated_with_associated" → **NOT conflicting** (non-directional "associated" never conflicts)

---

## 10. Component 6: Evidence Strength Classifier

**File:** `graph/evidence_strength_classifier.py`
**Class:** `EvidenceStrengthClassifier`

### What it does

This component is specifically for **LLM-extracted triples** (open-world relationships). Unlike regex-extracted relationships which have p-values and article types, LLM triples only have a confidence score and a section type. The classifier uses these to assign evidence strength.

### Classification rules

**For a single triple:**

```
IF confidence < 0.7:
    → "weak"  (regardless of section)

IF section is abstract or introduction:
    → "weak"  (even with high confidence)

IF section is results or discussion:
    IF confidence ≥ 0.85 AND article_type is original_research/meta_analysis:
        → "strong"
    IF confidence ≥ 0.7:
        → "moderate"

Otherwise:
    → "weak"
```

**For an aggregated OpenWorldClaim:**

```
IF ≥ 3 papers with individual strength "moderate" or "strong":
    → "strong"  (promoted by cross-paper consensus)
ELSE:
    → strongest among all individual strengths
```

### Why this exists

Before this component was wired in, all LLM-extracted relationships always landed as "weak" regardless of how confident or well-sourced they were. Now a high-confidence triple from the results section of an original research paper correctly gets classified as "strong" or "moderate".

---

## 11. Component 7: Triple Promoter (Open-World Pipeline)

**File:** `graph/triple_promoter.py`
**Class:** `TriplePromoter`

### What problem it solves

The regex extractor only captures 3 types of relationships. But scientific papers contain hundreds of relationship types:
- "Akkermansia muciniphila *produces* butyrate"
- "F. prausnitzii *inhibits* NF-κB signaling"
- "NOD2 mutations *predispose* to Crohn's disease"
- "SCFA *activates* regulatory T cells"

These are all scientifically important but the regex system can't capture them. The LLM extractor (Component 8) extracts these as raw text, and the TriplePromoter **enriches them** into first-class relationships.

### What it does to each raw LLM triple

```
Raw LLM triple:
{
  "subject": "Akkermansia muciniphila",
  "subject_type": "taxon",
  "predicate": "produces",
  "object": "butyrate",
  "object_type": "metabolite",
  "confidence": 0.9,
  "evidence": "Akkermansia muciniphila produces butyrate which activates...",
  "paper_id": "doi:10.xxxx",
  "section_type": "results"
}

After TriplePromoter:
PromotedTriple {
  subject_id:          "ncbi:239935"  ← grounded via NCBI Taxonomy
  subject_name:        "Akkermansia muciniphila"
  subject_grounded:    True
  object_id:           "chebi:17968"  ← grounded via ChEBI
  object_name:         "butyric acid"
  object_grounded:     True
  canonical_predicate: "PRODUCES"    ← normalized via PredicateRegistry
  evidence_strength:   "strong"      ← from results + original_research + 0.9
  provenance:          ProvenanceMetadata(...)  ← full audit trail
}
```

### Claim aggregation

After all papers are processed, the TriplePromoter groups PromotedTriples by `(subject_id, canonical_predicate, object_id)`. If 2+ distinct papers report the same triple, an `OpenWorldClaim` is created — the open-world equivalent of a `ScientificClaim`.

---

## 12. Component 8: LLM Triple Extractor

**File:** `graph/llm_triple_extractor.py`
**Class:** `LLMTripleExtractor`

### What it does

When `USE_LLM_LAYER3=true`, this component sends each paper section to an Ollama LLM (local AI model) and asks it to extract all scientific relationships expressed in the text as (subject, predicate, object) triples.

### The prompt

The model receives a prompt like:
```
Extract biomedical (subject, predicate, object) triples from the TEXT below.
Rules:
- Only extract relationships explicitly stated in the text
- Subject and object must be specific named entities
- Confidence: 0.9=stated with stats, 0.7=clearly stated, 0.5=implied

Output format: {"triples": [{"subject": "...", "predicate": "...", ...}]}

TEXT: [section content up to 2500 chars]
```

### Section priority order

Sections are processed in this order (most scientifically valuable first):
1. results
2. discussion
3. abstract
4. introduction
5. other

### Caching

Results are cached in `graph/triple_cache.json` keyed by a hash of `(paper_id, section_type, text[:500])`. Once a section has been processed, it's never sent to the LLM again, even across pipeline runs.

### Current limitation

The configured model (`qwen2.5:0.5b`) is a 500M parameter model running on CPU. It's very small and slow — each section takes 2–4 minutes on CPU hardware. This is why `USE_LLM_LAYER3=false` is currently set. A larger model (`qwen2.5:7b` or `llama3.1:8b`) would produce much better quality extractions.

---

## 13. Component 9: Predicate Registry

**File:** `graph/predicate_registry.py`
**Class:** `PredicateRegistry`

### What it does

When the LLM extracts a relationship, it might say "produces", "synthesizes", "generates" — all meaning the same scientific thing. The PredicateRegistry normalizes these raw predicate strings to canonical uppercase forms and tracks which novel predicates appear frequently enough to become first-class relationship types.

### Normalization map (sample)

| Raw predicate | Canonical form | Category |
|--------------|---------------|---------|
| "produces", "synthesizes", "generates" | `PRODUCES` | biosynthetic |
| "inhibits", "suppresses" | `INHIBITS` | regulatory |
| "associated with", "linked to" | `ASSOCIATED_WITH` | associative |
| "improves" | `IMPROVES` | clinical |
| "causes", "induces" | `CAUSES` | causal |
| "biomarker for" | `BIOMARKER_FOR` | biomarker |
| anything unknown | `RELATES_TO` | generic |

### Novel predicate promotion

If the LLM keeps extracting a predicate that isn't in the normalization map, it gets tracked in `predicate_registry.db` (SQLite). When it appears in 10+ distinct papers (configurable via `PREDICATE_PROMOTION_THRESHOLD`), it's automatically **promoted** — added to the normalization map and given a canonical form.

**Quality gates (prevents garbage from being promoted):**
- Predicate longer than 60 characters → rejected (it's a sentence fragment, not a predicate)
- Pure stop-words ("are", "is", "was") → rejected
- Metadata markers ("author", "doi", "title", "journal") → rejected

This prevents the small LLM from promoting nonsense like `"AIMED_TO_LONGITUDINALLY_ASSESS_GUT_MICROBIOTA_DEVELOPMENT"` as a relationship type.

---

## 14. Component 10: Enhanced Neo4j Loader

**File:** `graph/enhanced_kg_pipeline.py` (class `EnhancedNeo4jLoader`)

### What it does

Takes all extracted data — edges, claims, open-world triples, promoted triples, open-world claims — and writes them to Neo4j in efficient batches.

### Loading methods

| Method | What it loads | Batch size |
|--------|-------------|-----------|
| `load_edges()` | REPORTS_ASSOCIATION, REPORTS_INTERVENTION_EFFECT, USES_METHODOLOGY relationships | 10,000 per transaction |
| `load_claims()` | ScientificClaim nodes + SUPPORTED_BY/CONTRADICTED_BY links | 10,000 per transaction |
| `load_open_world_triples()` | Raw LLM triples as RELATES_TO relationships | 10,000 per transaction |
| `load_promoted_triples()` | PromotedTriple objects with full provenance | 10,000 per transaction |
| `load_open_world_claims()` | OpenWorldClaim nodes + SUPPORTED_BY_TRIPLE links | 10,000 per transaction |

### Cypher pattern for edge loading

For each edge, three operations are executed:

```cypher
-- 1. Create or update Paper node
MERGE (source:Paper {id: $doi})
ON CREATE SET source.title = $title, source.year = $year, ...
ON MATCH SET source.title = $title, ...

-- 2. Create or update target node (Taxon, Method, etc.)
MERGE (target:Taxon {id: $canonical_ontology_id})
ON CREATE SET target.name = $canonical_name, target.ontology = $ontology, ...

-- 3. Create the relationship with ALL properties embedded
MATCH (source:Paper {id: $source_id})
MATCH (target:Taxon {id: $target_id})
CREATE (source)-[r:REPORTS_ASSOCIATION]->(target)
SET r = {direction: $direction, confidence: $confidence, p_value: $p_value,
         source_sentence: $sentence, extraction_method: $method, ...}
```

---

## 15. Component 11: Enhanced KG Pipeline (Orchestrator)

**File:** `graph/enhanced_kg_pipeline.py`
**Class:** `EnhancedKGPipeline`

### What it does

This is the top-level coordinator that wires everything together and manages the parallel execution strategy.

### Initialization sequence

```python
EnhancedKGPipeline.__init__():
  1. Load PipelineConfig from environment variables
  2. Connect to Neo4j (bolt://localhost:7688)
  3. Create Neo4j indexes
  4. Instantiate TriplePromoter with:
     - EntityNormalizer (shared across all workers)
     - PredicateRegistry
     - EvidenceStrengthClassifier
     - promotion_threshold from PREDICATE_PROMOTION_THRESHOLD env var
```

### run() sequence

```python
EnhancedKGPipeline.run(enriched_papers):
  1. Convert dicts to EnrichedPaperRecord objects
  2. Pre-warm entity cache (_prewarm_entity_cache)
  3. Split into batches of ENHANCED_BATCH_SIZE (default: 100)
  4. ThreadPoolExecutor(max_workers=ENHANCED_NUM_WORKERS) — default 8
     → _process_batch(batch, idx) for each batch in parallel
  5. Merge all builders (collect relationships from all workers)
  6. create_reified_claims() — serial, aggregates across all papers
  7. Collect statistics
  8. Build title_lookup (DOI → paper title mapping)
  9. _save_results() — write JSON files
  10. load_edges(), load_claims(), load_open_world_triples(),
      load_promoted_triples(), load_open_world_claims()
  11. Return stats dict
```

### PipelineConfig (environment variables)

| Env var | Default | Description |
|---------|---------|-------------|
| `NEO4J_ENHANCED_URI` | `bolt://localhost:7687` | Neo4j connection (use 7688 in your setup) |
| `NEO4J_ENHANCED_USER` | `neo4j` | Neo4j username |
| `NEO4J_ENHANCED_PASSWORD` | `password` | Neo4j password |
| `NEO4J_ENHANCED_DATABASE` | `neo4j` | Database name |
| `ENHANCED_BATCH_SIZE` | `100` | Papers per batch |
| `ENHANCED_NUM_WORKERS` | `8` | Parallel workers |
| `ENHANCED_PIPELINE_ENABLED` | `true` | Enable/disable |
| `PREDICATE_PROMOTION_THRESHOLD` | `10` | Papers before promoting a predicate |
| `USE_LLM_LAYER3` | `false` | Enable LLM triple extraction |

---

## 16. Component 12: Research Query Engine

**File:** `graph/research_query_engine.py`
**Class:** `ResearchQueryEngine`

### What it does

Provides 5 pre-built scientific queries against the knowledge graph. All queries use parameterized Cypher (no string concatenation) to prevent injection attacks, and results are cached for 24 hours.

### The 5 queries

**Query 1: Cross-study associations**
*"Which gut microbiome taxa show consistent association with [disease] across [study_type] studies with open sequencing data?"*

Returns: taxon names with paper counts, consensus confidence, consensus direction, direction breakdown (increased/decreased/no_change/associated counts).

**Query 2: Intervention effectiveness**
*"What interventions (probiotics, FMT, diet) have RCT-level evidence for modifying specific gut taxa?"*

Returns: intervention type, affected taxon, effect direction, paper count, total sample size.

**Query 3: Methodology landscape**
*"Which microbiome studies from year X to Y deposited data on SRA/ENA and used shotgun metagenomics vs 16S?"*

Returns: method, year, paper count, data availability percentage.

**Query 4: Top associations by evidence**
*"Top 10 taxa associated with [disease] ranked by evidence quality."*

Returns: taxa ranked by (paper count, average confidence).

**Query 5: Conflicting evidence**
*"Which taxa show conflicting associations (increased vs decreased) for [disease]?"*

Returns: taxa with both increased and decreased evidence, counts per direction.

### Query caching

Results are cached in a SQLite-backed `QueryCache` with 24-hour TTL. The cache key is the query name + parameter values. Call `engine.invalidate_cache()` after loading new data.

---
