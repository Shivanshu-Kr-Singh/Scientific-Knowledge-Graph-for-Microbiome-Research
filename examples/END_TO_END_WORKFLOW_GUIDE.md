# End-to-End Workflow Guide

## Overview

This guide demonstrates the complete Scientific Knowledge Graph pipeline from data collection through research queries. The workflow showcases how raw scientific papers are transformed into a queryable knowledge graph that answers specific research questions.

**Requirements Validated:** 1.1, 1.2, 1.3

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Quick Start](#quick-start)
3. [Workflow Steps](#workflow-steps)
4. [Research Questions](#research-questions)
5. [Expected Outputs](#expected-outputs)
6. [Interpretation Guide](#interpretation-guide)
7. [Troubleshooting](#troubleshooting)

---

## Pipeline Overview

The Scientific Knowledge Graph pipeline consists of four main stages:

```
┌─────────────────┐
│  1. COLLECTION  │  Fetch papers from PubMed, EuropePMC, etc.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. ENRICHMENT  │  Extract entities, classify articles, parse sections
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. GRAPH BUILD  │  Create semantic relationships with provenance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. QUERIES     │  Execute research questions, aggregate evidence
└─────────────────┘
```

### Key Features

- **Semantic Relationships**: Captures scientific meaning (direction, effect size, p-values)
- **Complete Provenance**: Traces every claim back to source sentence
- **Evidence Aggregation**: Combines findings across multiple papers
- **Research Queries**: Answers 5 core scientific questions

---

## Quick Start

### Prerequisites

1. **Neo4j Database**: Running instance with enhanced schema
2. **Python Environment**: Python 3.8+ with dependencies installed
3. **Data Sources**: Access to PubMed/EuropePMC APIs (optional for demo)

### Installation

```bash
# Clone repository
cd /path/to/project

# Install dependencies
pip install -r requirements.txt

# Verify Neo4j connection
python test_neo4j_connection.py
```

### Running the Workflow

#### Demo Mode (No Database Required)

Shows expected outputs without connecting to Neo4j:

```bash
python examples/end_to_end_workflow.py --mode demo
```

#### Query-Only Mode

Executes research queries on existing graph:

```bash
python examples/end_to_end_workflow.py --mode query-only \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password your_password \
    --neo4j-database neo4j_enhanced
```

#### Full Pipeline Mode

Runs complete pipeline from collection to queries:

```bash
python examples/end_to_end_workflow.py --mode full \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password your_password \
    --neo4j-database neo4j_enhanced
```

---

## Workflow Steps

### Step 1: Data Collection

**Purpose**: Fetch papers from scientific databases

**Input**: Search query (e.g., "microbiome diabetes")

**Output**: List of paper metadata (title, abstract, DOI, year)

**Example**:

```python
papers = workflow.step_1_collection(
    query="microbiome diabetes",
    max_papers=5
)
```

**Sample Output**:

```
✓ Collected 5 papers
  1. [PubMed] Gut microbiome alterations in Type 2 Diabetes...
     PMID: 12345678, Year: 2024
  2. [EuropePMC] Probiotic intervention modulates gut microbiota...
     PMID: 23456789, Year: 2023
```

**Data Sources**:
- PubMed (via Entrez API)
- EuropePMC (via REST API)
- Semantic Scholar (optional)
- bioRxiv (optional)

---

### Step 2: NLP Enrichment

**Purpose**: Extract structured information from unstructured text

**Input**: Raw paper metadata

**Output**: Enriched papers with entities, methods, classifications

**Processing**:

1. **Entity Extraction**: Identify taxa, diseases, treatments
2. **Method Extraction**: Detect sequencing methods, platforms
3. **Article Classification**: Categorize as RCT, review, meta-analysis
4. **Data Availability**: Check for accession numbers (SRA, ENA)

**Example**:

```python
enriched_papers = workflow.step_2_enrichment(papers)
```

**Sample Output**:

```
Processing paper 1/5: Gut microbiome alterations in Type 2 Diabetes...
  ✓ Extracted 4 entities
    - Bacteroides fragilis (taxon)
    - Faecalibacterium prausnitzii (taxon)
    - Type 2 Diabetes (disease)
    - T2D (disease)
  ✓ Identified 1 methods
    - 16S rRNA sequencing
  ✓ Article type: original_research
  ✓ Data availability: open
  ✓ Accessions: PRJNA123456
```

**Entity Types**:
- **Taxon**: Bacterial species/genera (e.g., "Bacteroides fragilis")
- **Disease**: Medical conditions (e.g., "Type 2 Diabetes", "IBD")
- **Treatment**: Interventions (e.g., "probiotic", "FMT")
- **Method**: Sequencing techniques (e.g., "16S rRNA", "shotgun metagenomics")

---

### Step 3: Graph Construction

**Purpose**: Build knowledge graph with semantic relationships

**Input**: Enriched paper records

**Output**: Neo4j graph with nodes and relationships

**Graph Schema**:

```
(Paper)-[:REPORTS_ASSOCIATION {
    direction: "increased",
    comparison: "T2D vs healthy",
    statistical_measure: "LDA score",
    effect_size: 3.2,
    p_value: 0.001,
    section: "results",
    source_sentence: "Bacteroides fragilis showed...",
    confidence: 0.87,
    extraction_method: "llm_v1.2"
}]->(Taxon)
```

**Example**:

```python
graph_stats = workflow.step_3_graph_construction(enriched_papers)
```

**Sample Output**:

```
Processing paper 1/5: Gut microbiome alterations in Type 2 Diabetes...
  ✓ Created 6 nodes
    - 1 Paper node
    - 2 Taxon nodes
    - 2 Disease nodes
    - 1 Method node
  ✓ Created 8 relationships
    - 2 REPORTS_ASSOCIATION
    - 1 USES_METHODOLOGY
    - 5 HAS_ENTITY

Graph Construction Summary:
  Papers processed: 5
  Nodes created: 28
  Relationships created: 42
  Reified claims: 3
```

**Relationship Types**:

1. **REPORTS_ASSOCIATION**: Taxon-disease associations
   - Properties: direction, effect_size, p_value, statistical_measure
   
2. **REPORTS_INTERVENTION_EFFECT**: Intervention effects on taxa
   - Properties: intervention_type, effect_direction, duration, dosage
   
3. **USES_METHODOLOGY**: Paper methodology information
   - Properties: method_name, sequencing_platform, sample_size

**Provenance Tracking**:

Every relationship includes:
- `section`: Where claim was found (abstract, methods, results, discussion)
- `source_sentence`: Exact sentence supporting the claim
- `extraction_method`: Algorithm used (regex_ner, biobert_ner, llm_extractor)
- `extraction_timestamp`: When extraction occurred
- `confidence`: Extraction confidence score (0.0-1.0)

---

### Step 4: Research Queries

**Purpose**: Answer scientific research questions

**Input**: Constructed knowledge graph

**Output**: Query results with aggregated evidence

**Example**:

```python
query_results = workflow.step_4_research_queries()
```

See [Research Questions](#research-questions) section for detailed examples.

---

## Research Questions

The knowledge graph is designed to answer five core research questions:

### Q1: Cross-Study Disease-Microbiome Associations

**Question**: "Which gut microbiome taxa show consistent association with Type 2 Diabetes across RCT studies with open sequencing data?"

**Query**:

```python
result = query_engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    study_type="RCT",
    min_papers=3,
    confidence_threshold=0.7,
    require_open_data=True
)
```

**Expected Output**:

```
Query: Cross-study associations for Type 2 Diabetes
Execution time: 1234.5 ms
Results: 5 taxa

Taxon: Bacteroides fragilis
  Papers: 5
  Consensus confidence: 0.85
  Consensus direction: increased
  Direction consistency: 80.0%
  Increased: 4, Decreased: 1, No change: 0
  Paper IDs: PMID:12345678, DOI:10.1234/test1, PMID:87654321

Taxon: Faecalibacterium prausnitzii
  Papers: 4
  Consensus confidence: 0.82
  Consensus direction: decreased
  Direction consistency: 100.0%
  Increased: 0, Decreased: 4, No change: 0
  Paper IDs: PMID:23456789, PMID:34567890, DOI:10.5678/test2
```

**Interpretation**:
- **Bacteroides fragilis**: Strong evidence (5 papers) for increased abundance in T2D
- **High consensus confidence (0.85)**: Reliable finding across studies
- **80% direction consistency**: Some heterogeneity, but clear trend
- **Recommendation**: Prioritize for biomarker validation studies

---

### Q2: Intervention Effectiveness Evidence

**Question**: "What interventions (probiotics, FMT, diet) have RCT-level evidence for modifying specific gut taxa?"

**Query**:

```python
result = query_engine.query_intervention_evidence(
    intervention_types=["probiotic", "FMT", "diet"],
    min_sample_size=50,
    evidence_strength="strong"
)
```

**Expected Output**:

```
Query: Intervention effectiveness
Execution time: 1456.2 ms
Results: 8 interventions

Intervention: probiotic
  Taxon: Lactobacillus acidophilus
  Effect: increased
  Papers: 8
  Total sample size: 450
  Average confidence: 0.87
  Paper IDs: PMID:11111111, PMID:22222222, DOI:10.1111/test

Intervention: FMT
  Taxon: Faecalibacterium prausnitzii
  Effect: increased
  Papers: 6
  Total sample size: 320
  Average confidence: 0.84
  Paper IDs: PMID:33333333, DOI:10.2222/test, PMID:44444444
```

**Interpretation**:
- **Probiotic → Lactobacillus**: Strong evidence (8 papers, 450 participants)
- **High confidence (0.87)**: Reliable effect direction
- **Sufficient sample size**: Adequate for clinical recommendations
- **Recommendation**: Consider for evidence-based treatment protocols

---

### Q3: Methodology Landscape and Data Availability

**Question**: "Which microbiome studies from 2020-2024 deposited data on SRA/ENA and used shotgun metagenomics vs 16S sequencing?"

**Query**:

```python
result = query_engine.query_methodology_landscape(
    year_start=2020,
    year_end=2024,
    sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
    require_deposited_data=True
)
```

**Expected Output**:

```
Query: Methodology landscape (2020-2024)
Execution time: 1789.3 ms
Results: 10 method-year combinations

2024 - shotgun metagenomics
  Total papers: 45
  Papers with data: 38
  Data availability: 84.4%
  NCBI SRA: 30
  ENA: 12
  Both repositories: 4

2024 - 16S rRNA sequencing
  Total papers: 120
  Papers with data: 87
  Data availability: 72.5%
  NCBI SRA: 75
  ENA: 15
  Both repositories: 3
```

**Interpretation**:
- **High compliance (84.4%)**: Shotgun metagenomics studies share data more
- **Repository preference**: NCBI SRA is more popular than ENA
- **Trend analysis**: Data availability improving over time
- **Policy impact**: Funding agencies can assess open data compliance

---

### Q4: Top Associations by Evidence Quality

**Question**: "Top 10 taxa associated with IBD across multiple papers with high confidence, ranked by evidence quality."

**Query**:

```python
result = query_engine.query_top_associations_by_evidence(
    disease="IBD",
    top_n=10,
    min_confidence=0.7
)
```

**Expected Output**:

```
Query: Top 10 associations for IBD
Execution time: 987.6 ms
Results: 10 taxa

1. Faecalibacterium prausnitzii
   Papers: 12
   Avg confidence: 0.89
   Direction: decreased
   Consistency: 91.7%
   Breakdown: +1 -11 =0

2. Escherichia coli
   Papers: 10
   Avg confidence: 0.86
   Direction: increased
   Consistency: 90.0%
   Breakdown: +9 -1 =0

3. Roseburia intestinalis
   Papers: 8
   Avg confidence: 0.84
   Direction: decreased
   Consistency: 87.5%
   Breakdown: +1 -7 =0
```

**Interpretation**:
- **Top-ranked taxa**: Strongest evidence base for IBD association
- **High consistency (>90%)**: Robust findings across studies
- **Clear direction**: Decreased beneficial taxa, increased pathobionts
- **Recommendation**: Prioritize for meta-analysis and systematic review

---

### Q5: Conflicting Evidence Detection

**Question**: "Which taxa show conflicting associations (increased vs decreased) for Crohn's disease?"

**Query**:

```python
result = query_engine.query_conflicting_evidence(
    disease="Crohn's Disease",
    min_papers_per_direction=2
)
```

**Expected Output**:

```
Query: Conflicting evidence for Crohn's Disease
Execution time: 3245.7 ms
Results: 4 taxa with conflicts

Taxon: Escherichia coli
  Total papers: 8
  Increased: 5 papers (62.5%)
  Decreased: 3 papers (37.5%)
  Direction balance: 2
  
  Increased papers:
    - PMID:12345678 (2023, RCT)
    - DOI:10.1234/test1 (2022, observational)
    - PMID:23456789 (2021, RCT)
  
  Decreased papers:
    - DOI:10.5678/test2 (2023, observational)
    - PMID:34567890 (2020, RCT)
    - PMID:45678901 (2019, observational)

Taxon: Bacteroides fragilis
  Total papers: 6
  Increased: 3 papers (50.0%)
  Decreased: 3 papers (50.0%)
  Direction balance: 0
```

**Interpretation**:
- **Conflicting evidence**: Suggests heterogeneity in study populations
- **Possible causes**: Different E. coli strains, disease subtypes, methodologies
- **Research need**: Requires subgroup analysis or meta-regression
- **Opportunity**: Follow-up studies to resolve discrepancy

---

## Expected Outputs

### Performance Metrics

| Query | Expected Time | Result Count | Cache Hit Rate |
|-------|--------------|--------------|----------------|
| Q1: Cross-Study Associations | 1.0-2.0s | 5-20 taxa | 85% |
| Q2: Intervention Evidence | 1.5-2.5s | 8-15 interventions | 80% |
| Q3: Methodology Landscape | 1.5-3.0s | 10-20 combinations | 75% |
| Q4: Top Associations | 0.8-1.5s | 10 taxa | 90% |
| Q5: Conflicting Evidence | 2.5-5.0s | 3-10 taxa | 70% |

### Data Quality Indicators

**High-Quality Results**:
- Consensus confidence ≥ 0.80
- Direction consistency ≥ 85%
- Paper count ≥ 5
- Sample size ≥ 100 (for interventions)

**Moderate-Quality Results**:
- Consensus confidence 0.60-0.79
- Direction consistency 70-84%
- Paper count 3-4
- Sample size 50-99

**Low-Quality Results**:
- Consensus confidence < 0.60
- Direction consistency < 70%
- Paper count 1-2
- Sample size < 50

---

## Interpretation Guide

### Understanding Consensus Confidence

**Consensus confidence** is a weighted average of individual relationship confidences:

```
consensus_confidence = Σ(confidence_i × sample_size_i) / Σ(sample_size_i)
```

**Interpretation**:
- **0.90-1.00**: Very high confidence, strong evidence
- **0.80-0.89**: High confidence, reliable findings
- **0.70-0.79**: Moderate confidence, acceptable evidence
- **0.50-0.69**: Low confidence, needs more research
- **< 0.50**: Very low confidence, unreliable

### Understanding Direction Consistency

**Direction consistency** measures agreement across papers:

```
direction_consistency = (papers_with_dominant_direction / total_papers) × 100%
```

**Interpretation**:
- **100%**: Perfect agreement, no conflicting evidence
- **90-99%**: Very high consistency, minor outliers
- **80-89%**: High consistency, some heterogeneity
- **70-79%**: Moderate consistency, notable variation
- **< 70%**: Low consistency, conflicting evidence

### Understanding Evidence Strength

**Evidence strength** classification:

| Strength | Criteria |
|----------|----------|
| **Strong** | p < 0.01, RCT or meta-analysis |
| **Moderate** | p < 0.05, observational with controls |
| **Weak** | p < 0.1 or no p-value |
| **Conflicting** | ≥2 papers per opposite direction |

### Clinical Significance

**When to act on findings**:

✅ **High Priority** (Act Now):
- Consensus confidence ≥ 0.85
- Direction consistency ≥ 90%
- Paper count ≥ 5
- Evidence strength: strong

⚠️ **Medium Priority** (Consider):
- Consensus confidence 0.70-0.84
- Direction consistency 80-89%
- Paper count 3-4
- Evidence strength: moderate

❌ **Low Priority** (More Research Needed):
- Consensus confidence < 0.70
- Direction consistency < 80%
- Paper count < 3
- Evidence strength: weak or conflicting

---

## Troubleshooting

### Common Issues

#### 1. Empty Query Results

**Problem**: Query returns 0 results

**Solutions**:
```python
# Lower confidence threshold
result = query_engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    confidence_threshold=0.5  # Instead of 0.7
)

# Reduce minimum papers
result = query_engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    min_papers=1  # Instead of 3
)

# Change study type to "any"
result = query_engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    study_type="any"  # Instead of "RCT"
)

# Disable open data requirement
result = query_engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    require_open_data=False
)
```

#### 2. Slow Query Performance

**Problem**: Query takes > 5 seconds

**Solutions**:
```bash
# Check Neo4j indexes
cypher-shell -u neo4j -p password
> SHOW INDEXES;

# Create missing indexes
> CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year);
> CREATE INDEX taxon_name IF NOT EXISTS FOR (t:Taxon) ON (t.name);

# Enable query cache
export QUERY_CACHE_ENABLED=true

# Increase Neo4j memory
# Edit neo4j.conf:
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g
```

#### 3. Connection Errors

**Problem**: Cannot connect to Neo4j

**Solutions**:
```bash
# Verify Neo4j is running
systemctl status neo4j

# Check connection
python test_neo4j_connection.py

# Verify credentials
neo4j-admin set-initial-password new_password

# Check firewall
sudo ufw allow 7687/tcp
```

#### 4. Missing Data

**Problem**: Graph has no data

**Solutions**:
```bash
# Run migration script
python scripts/migrate_to_enhanced_schema.py

# Load sample data
python graph/demo_batch_processing.py

# Verify data
cypher-shell -u neo4j -p password
> MATCH (n) RETURN count(n);
```

---

## Advanced Usage

### Custom Queries

Create custom queries by extending `ResearchQueryEngine`:

```python
from graph.research_query_engine import ResearchQueryEngine

class CustomQueryEngine(ResearchQueryEngine):
    def query_custom_analysis(self, param1, param2):
        cypher = """
            MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)
            WHERE r.disease = $param1
              AND r.confidence >= $param2
            RETURN t.name, count(p) as papers
            ORDER BY papers DESC
        """
        
        return self.execute_query(
            cypher_query=cypher,
            parameters={"param1": param1, "param2": param2},
            description="Custom analysis query"
        )
```

### Batch Processing

Process multiple queries in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

diseases = ["Type 2 Diabetes", "IBD", "Crohn's Disease"]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(
            query_engine.query_cross_study_associations,
            disease=disease
        )
        for disease in diseases
    ]
    
    results = [f.result() for f in futures]
```

### Export Results

Export query results to various formats:

```python
import json
import csv

# Export to JSON
with open("results.json", "w") as f:
    json.dump(result.results, f, indent=2)

# Export to CSV
with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result.results[0].keys())
    writer.writeheader()
    writer.writerows(result.results)
```

---

## Additional Resources

- **[QUERY_EXAMPLES.md](../QUERY_EXAMPLES.md)**: Detailed query examples
- **[README.md](../README.md)**: System overview
- **[ARCHITECTURE.md](../ARCHITECTURE.md)**: Architecture documentation
- **[MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md)**: Migration instructions

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [QUERY_EXAMPLES.md](../QUERY_EXAMPLES.md)
3. Open an issue on GitHub
4. Contact the development team

---

**Last Updated**: 2024
**Version**: 1.0
**Requirements**: 1.1, 1.2, 1.3
