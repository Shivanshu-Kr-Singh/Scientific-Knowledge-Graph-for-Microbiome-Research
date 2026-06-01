# Examples Directory

This directory contains example scripts and workflows demonstrating the Scientific Knowledge Graph system.

## Contents

### 1. End-to-End Workflow

**File**: `end_to_end_workflow.py`

Complete pipeline demonstration from data collection through research queries.

**Features**:
- Data collection from PubMed/EuropePMC
- NLP enrichment (entity extraction, classification)
- Graph construction with semantic relationships
- All 5 core research queries
- Expected outputs and interpretation

**Usage**:

```bash
# Demo mode (no database required)
python examples/end_to_end_workflow.py --mode demo

# Query-only mode (requires existing graph)
python examples/end_to_end_workflow.py --mode query-only

# Full pipeline mode
python examples/end_to_end_workflow.py --mode full
```

**Documentation**: See [END_TO_END_WORKFLOW_GUIDE.md](END_TO_END_WORKFLOW_GUIDE.md)

---

## Quick Start

### Prerequisites

1. Python 3.8+
2. Neo4j database (for non-demo modes)
3. Required dependencies installed

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_neo4j_connection.py
```

### Running Examples

#### Demo Mode (Recommended for First-Time Users)

Shows expected outputs without requiring a database:

```bash
python examples/end_to_end_workflow.py --mode demo
```

#### With Database

```bash
python examples/end_to_end_workflow.py --mode query-only \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password your_password \
    --neo4j-database neo4j_enhanced
```

---

## Example Workflows

### 1. Cross-Study Disease Associations

Find taxa with consistent disease associations:

```python
from graph.research_query_engine import ResearchQueryEngine
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
engine = ResearchQueryEngine(driver)

result = engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    study_type="RCT",
    min_papers=3,
    confidence_threshold=0.7,
    require_open_data=True
)

for taxon in result.results:
    print(f"{taxon['taxon_name']}: {taxon['paper_count']} papers, "
          f"confidence {taxon['consensus_confidence']:.2f}")
```

### 2. Intervention Effectiveness

Find evidence-based interventions:

```python
result = engine.query_intervention_evidence(
    intervention_types=["probiotic", "FMT", "diet"],
    min_sample_size=50,
    evidence_strength="strong"
)

for intervention in result.results:
    print(f"{intervention['intervention_type']} → {intervention['taxon_name']}: "
          f"{intervention['paper_count']} papers, "
          f"n={intervention['total_sample_size']}")
```

### 3. Methodology Landscape

Survey data availability trends:

```python
result = engine.query_methodology_landscape(
    year_start=2020,
    year_end=2024,
    sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
    require_deposited_data=True
)

for row in result.results:
    print(f"{row['year']} - {row['method']}: "
          f"{row['data_availability_pct']:.1f}% data availability")
```

### 4. Top Associations

Find top-ranked taxa by evidence quality:

```python
result = engine.query_top_associations_by_evidence(
    disease="IBD",
    top_n=10,
    min_confidence=0.7
)

for i, taxon in enumerate(result.results, 1):
    print(f"{i}. {taxon['taxon_name']}: "
          f"{taxon['paper_count']} papers, "
          f"confidence {taxon['avg_confidence']:.2f}")
```

### 5. Conflicting Evidence

Identify taxa with conflicting findings:

```python
result = engine.query_conflicting_evidence(
    disease="Crohn's Disease",
    min_papers_per_direction=2
)

for taxon in result.results:
    print(f"{taxon['taxon_name']}: "
          f"{taxon['increased_count']} increased, "
          f"{taxon['decreased_count']} decreased")
```

---

## Expected Outputs

### Query 1: Cross-Study Associations

```
Taxon: Bacteroides fragilis
  Papers: 5
  Consensus confidence: 0.85
  Consensus direction: increased
  Direction consistency: 80.0%
  Increased: 4, Decreased: 1, No change: 0
```

**Interpretation**:
- Strong evidence (5 papers) for increased abundance
- High consensus confidence (0.85) indicates reliable findings
- 80% direction consistency suggests some heterogeneity
- Recommended for biomarker validation studies

### Query 2: Intervention Evidence

```
Intervention: probiotic
  Taxon: Lactobacillus acidophilus
  Effect: increased
  Papers: 8
  Total sample size: 450
  Average confidence: 0.87
```

**Interpretation**:
- Strong evidence (8 papers, 450 participants)
- High confidence (0.87) in effect direction
- Sufficient sample size for clinical recommendations
- Consider for evidence-based treatment protocols

### Query 3: Methodology Landscape

```
2024 - shotgun metagenomics
  Total papers: 45
  Papers with data: 38
  Data availability: 84.4%
  NCBI SRA: 30, ENA: 12
```

**Interpretation**:
- High data sharing compliance (84.4%)
- NCBI SRA is preferred repository
- Trend shows improving data availability
- Useful for policy assessment

### Query 4: Top Associations

```
1. Faecalibacterium prausnitzii
   Papers: 12, Avg confidence: 0.89
   Direction: decreased, Consistency: 91.7%

2. Escherichia coli
   Papers: 10, Avg confidence: 0.86
   Direction: increased, Consistency: 90.0%
```

**Interpretation**:
- Top-ranked taxa have strongest evidence base
- High consistency indicates robust findings
- Prioritize for meta-analysis
- Use for educational materials

### Query 5: Conflicting Evidence

```
Taxon: Escherichia coli
  Total papers: 8
  Increased: 5 papers (62.5%)
  Decreased: 3 papers (37.5%)
  Direction balance: 2
```

**Interpretation**:
- Conflicting evidence suggests heterogeneity
- May indicate different strains or disease subtypes
- Requires subgroup analysis
- Opportunity for follow-up studies

---

## Performance Benchmarks

| Query | Expected Time | Result Count | Cache Hit Rate |
|-------|--------------|--------------|----------------|
| Cross-Study Associations | 1.0-2.0s | 5-20 taxa | 85% |
| Intervention Evidence | 1.5-2.5s | 8-15 interventions | 80% |
| Methodology Landscape | 1.5-3.0s | 10-20 combinations | 75% |
| Top Associations | 0.8-1.5s | 10 taxa | 90% |
| Conflicting Evidence | 2.5-5.0s | 3-10 taxa | 70% |

---

## Troubleshooting

### Empty Results

**Problem**: Query returns 0 results

**Solutions**:
- Lower `confidence_threshold` (try 0.5 instead of 0.7)
- Reduce `min_papers` (try 1 or 2 instead of 3)
- Change `study_type` to "any"
- Set `require_open_data=False`
- Check disease name spelling

### Slow Queries

**Problem**: Query takes > 5 seconds

**Solutions**:
- Check if indexes exist: `SHOW INDEXES` in Neo4j
- Enable query caching: `QUERY_CACHE_ENABLED=true`
- Reduce result set with stricter filters
- Check Neo4j memory settings

### Connection Errors

**Problem**: Cannot connect to Neo4j

**Solutions**:
- Verify Neo4j is running: `systemctl status neo4j`
- Check connection: `python test_neo4j_connection.py`
- Verify credentials
- Check firewall settings

---

## Additional Resources

- **[END_TO_END_WORKFLOW_GUIDE.md](END_TO_END_WORKFLOW_GUIDE.md)**: Comprehensive workflow guide
- **[../QUERY_EXAMPLES.md](../QUERY_EXAMPLES.md)**: Detailed query examples
- **[../README.md](../README.md)**: System overview
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)**: Architecture documentation

---

## Contributing

To add new examples:

1. Create a new Python script in this directory
2. Follow the existing naming convention: `example_*.py`
3. Include docstring with requirements validation
4. Add usage examples and expected outputs
5. Update this README

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Review documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Requirements Validated**: 1.1, 1.2, 1.3
**Last Updated**: 2024
**Version**: 1.0
