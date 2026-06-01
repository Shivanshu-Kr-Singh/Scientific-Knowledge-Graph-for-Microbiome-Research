# Research Query Examples

This document provides comprehensive examples for all five research queries supported by the Scientific Knowledge Graph system.

---

## Table of Contents

1. [Query 1: Cross-Study Disease-Microbiome Associations](#query-1-cross-study-disease-microbiome-associations)
2. [Query 2: Intervention Effectiveness Evidence](#query-2-intervention-effectiveness-evidence)
3. [Query 3: Methodology Landscape and Data Availability](#query-3-methodology-landscape-and-data-availability)
4. [Query 4: Top Associations by Evidence Quality](#query-4-top-associations-by-evidence-quality)
5. [Query 5: Conflicting Evidence Detection](#query-5-conflicting-evidence-detection)

---

## Query 1: Cross-Study Disease-Microbiome Associations

### Purpose
Find taxa with consistent disease associations across multiple studies, filtered by study type and data availability.

### Research Question
"Which gut microbiome taxa show consistent association with Type 2 Diabetes across RCT studies with open sequencing data?"

### Python Example

```python
from graph.research_query_engine import ResearchQueryEngine
from neo4j import GraphDatabase

# Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    database="neo4j_enhanced"
)
engine = ResearchQueryEngine(driver)

# Execute query
result = engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    study_type="RCT",
    min_papers=3,
    confidence_threshold=0.7,
    require_open_data=True
)

# Print results
print(f"Query: {result.query_description}")
print(f"Execution time: {result.execution_time_ms:.1f}ms")
print(f"Results: {result.result_count} taxa\n")

for taxon in result.results:
    print(f"Taxon: {taxon['taxon_name']}")
    print(f"  Papers: {taxon['paper_count']}")
    print(f"  Consensus confidence: {taxon['consensus_confidence']:.2f}")
    print(f"  Consensus direction: {taxon['consensus_direction']}")
    print(f"  Direction consistency: {taxon['direction_consistency']:.1%}")
    print(f"  Increased: {taxon['increased_count']}, "
          f"Decreased: {taxon['decreased_count']}, "
          f"No change: {taxon['no_change_count']}")
    print(f"  Paper IDs: {', '.join(taxon['paper_ids'][:3])}...")
    print()
```


### Example Output

```
Query: Cross-study associations for Type 2 Diabetes (study_type=RCT, min_papers=3, confidence>=0.7, open_data=True)
Execution time: 1234.5ms
Results: 5 taxa

Taxon: Bacteroides fragilis
  Papers: 5
  Consensus confidence: 0.85
  Consensus direction: increased
  Direction consistency: 80.0%
  Increased: 4, Decreased: 1, No change: 0
  Paper IDs: PMID:12345678, DOI:10.1234/test1, PMID:87654321...

Taxon: Faecalibacterium prausnitzii
  Papers: 4
  Consensus confidence: 0.82
  Consensus direction: decreased
  Direction consistency: 100.0%
  Increased: 0, Decreased: 4, No change: 0
  Paper IDs: PMID:23456789, PMID:34567890, DOI:10.5678/test2...

Taxon: Akkermansia muciniphila
  Papers: 3
  Consensus confidence: 0.78
  Consensus direction: increased
  Direction consistency: 66.7%
  Increased: 2, Decreased: 1, No change: 0
  Paper IDs: DOI:10.9012/test3, PMID:45678901, PMID:56789012...
```

### REST API Example

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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disease` | string | required | Disease entity name (e.g., "Type 2 Diabetes", "IBD") |
| `study_type` | string | "RCT" | Study type: "RCT", "observational", "meta_analysis", "any" |
| `min_papers` | integer | 3 | Minimum number of papers required |
| `confidence_threshold` | float | 0.7 | Minimum confidence score (0.0-1.0) |
| `require_open_data` | boolean | true | Only include papers with open data |

### Use Cases

1. **Biomarker Discovery**: Identify robust disease biomarkers supported by multiple RCTs
2. **Meta-Analysis Preparation**: Find taxa with sufficient evidence for meta-analysis
3. **Research Gap Identification**: Identify taxa with low paper counts needing more research
4. **Data Availability Assessment**: Filter for taxa with open data for replication studies

---

## Query 2: Intervention Effectiveness Evidence

### Purpose
Find interventions with RCT-level evidence for modifying specific taxa.

### Research Question
"What interventions (probiotics, FMT, diet) have RCT-level evidence for modifying specific gut taxa, and what effect directions are reported?"

### Python Example

```python
# Execute query
result = engine.query_intervention_evidence(
    intervention_types=["probiotic", "FMT", "diet"],
    min_sample_size=50,
    evidence_strength="strong"
)

# Print results
print(f"Query: {result.query_description}")
print(f"Execution time: {result.execution_time_ms:.1f}ms")
print(f"Results: {result.result_count} interventions\n")

for intervention in result.results:
    print(f"Intervention: {intervention['intervention_type']}")
    print(f"  Taxon: {intervention['taxon_name']}")
    print(f"  Effect: {intervention['effect_direction']}")
    print(f"  Papers: {intervention['paper_count']}")
    print(f"  Total sample size: {intervention['total_sample_size']}")
    print(f"  Average confidence: {intervention['avg_confidence']:.2f}")
    print(f"  Paper IDs: {', '.join(intervention['paper_ids'][:3])}...")
    print()
```


### Example Output

```
Query: Intervention effectiveness (types=['probiotic', 'FMT', 'diet'], min_sample_size=50, evidence_strength=strong)
Execution time: 1456.2ms
Results: 8 interventions

Intervention: probiotic
  Taxon: Lactobacillus acidophilus
  Effect: increased
  Papers: 8
  Total sample size: 450
  Average confidence: 0.87
  Paper IDs: PMID:11111111, PMID:22222222, DOI:10.1111/test...

Intervention: FMT
  Taxon: Faecalibacterium prausnitzii
  Effect: increased
  Papers: 6
  Total sample size: 320
  Average confidence: 0.84
  Paper IDs: PMID:33333333, DOI:10.2222/test, PMID:44444444...

Intervention: diet
  Taxon: Prevotella copri
  Effect: decreased
  Papers: 5
  Total sample size: 280
  Average confidence: 0.81
  Paper IDs: DOI:10.3333/test, PMID:55555555, PMID:66666666...
```

### REST API Example

```bash
curl -X POST http://localhost:8000/query/intervention-evidence \
  -H "Content-Type: application/json" \
  -d '{
    "intervention_types": ["probiotic", "FMT", "diet"],
    "min_sample_size": 50,
    "evidence_strength": "strong"
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intervention_types` | array[string] | required | List of intervention types: "probiotic", "FMT", "diet", "antibiotic" |
| `min_sample_size` | integer | 50 | Minimum total sample size across all papers |
| `evidence_strength` | string | "strong" | Minimum evidence: "strong", "moderate", "weak", "any" |

### Use Cases

1. **Clinical Decision Support**: Find evidence-based interventions for modulating specific taxa
2. **Treatment Planning**: Identify interventions with sufficient sample sizes for clinical use
3. **Research Prioritization**: Find intervention-taxon pairs needing more research
4. **Systematic Review**: Gather evidence for intervention effectiveness reviews

---

## Query 3: Methodology Landscape and Data Availability

### Purpose
Survey data availability and methodology trends over time.

### Research Question
"Which microbiome studies from 2020-2024 deposited data on SRA/ENA and used shotgun metagenomics vs 16S sequencing?"

### Python Example

```python
# Execute query
result = engine.query_methodology_landscape(
    year_start=2020,
    year_end=2024,
    sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
    require_deposited_data=True
)

# Print results
print(f"Query: {result.query_description}")
print(f"Execution time: {result.execution_time_ms:.1f}ms")
print(f"Results: {result.result_count} method-year combinations\n")

for row in result.results:
    print(f"{row['year']} - {row['method']}")
    print(f"  Total papers: {row['total_papers']}")
    print(f"  Papers with data: {row['papers_with_data']}")
    print(f"  Data availability: {row['data_availability_pct']:.1f}%")
    print(f"  NCBI SRA: {row['ncbi_sra_count']}")
    print(f"  ENA: {row['ena_count']}")
    print(f"  Both repositories: {row['both_repositories_count']}")
    print()
```


### Example Output

```
Query: Methodology landscape (2020-2024, methods=['16S rRNA sequencing', 'shotgun metagenomics'], require_deposited_data=True)
Execution time: 1789.3ms
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

2023 - shotgun metagenomics
  Total papers: 38
  Papers with data: 30
  Data availability: 78.9%
  NCBI SRA: 25
  ENA: 10
  Both repositories: 5
```

### REST API Example

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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `year_start` | integer | required | Start year (inclusive) |
| `year_end` | integer | required | End year (inclusive) |
| `sequencing_methods` | array[string] | required | List of sequencing methods |
| `require_deposited_data` | boolean | true | Only include papers with deposited data |

### Use Cases

1. **Funding Agency Reports**: Assess compliance with open data policies
2. **Methodology Trends**: Track adoption of new sequencing technologies
3. **Data Availability Analysis**: Identify gaps in data sharing
4. **Repository Usage**: Compare NCBI SRA vs ENA usage patterns

---

## Query 4: Top Associations by Evidence Quality

### Purpose
Find top taxa associated with a disease ranked by evidence quality.

### Research Question
"Top 10 taxa associated with IBD across multiple papers with high confidence, ranked by evidence quality."

### Python Example

```python
# Execute query
result = engine.query_top_associations_by_evidence(
    disease="IBD",
    top_n=10,
    min_confidence=0.7
)

# Print results
print(f"Query: {result.query_description}")
print(f"Execution time: {result.execution_time_ms:.1f}ms")
print(f"Results: {result.result_count} taxa\n")

for i, taxon in enumerate(result.results, 1):
    print(f"{i}. {taxon['taxon_name']}")
    print(f"   Papers: {taxon['paper_count']}")
    print(f"   Avg confidence: {taxon['avg_confidence']:.2f}")
    print(f"   Direction: {taxon['consensus_direction']}")
    print(f"   Consistency: {taxon['direction_consistency']:.1%}")
    print(f"   Breakdown: +{taxon['increased_count']} "
          f"-{taxon['decreased_count']} "
          f"={taxon['no_change_count']}")
    print()
```


### Example Output

```
Query: Top 10 associations for IBD (min_confidence>=0.7)
Execution time: 987.6ms
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

4. Bacteroides fragilis
   Papers: 7
   Avg confidence: 0.82
   Direction: increased
   Consistency: 85.7%
   Breakdown: +6 -1 =0
```

### REST API Example

```bash
curl -X POST http://localhost:8000/query/top-associations \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "IBD",
    "top_n": 10,
    "min_confidence": 0.7
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disease` | string | required | Disease entity name |
| `top_n` | integer | 10 | Maximum number of taxa to return |
| `min_confidence` | float | 0.7 | Minimum confidence score (0.0-1.0) |

### Use Cases

1. **Literature Review**: Quickly identify most studied taxa for a disease
2. **Biomarker Prioritization**: Focus on taxa with strongest evidence
3. **Research Planning**: Identify well-studied vs understudied taxa
4. **Educational Materials**: Create ranked lists for teaching

---

## Query 5: Conflicting Evidence Detection

### Purpose
Identify taxa with conflicting associations (increased vs decreased) to guide follow-up research.

### Research Question
"Which taxa show conflicting associations (increased vs decreased) for Crohn's disease?"

### Python Example

```python
# Execute query
result = engine.query_conflicting_evidence(
    disease="Crohn's Disease",
    min_papers_per_direction=2
)

# Print results
print(f"Query: {result.query_description}")
print(f"Execution time: {result.execution_time_ms:.1f}ms")
print(f"Results: {result.result_count} taxa with conflicts\n")

for taxon in result.results:
    print(f"Taxon: {taxon['taxon_name']}")
    print(f"  Total papers: {taxon['total_paper_count']}")
    print(f"  Increased: {taxon['increased_count']} papers "
          f"({taxon['increased_percentage']:.1f}%)")
    print(f"  Decreased: {taxon['decreased_count']} papers "
          f"({taxon['decreased_percentage']:.1f}%)")
    print(f"  Direction balance: {taxon['direction_balance']}")
    
    print(f"  Increased papers:")
    for paper in taxon['increased_papers'][:3]:
        print(f"    - {paper['doi']} ({paper['year']}, {paper['study_design']})")
    
    print(f"  Decreased papers:")
    for paper in taxon['decreased_papers'][:3]:
        print(f"    - {paper['doi']} ({paper['year']}, {paper['study_design']})")
    print()
```


### Example Output

```
Query: Conflicting evidence for Crohn's Disease (min_papers_per_direction=2)
Execution time: 3245.7ms
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
  Increased papers:
    - PMID:56789012 (2024, RCT)
    - DOI:10.9012/test3 (2023, meta_analysis)
    - PMID:67890123 (2022, RCT)
  Decreased papers:
    - PMID:78901234 (2023, RCT)
    - DOI:10.3456/test4 (2021, observational)
    - PMID:89012345 (2020, RCT)
```

### REST API Example

```bash
curl -X POST http://localhost:8000/query/conflicting-evidence \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "Crohn'\''s Disease",
    "min_papers_per_direction": 2
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disease` | string | required | Disease entity name |
| `min_papers_per_direction` | integer | 2 | Minimum papers required for each direction |

### Use Cases

1. **Research Gap Identification**: Find taxa needing clarification studies
2. **Systematic Review**: Identify sources of heterogeneity
3. **Meta-Analysis Planning**: Find taxa requiring subgroup analysis
4. **Hypothesis Generation**: Identify potential moderating factors

---

## Advanced Usage

### Combining Multiple Queries

```python
# Find top associations
top_result = engine.query_top_associations_by_evidence(
    disease="Type 2 Diabetes",
    top_n=20,
    min_confidence=0.7
)

# Check for conflicts in top associations
for taxon in top_result.results:
    conflict_result = engine.query_conflicting_evidence(
        disease="Type 2 Diabetes",
        min_papers_per_direction=2
    )
    
    # Filter for this taxon
    conflicts = [c for c in conflict_result.results 
                 if c['taxon_name'] == taxon['taxon_name']]
    
    if conflicts:
        print(f"⚠️  {taxon['taxon_name']} has conflicting evidence!")
    else:
        print(f"✓ {taxon['taxon_name']} has consistent evidence")
```

### Caching and Performance

```python
# First query - cache miss
result1 = engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    study_type="RCT",
    min_papers=3
)
print(f"Execution time: {result1.execution_time_ms:.1f}ms")  # ~1200ms

# Second query - cache hit
result2 = engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    study_type="RCT",
    min_papers=3
)
print(f"Execution time: {result2.execution_time_ms:.1f}ms")  # ~50ms

# Check cache stats
stats = engine.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"Total queries: {stats['total_queries']}")
print(f"Cache hits: {stats['cache_hits']}")
```

### Error Handling

```python
try:
    result = engine.query_cross_study_associations(
        disease="Unknown Disease",
        study_type="RCT",
        min_papers=3
    )
    
    if result.error:
        print(f"Query failed: {result.error}")
    elif result.timeout:
        print("Query timed out after 30 seconds")
    elif result.result_count == 0:
        print("No results found")
    else:
        print(f"Found {result.result_count} results")
        
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Performance Tips

1. **Use Caching**: Identical queries return cached results (24-hour TTL)
2. **Filter Early**: Use higher confidence thresholds to reduce result sets
3. **Limit Results**: Use `top_n` parameter to limit result size
4. **Batch Queries**: Execute multiple queries in parallel when possible
5. **Monitor Performance**: Check `execution_time_ms` in results

---

## Troubleshooting

### Empty Results

**Problem:** Query returns 0 results

**Solutions:**
- Lower `confidence_threshold` (try 0.5 instead of 0.7)
- Lower `min_papers` (try 1 or 2 instead of 3)
- Change `study_type` to "any" instead of "RCT"
- Set `require_open_data=False`
- Check disease name spelling

### Slow Queries

**Problem:** Query takes > 5 seconds

**Solutions:**
- Check if indexes exist: `SHOW INDEXES` in Neo4j
- Enable query caching: `QUERY_CACHE_ENABLED=true`
- Reduce result set with stricter filters
- Check Neo4j memory settings

### Cache Not Working

**Problem:** Cache hit rate is 0%

**Solutions:**
- Verify cache is enabled: `QUERY_CACHE_ENABLED=true`
- Check cache stats: `engine.get_cache_stats()`
- Ensure query parameters are identical
- Clear cache: `engine.invalidate_cache()`

---

## Additional Resources

- [README.md](README.md) - System overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture documentation
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migration from old system
- [API Documentation](api/README.md) - REST API reference
