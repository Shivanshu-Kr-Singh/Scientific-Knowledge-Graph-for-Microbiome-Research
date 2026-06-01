# Migration Guide: Old System → Enhanced Knowledge Graph

This guide helps users migrate from the old flat relationship system to the new enhanced knowledge graph with semantic relationships, provenance tracking, and evidence aggregation.

---

## Table of Contents

1. [Overview](#overview)
2. [What's New](#whats-new)
3. [Migration Steps](#migration-steps)
4. [Configuration Changes](#configuration-changes)
5. [Query Migration](#query-migration)
6. [API Changes](#api-changes)
7. [Rollback Plan](#rollback-plan)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Migrate?

The old system stored simple adjacency relationships without scientific semantics or provenance. The new enhanced knowledge graph provides:

- **Semantic Relationships**: Rich properties (direction, statistical measures, effect sizes, p-values)
- **Complete Provenance**: Every relationship traces back to source sentence, extraction method, and confidence
- **Evidence Aggregation**: Claims are reified across multiple papers with consensus metrics
- **Entity Normalization**: Entities grounded to canonical ontologies (NCBI Taxonomy, MeSH)
- **Research Queries**: Five specialized queries answer specific scientific questions
- **Query Performance**: Optimized indexes and caching for fast queries

### Migration Timeline

The migration runs in parallel with the old system:

1. **Phase 1**: Enhanced pipeline runs alongside old system (both databases active)
2. **Phase 2**: Validation ensures >= 90% entity coverage and all queries work
3. **Phase 3**: New system becomes primary, old system kept for rollback
4. **Phase 4**: Old system decommissioned after validation period

**Current Status**: Phase 3 complete — Enhanced system is now primary

---

## What's New

### 1. Semantic Relationships

**Old System:**
```cypher
(Paper)-[:HAS_TAXON]->(Taxon)
```

**New System:**
```cypher
(Paper)-[:REPORTS_ASSOCIATION {
    direction: "increased",
    comparison: "T2D vs healthy",
    statistical_measure: "LDA score",
    effect_size: 3.2,
    p_value: 0.001,
    confidence: 0.87,
    evidence_strength: "strong",
    section: "results",
    source_sentence: "Bacteroides fragilis was significantly increased...",
    extraction_method: "llm_extractor_v1.2",
    extraction_timestamp: "2024-01-15T10:30:00Z"
}]->(Taxon)
```

### 2. Relationship Types

| Relationship Type | Purpose | Properties |
|-------------------|---------|------------|
| `REPORTS_ASSOCIATION` | Taxon-disease associations | direction, statistical_measure, effect_size, p_value, comparison |
| `REPORTS_INTERVENTION_EFFECT` | Intervention effects on taxa | intervention_type, effect_direction, duration, dosage, sample_size |
| `USES_METHODOLOGY` | Methodology usage | method_name, sequencing_platform, sample_size, data_availability |

### 3. Provenance Tracking

Every relationship includes:
- `section`: Source section (abstract, methods, results, discussion)
- `source_sentence`: Exact sentence supporting the relationship
- `extraction_method`: Method used (regex_ner, biobert_ner, llm_extractor_v1.2)
- `extraction_timestamp`: When the relationship was extracted
- `confidence`: Confidence score (0.0-1.0)
- `surrounding_context`: ±2 sentences for context (optional)

### 4. Evidence Aggregation

**Reified Claims** aggregate evidence across multiple papers:

```python
{
    "claim_id": "uuid",
    "subject_entity": "Bacteroides fragilis",
    "predicate": "associated_with_increased_abundance",
    "object_entity": "Type 2 Diabetes",
    "supporting_papers": ["PMID:12345", "PMID:67890", ...],
    "contradicting_papers": [],
    "consensus_confidence": 0.85,
    "effect_direction_consistency": 0.80,
    "evidence_strength": "strong",
    "total_sample_size": 450
}
```

### 5. Entity Normalization

Entities are grounded to canonical ontologies:

**Old System:**
```
Taxon: "Bacteroides fragilis" (string)
Disease: "Type 2 Diabetes" (string)
```

**New System:**
```
Taxon: {
    name: "Bacteroides fragilis",
    ncbi_taxonomy_id: "817",
    grounded: true,
    canonical_name: "Bacteroides fragilis"
}

Disease: {
    name: "Type 2 Diabetes",
    mesh_id: "D003924",
    grounded: true,
    canonical_name: "Diabetes Mellitus, Type 2"
}
```

### 6. Research Queries

Five specialized queries replace manual Cypher:

1. **Cross-Study Associations**: Find taxa with consistent disease associations
2. **Intervention Evidence**: Find interventions with RCT-level evidence
3. **Methodology Landscape**: Survey data availability and methodology trends
4. **Top Associations**: Find top taxa by evidence quality
5. **Conflicting Evidence**: Identify taxa with contradictory findings

---

## Migration Steps

### Step 1: Backup Your Data

```bash
# Backup old Neo4j database
neo4j-admin dump --database=neo4j --to=/path/to/backup/neo4j_old.dump

# Backup configuration
cp config.py config.py.backup
cp .env .env.backup
```

### Step 2: Update Configuration

Edit `.env`:

```env
# Enhanced Knowledge Graph (primary system)
NEO4J_ENHANCED_URI=bolt://localhost:7687
NEO4J_ENHANCED_USER=neo4j
NEO4J_ENHANCED_PASSWORD=your_password
NEO4J_ENHANCED_DATABASE=neo4j_enhanced
ENHANCED_PIPELINE_ENABLED=true

# Enhanced Pipeline Settings
ENHANCED_BATCH_SIZE=100
ENHANCED_NUM_WORKERS=8

# Query Engine Settings
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL_HOURS=24
QUERY_TIMEOUT_SECONDS=30

# Entity Normalization
ENTITY_NORMALIZATION_ENABLED=true
ENTITY_FUZZY_MATCH_THRESHOLD=2

# Provenance Tracking
PROVENANCE_CONTEXT_SENTENCES=2
PROVENANCE_VALIDATION_STRICT=true

# Evidence Aggregation
MIN_CONFIDENCE_THRESHOLD=0.5
REIFICATION_ENABLED=true

# Legacy System (for rollback only - DO NOT USE FOR NEW DATA)
NEO4J_LEGACY_URI=bolt://localhost:7688
NEO4J_LEGACY_USER=neo4j
NEO4J_LEGACY_PASSWORD=your_password
NEO4J_LEGACY_DATABASE=neo4j_legacy
```

### Step 3: Start Enhanced Neo4j Database

```bash
# Using Docker Compose (recommended)
docker-compose -f docker-compose.neo4j-dual.yml up -d

# Or create a new database in Neo4j Desktop named "neo4j_enhanced"
```

### Step 4: Run Enhanced Pipeline

```bash
# Process existing papers with enhanced pipeline
python -m graph.enhanced_kg_pipeline

# This will:
# 1. Extract semantic relationships with provenance
# 2. Normalize entities to ontologies
# 3. Reify claims across papers
# 4. Create optimized indexes
# 5. Load to Neo4j enhanced database
```

### Step 5: Validate Migration

```bash
# Run validation script
python validate_migration_completeness.py

# Expected output:
# ✅ Entity coverage: 95.2% (>= 90% required)
# ✅ All relationships have provenance
# ✅ All 5 research queries execute successfully
# ✅ Query performance meets requirements
# ✅ Cache hit rate: 75%
```

### Step 6: Update Your Code

See [Query Migration](#query-migration) section below.

### Step 7: Test Thoroughly

```bash
# Run all tests
pytest

# Run specific test suites
pytest graph/test_research_query_engine.py
pytest graph/test_semantic_extractor.py
pytest graph/test_provenance.py
pytest api/test_query_api.py
```

### Step 8: Switch to Enhanced System

Once validation passes, update your application to use the enhanced system:

```python
# Old code
from graph.kg_pipeline import KGPipeline
pipeline = KGPipeline()

# New code
from graph.enhanced_kg_pipeline import EnhancedKGPipeline
pipeline = EnhancedKGPipeline()
```

---

## Configuration Changes

### Environment Variables

| Variable | Old System | New System | Notes |
|----------|-----------|------------|-------|
| `NEO4J_URI` | Primary | Deprecated | Use `NEO4J_ENHANCED_URI` |
| `NEO4J_DATABASE` | `neo4j` | `neo4j_enhanced` | Separate database |
| `ENHANCED_PIPELINE_ENABLED` | N/A | `true` | Enable enhanced features |
| `QUERY_CACHE_ENABLED` | N/A | `true` | Enable query caching |
| `ENTITY_NORMALIZATION_ENABLED` | N/A | `true` | Enable entity grounding |
| `REIFICATION_ENABLED` | N/A | `true` | Enable evidence aggregation |

### config.py Changes

**Removed:**
- `NEO4J_URI` (replaced by `NEO4J_ENHANCED_URI`)
- `NEO4J_USER` (replaced by `NEO4J_ENHANCED_USER`)
- `NEO4J_PASSWORD` (replaced by `NEO4J_ENHANCED_PASSWORD`)

**Added:**
- `NEO4J_ENHANCED_*` variables for enhanced system
- `QUERY_CACHE_*` variables for query caching
- `ENTITY_NORMALIZATION_*` variables for entity grounding
- `PROVENANCE_*` variables for provenance tracking
- `REIFICATION_ENABLED` for evidence aggregation
- `NEO4J_LEGACY_*` variables for rollback (deprecated)

---

## Query Migration

### Manual Cypher → Research Queries

**Old System (Manual Cypher):**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("""
        MATCH (p:Paper)-[:HAS_TAXON]->(t:Taxon)
        WHERE p.disease = $disease
        RETURN t.name, count(p) as paper_count
        ORDER BY paper_count DESC
    """, disease="Type 2 Diabetes")
    
    for record in result:
        print(f"{record['t.name']}: {record['paper_count']} papers")
```

**New System (Research Queries):**
```python
from graph.research_query_engine import ResearchQueryEngine
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    database="neo4j_enhanced"
)
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
          f"confidence={taxon['consensus_confidence']:.2f}, "
          f"direction={taxon['consensus_direction']}")
```

### Query Mapping

| Old Query | New Query Method |
|-----------|------------------|
| Find taxa for disease | `query_cross_study_associations()` |
| Find interventions | `query_intervention_evidence()` |
| Survey methodology | `query_methodology_landscape()` |
| Top associations | `query_top_associations_by_evidence()` |
| Find conflicts | `query_conflicting_evidence()` |

### Example Migrations

**Example 1: Find Taxa for Disease**

Old:
```cypher
MATCH (p:Paper)-[:HAS_TAXON]->(t:Taxon)
WHERE p.disease = "IBD"
RETURN t.name, count(p) as paper_count
ORDER BY paper_count DESC
LIMIT 10
```

New:
```python
result = engine.query_top_associations_by_evidence(
    disease="IBD",
    top_n=10,
    min_confidence=0.7
)
```

**Example 2: Find Interventions**

Old:
```cypher
MATCH (p:Paper)-[:HAS_INTERVENTION]->(i:Intervention)
WHERE i.type = "probiotic"
RETURN i.name, count(p) as paper_count
```

New:
```python
result = engine.query_intervention_evidence(
    intervention_types=["probiotic"],
    min_sample_size=50,
    evidence_strength="strong"
)
```

**Example 3: Survey Data Availability**

Old:
```cypher
MATCH (p:Paper)
WHERE p.year >= 2020 AND p.year <= 2024
RETURN p.year, count(p) as total,
       sum(CASE WHEN p.has_data THEN 1 ELSE 0 END) as with_data
```

New:
```python
result = engine.query_methodology_landscape(
    year_start=2020,
    year_end=2024,
    sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
    require_deposited_data=True
)
```

---

## API Changes

### REST API Endpoints

**Old System:**
```bash
# No REST API - direct Cypher queries only
```

**New System:**
```bash
# Query cross-study associations
curl -X POST http://localhost:8000/query/cross-study-associations \
  -H "Content-Type: application/json" \
  -d '{"disease": "Type 2 Diabetes", "study_type": "RCT", "min_papers": 3}'

# Query intervention evidence
curl -X POST http://localhost:8000/query/intervention-evidence \
  -H "Content-Type: application/json" \
  -d '{"intervention_types": ["probiotic"], "min_sample_size": 50}'

# Query methodology landscape
curl -X POST http://localhost:8000/query/methodology-landscape \
  -H "Content-Type: application/json" \
  -d '{"year_start": 2020, "year_end": 2024, "sequencing_methods": ["16S rRNA sequencing"]}'

# Query top associations
curl -X POST http://localhost:8000/query/top-associations \
  -H "Content-Type: application/json" \
  -d '{"disease": "IBD", "top_n": 10, "min_confidence": 0.7}'

# Query conflicting evidence
curl -X POST http://localhost:8000/query/conflicting-evidence \
  -H "Content-Type: application/json" \
  -d '{"disease": "Crohn'\''s Disease", "min_papers_per_direction": 2}'
```

### Python Client

**Old System:**
```python
# Direct Neo4j driver usage
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as session:
    result = session.run("MATCH (p:Paper) RETURN p")
```

**New System:**
```python
# Research Query Engine
from graph.research_query_engine import ResearchQueryEngine
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    database="neo4j_enhanced"
)
engine = ResearchQueryEngine(driver)

result = engine.query_cross_study_associations(
    disease="Type 2 Diabetes",
    study_type="RCT",
    min_papers=3
)
```

---

## Rollback Plan

If you need to rollback to the old system:

### Step 1: Switch Configuration

Edit `.env`:

```env
# Disable enhanced pipeline
ENHANCED_PIPELINE_ENABLED=false

# Switch to legacy database
NEO4J_URI=bolt://localhost:7688
NEO4J_DATABASE=neo4j_legacy
```

### Step 2: Restart Services

```bash
# Stop enhanced database
docker-compose -f docker-compose.neo4j-dual.yml down

# Start legacy database
docker-compose -f docker-compose.neo4j-legacy.yml up -d
```

### Step 3: Update Code

```python
# Revert to old pipeline
from graph.kg_pipeline import KGPipeline
pipeline = KGPipeline()
```

### Step 4: Verify Rollback

```bash
# Test old system
python -m graph.kg_pipeline
```

**Note:** The legacy system is deprecated and will be removed in a future release. Rollback is only for emergency situations.

---

## Troubleshooting

### Issue: Migration Validation Fails

**Symptom:** `validate_migration_completeness.py` reports < 90% entity coverage

**Solution:**
1. Check extraction logs: `tail -f logs/miner.log`
2. Verify NLP pipeline is running: `python -m nlp.pipeline`
3. Re-run enhanced pipeline: `python -m graph.enhanced_kg_pipeline`
4. Check for missing papers: `python scripts/check_missing_papers.py`

### Issue: Queries Return Empty Results

**Symptom:** Research queries return 0 results

**Solution:**
1. Verify database connection: `python test_neo4j_connection.py`
2. Check if data is loaded: `MATCH (n) RETURN count(n)`
3. Verify indexes are created: `SHOW INDEXES`
4. Check query parameters (e.g., confidence threshold too high)
5. Invalidate cache: `engine.invalidate_cache()`

### Issue: Slow Query Performance

**Symptom:** Queries take > 5 seconds

**Solution:**
1. Check if indexes exist: `SHOW INDEXES`
2. Create missing indexes: `python -m graph.create_paper_indexes`
3. Enable query caching: `QUERY_CACHE_ENABLED=true`
4. Reduce batch size: `ENHANCED_BATCH_SIZE=50`
5. Check Neo4j memory settings in `neo4j.conf`

### Issue: Entity Normalization Failures

**Symptom:** Many entities have `grounded=false`

**Solution:**
1. Check entity normalization logs: `grep "normalization failed" logs/miner.log`
2. Verify NCBI Taxonomy database is accessible
3. Increase fuzzy match threshold: `ENTITY_FUZZY_MATCH_THRESHOLD=3`
4. Review failed entities: `SELECT * FROM entity_normalization_failures`
5. Manually curate failed entities

### Issue: Provenance Validation Errors

**Symptom:** Relationships rejected due to missing provenance

**Solution:**
1. Check extraction method is registered: `python -m graph.extractor_registry`
2. Verify source sentences are captured: `grep "source_sentence" logs/miner.log`
3. Disable strict validation temporarily: `PROVENANCE_VALIDATION_STRICT=false`
4. Re-run extraction with updated extractor

### Issue: Cache Not Working

**Symptom:** Cache hit rate is 0%

**Solution:**
1. Verify cache is enabled: `QUERY_CACHE_ENABLED=true`
2. Check cache stats: `engine.get_cache_stats()`
3. Increase TTL: `QUERY_CACHE_TTL_HOURS=48`
4. Clear cache: `engine.invalidate_cache()`

---

## Support

For additional help:

1. Check the [README.md](README.md) for detailed documentation
2. Review test files for usage examples
3. Check logs in `logs/miner.log`
4. Run validation scripts in `scripts/`
5. Open an issue on GitHub

---

## Migration Checklist

- [ ] Backup old database
- [ ] Update `.env` configuration
- [ ] Start enhanced Neo4j database
- [ ] Run enhanced pipeline
- [ ] Validate migration (>= 90% coverage)
- [ ] Update code to use research queries
- [ ] Test all queries
- [ ] Verify query performance
- [ ] Check cache hit rate
- [ ] Update documentation
- [ ] Train team on new system
- [ ] Monitor for issues
- [ ] Decommission old system (after validation period)

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Parallel operation | 2 weeks | ✅ Complete |
| Phase 2: Validation | 1 week | ✅ Complete |
| Phase 3: Primary system | 2 weeks | ✅ Complete |
| Phase 4: Decommission old system | 1 week | ✅ Complete |

**Current Status:** Migration complete. Enhanced system is now primary. Old system decommissioned.
