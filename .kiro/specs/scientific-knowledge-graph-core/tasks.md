# Implementation Plan: Scientific Knowledge Graph Core

## Overview

This implementation transforms the microbiome research knowledge graph from a data pipeline into a scientific discovery system. The migration follows a 4-phase strategy: (1) Parallel Implementation - build new components without disrupting existing pipeline, (2) Incremental Cutover - replace old components with new ones, (3) Query Layer Deployment - implement research queries and indexes, (4) Decommission Old System - remove legacy code and validate migration.

The implementation uses Python with Pydantic models for data validation, Neo4j for graph storage, and maintains backward compatibility during migration.

## Tasks

### Phase 1: Parallel Implementation (Weeks 1-2)

- [x] 1. Create new graph module structure and data models
  - [x] 1.1 Create provenance data models and encoder
    - Create `graph/provenance.py` with `ProvenanceMetadata` Pydantic model
    - Implement `ProvenanceEncoder` class with `encode()` and `validate_provenance()` methods
    - Add validation for confidence scores [0.0, 1.0], non-empty sentences, registered extraction methods
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 10.2_
  
  - [x] 1.2 Write property test for provenance completeness
    - **Property 1: Provenance Completeness**
    - **Validates: Requirements 3.5**
    - Test that all created ProvenanceMetadata instances have required fields (paper_id, section, source_sentence, extraction_method, timestamp, confidence)
    - Test that confidence scores are always in range [0.0, 1.0]
  
  - [x] 1.3 Create semantic relationship data models
    - Create `graph/semantic_relationships.py` with `SemanticRelationship` Pydantic model
    - Define relationship type enums: REPORTS_ASSOCIATION, REPORTS_INTERVENTION_EFFECT, USES_METHODOLOGY
    - Add properties dict structure for each relationship type with validation
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [x] 1.4 Create reified claim data models
    - Create `graph/reified_claims.py` with `ScientificClaim` and `ReifiedClaimNode` Pydantic models
    - Define `EvidenceStrength` enum (strong, moderate, weak, conflicting)
    - Add validation for consensus metrics (consensus_confidence, effect_direction_consistency in [0.0, 1.0])
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_


- [x] 2. Implement Semantic Relationship Extractor
  - [x] 2.1 Create base semantic relationship extractor class
    - Create `graph/semantic_extractor.py` with `SemanticRelationshipExtractor` class
    - Implement section parser to extract results, methods, discussion sections from EnrichedPaperRecord
    - Add helper methods for parsing statistical measures (p-values, effect sizes, fold changes)
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 2.2 Implement association extraction with statistical properties
    - Implement `extract_associations()` method to extract taxon-disease associations
    - Parse direction (increased/decreased/no_change), comparison context, statistical measures
    - Extract effect sizes and p-values from results sections using regex patterns
    - Create ProvenanceMetadata for each extracted association
    - Filter relationships to only include confidence >= 0.5
    - _Requirements: 2.1, 2.4, 3.1, 3.2_
  
  - [x] 2.3 Write unit tests for association extraction
    - Test extraction from sample papers with known associations
    - Test p-value parsing for various formats (0.001, <0.05, p=0.03)
    - Test direction detection from common phrases ("significantly increased", "reduced abundance")
    - _Requirements: 2.1, 2.4_
  
  - [x] 2.4 Implement intervention effect extraction
    - Implement `extract_intervention_effects()` method for RCT and intervention studies
    - Parse intervention types (probiotic, FMT, diet, antibiotic) from methods sections
    - Extract effect direction, duration, dosage from results sections
    - Only include relationships with p_value < 0.05 or explicit significance statements
    - _Requirements: 2.2, 5.2_
  
  - [x] 2.5 Implement methodology extraction
    - Implement `extract_methodology_usage()` method to link papers to methods
    - Extract sequencing platform (Illumina, PacBio), method name (16S, shotgun metagenomics)
    - Parse sample size from methods sections
    - Link to data_availability status from EnrichedPaperRecord
    - _Requirements: 2.3, 8.1, 8.4_


- [x] 3. Implement Relationship Reifier
  - [x] 3.1 Create relationship reifier class and claim aggregation
    - Create `graph/relationship_reifier.py` with `RelationshipReifier` class
    - Implement `reify_claim()` method to create ScientificClaim from multiple evidence pieces
    - Generate unique claim_id using UUID
    - Calculate consensus_confidence as weighted average of individual confidences
    - _Requirements: 4.1, 4.3_
  
  - [x] 3.2 Implement evidence strength classification
    - Implement evidence strength logic: strong (p<0.01, RCT/meta-analysis), moderate (p<0.05), weak (p<0.1 or no p-value)
    - Set evidence_strength to "conflicting" when claim has both supporting and contradicting evidence
    - Validate p_values are in range [0.0, 1.0]
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 3.3 Write property test for reified claim consistency
    - **Property 2: Reified Claim Consistency**
    - **Validates: Requirements 4.2, 4.3**
    - Test that supporting_papers and contradicting_papers lists have no overlap
    - Test that consensus_confidence is always in range [0.0, 1.0]
    - Test that first_reported <= last_updated for all claims
  
  - [x] 3.4 Implement claim update with new evidence
    - Implement `update_claim_with_new_evidence()` method to add supporting or contradicting evidence
    - Recalculate consensus_confidence when new evidence is added
    - Update last_updated timestamp to current time
    - Update evidence_strength if contradicting evidence changes classification
    - _Requirements: 4.5, 4.6_
  
  - [x] 3.5 Implement conflicting claim detection
    - Implement `detect_conflicting_claims()` method to find claim pairs with opposite predicates
    - Only return pairs with same subject and object entities
    - Return empty list if no conflicts found
    - _Requirements: 4.6, 9.1_


- [x] 4. Create enhanced graph builder with new components
  - [x] 4.1 Create new graph builder integrating semantic extractor and reifier
    - Create `graph/enhanced_graph_builder.py` with `EnhancedGraphBuilder` class
    - Integrate SemanticRelationshipExtractor to extract rich relationships
    - Integrate RelationshipReifier to create reified claims
    - Create EnhancedGraphEdge objects with embedded provenance
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 4.1_
  
  - [x] 4.2 Implement extraction method registry
    - Create `graph/extractor_registry.py` with registered extraction methods
    - Add method identifiers: regex_ner, biobert_ner, llm_extractor_v1.2
    - Implement validation to check extraction_method exists before creating relationships
    - Store extractor versions and source code hashes for reproducibility
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 19.1_
  
  - [x] 4.3 Implement entity normalization with ontology grounding
    - Update `graph/entity_normalizer.py` to ground taxa to NCBI Taxonomy and diseases to MeSH
    - Implement fuzzy matching with edit distance <= 2 for failed exact matches
    - Create "ungrounded" nodes with grounded=false when normalization fails
    - Log normalization failures to entity_normalization_failures table
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [x] 4.4 Write integration test for end-to-end graph construction
    - Test complete pipeline from EnrichedPaperRecord to EnhancedGraphEdge
    - Verify provenance metadata is complete for all edges
    - Verify entity normalization succeeds for known entities
    - _Requirements: 20.3_
  
  - [x] 4.5 Implement parallel pipeline execution
    - Create `graph/enhanced_kg_pipeline.py` that runs new pipeline in parallel with old system
    - Write output to separate Neo4j database instance (neo4j_enhanced)
    - Add configuration flag to enable/disable enhanced pipeline
    - _Requirements: 16.1, 17.2_


- [x] 5. Checkpoint - Validate Phase 1 components
  - Ensure all tests pass, verify new components work in parallel with old system, ask the user if questions arise.

### Phase 2: Incremental Cutover (Weeks 3-4)

- [x] 6. Update Neo4j loader for enhanced edges
  - [x] 6.1 Create enhanced Neo4j loader with rich relationship properties
    - Create `graph/enhanced_neo4j_loader.py` with support for EnhancedGraphEdge
    - Implement loading of relationship properties (direction, p_value, effect_size, etc.)
    - Embed provenance metadata as relationship properties
    - Implement batch loading with 10,000 nodes/edges per transaction
    - _Requirement s: 12.5, 17.5_
  
  - [x] 6.2 Implement reified claim node creation in Neo4j
    - Add support for creating ScientificClaim nodes in Neo4j
    - Create relationships between claims and supporting/contradicting papers
    - Store consensus metrics as node properties
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 6.3 Write unit tests for Neo4j loader
    - Test batch loading with large datasets (10,000+ edges)
    - Test provenance metadata is correctly stored as relationship properties
    - Test reified claim nodes are created with correct structure
    - _Requirements: 17.5_
  
  - [x] 6.4 Implement data validation before loading
    - Validate confidence scores in range [0.0, 1.0]
    - Validate p_values in range [0.0, 1.0]
    - Validate direction values in {"increased", "decreased", "no_change"}
    - Validate evidence_strength in {"strong", "moderate", "weak", "conflicting"}
    - Store invalid relationships in validation queue for manual review
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_


- [x] 7. Migrate existing graph data to enhanced schema
  - [x] 7.1 Create migration script for existing relationships
    - Create `scripts/migrate_to_enhanced_schema.py` migration script
    - Read existing relationships from old Neo4j database
    - Add provenance metadata retroactively where possible (mark others as "legacy")
    - Verify >= 90% of entities from old system are extracted by new system
    - _Requirements: 16.2, 16.3_
  
  - [x] 7.2 Implement rollback mechanism
    - Maintain old Neo4j database instance as rollback option
    - Create backup of old database before migration
    - Implement rollback script to restore old database if needed
    - _Requirements: 16.4_
  
  - [x] 7.3 Write integration test for migration validation
    - Test that migrated data matches old data structure
    - Test that provenance metadata is added correctly
    - Test that entity counts match between old and new systems
    - _Requirements: 16.2, 20.3_
  
  - [x] 7.4 Replace old graph builder with enhanced version
    - Update `graph/kg_pipeline.py` to use EnhancedGraphBuilder instead of GraphBuilder
    - Update imports and configuration to point to enhanced components
    - Remove old GraphBuilder instantiation
    - _Requirements: 16.1_

- [x] 8. Checkpoint - Validate Phase 2 migration
  - Ensure all tests pass, verify data migration completed successfully, validate entity counts match, ask the user if questions arise.


### Phase 3: Query Layer Deployment (Weeks 5-6)

- [x] 9. Implement Research Query Engine
  - [x] 9.1 Create query engine base class and result models
    - Create `graph/research_query_engine.py` with `ResearchQueryEngine` class
    - Implement `QueryResult` Pydantic model with query metadata
    - Add query execution timing and result counting
    - Implement parameterized Cypher query generation to prevent injection
    - _Requirements: 1.1, 1.2, 1.3, 18.1_
  
  - [x] 9.2 Implement cross-study association query (Q1)
    - Implement `query_cross_study_associations()` method
    - Filter by disease, study_type, min_papers, confidence_threshold
    - When require_open_data=True, filter papers with data_availability="open" and non-empty accession_numbers
    - Aggregate by taxon, calculate consensus_confidence and direction_consistency
    - Sort results by consensus_confidence DESC, then paper_count DESC
    - _Requirements: 1.1, 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 9.3 Write property test for query result threshold compliance
    - **Property 3: Query Result Threshold Compliance**
    - **Validates: Requirements 6.4, 7.4**
    - Test that all returned results meet min_papers threshold
    - Test that all returned results meet confidence_threshold
    - Test that all returned results meet min_sample_size (for intervention queries)
  
  - [x] 9.4 Implement intervention effectiveness query (Q2)
    - Implement `query_intervention_evidence()` method
    - Filter by intervention_types, min_sample_size, evidence_strength
    - Only return interventions from article_type "original_research" or "meta_analysis"
    - Group by (intervention, taxon, effect_direction), calculate total_sample_size and paper_count
    - Sort by paper_count DESC, then total_sample_size DESC
    - _Requirements: 1.2, 7.1, 7.2, 7.3, 7.4, 7.5_


  - [x] 9.5 Implement methodology landscape query (Q3)
    - Implement `query_methodology_landscape()` method
    - Filter by year_range, sequencing_methods, require_deposited_data
    - When require_deposited_data=True, only return papers with non-empty accession_numbers
    - Group by (method, year), calculate total_papers, papers_with_data, data_availability_pct
    - Identify repository (NCBI SRA vs ENA) from accession number prefixes
    - Sort by year DESC, then method ASC
    - _Requirements: 1.3, 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [x] 9.6 Implement top associations by evidence query (Q4)
    - Implement `query_top_associations_by_evidence()` method
    - Filter by disease, top_n, min_confidence
    - Return at most top_n taxa sorted by (paper_count DESC, avg_confidence DESC)
    - Include aggregated statistics (paper_count, avg_confidence, direction_consistency)
    - _Requirements: 1.4_
  
  - [x] 9.7 Implement conflicting evidence detection query (Q5)
    - Implement `query_conflicting_evidence()` method
    - Find taxa with both "increased" and "decreased" associations for same disease
    - Only return taxa with >= min_papers_per_direction papers supporting each direction
    - Calculate percentage of papers supporting each direction
    - Return paper metadata (DOI, year, study_design) for all conflicting papers
    - Sort by total_paper_count DESC, then abs(increased_count - decreased_count) ASC
    - _Requirements: 1.4, 9.1, 9.2, 9.3, 9.4, 9.5_


- [x] 10. Create Neo4j indexes and optimize queries
  - [x] 10.1 Create indexes on paper properties
    - Create indexes on Paper.year, Paper.article_type, Paper.data_availability
    - Create composite index on (Paper.year, Paper.article_type)
    - Verify index creation with SHOW INDEXES command
    - _Requirements: 12.1, 12.4_
  
  - [x] 10.2 Create indexes on entity properties
    - Create indexes on Taxon.name, Disease.name, Method.name
    - Create index on entity canonical identifiers (Taxon.ncbi_id, Disease.mesh_id)
    - _Requirements: 12.2_
  
  - [x] 10.3 Create indexes on relationship properties
    - Create indexes on relationship.confidence, relationship.p_value, relationship.intervention_type
    - Create composite index on (relationship.evidence_strength, relationship.consensus_confidence)
    - _Requirements: 12.3, 12.4_
  
  - [x] 10.4 Write performance tests for query execution time
    - Test simple queries complete within 50ms
    - Test aggregation queries complete within 2 seconds
    - Test complex queries complete within 5 seconds
    - Test timeout mechanism cancels queries exceeding 30 seconds
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.4a, 13.4b_
  
  - [x] 10.5 Implement query result caching
    - Implement caching layer with 24-hour TTL for common research questions
    - Use query parameters as cache key
    - Invalidate cache when new data is loaded
    - _Requirements: 13.5_


- [x] 11. Create query API with security controls
  - [x] 11.1 Create Flask/FastAPI query endpoint
    - Create `api/query_api.py` with REST endpoints for all 5 research queries
    - Implement POST endpoints: /query/cross-study-associations, /query/intervention-evidence, /query/methodology-landscape, /query/top-associations, /query/conflicting-evidence
    - Return QueryResult JSON with execution metadata
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [x] 11.2 Implement input validation and sanitization
    - Validate disease names, taxa names against allowed entity lists
    - Validate numeric thresholds (confidence, min_papers, year ranges)
    - Sanitize all user inputs to prevent injection attacks
    - Return 400 Bad Request for invalid inputs with error details
    - _Requirements: 18.2_
  
  - [x] 11.3 Implement query complexity limits and rate limiting
    - Limit max result count to 1000 per query
    - Limit query depth to prevent expensive graph traversals
    - Implement rate limiting: 10 queries per minute per user
    - Return 429 Too Many Requests when rate limit exceeded
    - _Requirements: 18.3, 18.4_
  
  - [x] 11.4 Write integration tests for query API
    - Test all 5 query endpoints with valid inputs
    - Test input validation rejects invalid parameters
    - Test rate limiting blocks excessive requests
    - Test query timeout returns partial results with timeout flag
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 20.3_
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 20.3_

- [x] 12. Checkpoint - Validate Phase 3 query layer
  - Ensure all tests pass, verify all 5 research queries work on production data, validate query performance meets requirements, ask the user if questions arise.


### Phase 4: Decommission Old System (Week 7)

- [x] 13. Remove legacy code and finalize migration
  - [x] 13.1 Validate migration completeness
    - Run all 5 research queries on production data and verify results
    - Compare entity counts between old and new systems (should be >= 90% match)
    - Verify provenance metadata exists for all non-legacy relationships
    - _Requirements: 16.2, 16.5_
  
  - [x] 13.2 Remove old graph builder and relation extractor
    - Delete `graph/graph_builder.py` (old version)
    - Delete `graph/relation_extractor.py` (old version)
    - Remove old GraphRecord and GraphNode classes if not used by enhanced system
    - _Requirements: 16.5_
  
  - [x] 13.3 Update documentation and configuration
    - Update README.md with new architecture diagram and query examples
    - Document all 5 research queries with example parameters
    - Update configuration files to remove old system references
    - Add migration guide for users of old system
    - _Requirements: 16.5_
  
  - [x] 13.4 Implement error handling and recovery mechanisms
    - Add error handling for extraction failures (log warning, add to incomplete_extraction queue)
    - Handle conflicting statistics by creating separate edges and flagging paper
    - Handle entity normalization failures by creating ungrounded nodes
    - Implement query timeout handling with partial results and timeout flag
    - Handle conflicting claims by creating separate claims with CONFLICTS_WITH relationship
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_


  - [x] 13.5 Implement audit logging and reproducibility
    - Create audit log table for all graph modifications with timestamp and user ID
    - Store extraction method source code hash for reproducibility
    - Store LLM prompt hash for all LLM-based extractions
    - Implement query interface to find relationships by extraction method version
    - Implement rollback functionality to remove extractions by method version
    - _Requirements: 10.3, 10.4, 10.5, 18.5, 19.1, 19.2, 19.3, 19.4, 19.5_
  
  - [x] 13.6 Write property test for provenance traceability
    - **Property 4: Provenance Traceability**
    - **Validates: Requirements 3.1, 3.2, 20.5**
    - Test that every edge traces back to a valid section (abstract, methods, results, discussion)
    - Test that source_sentence is non-empty for all edges
    - Test that extraction_method exists in registered extractors
  
  - [x] 13.7 Decommission old Neo4j database
    - Create final backup of old Neo4j database
    - Shut down old Neo4j instance
    - Archive old database files for historical reference
    - Update connection strings to point only to enhanced database
    - _Requirements: 16.4, 16.5_

- [x] 14. Final checkpoint - Complete migration validation
  - Ensure all tests pass, verify all 5 research queries work correctly, confirm old system is fully decommissioned, ask the user if questions arise.


### Additional Quality and Testing Tasks

- [x] 15. Implement comprehensive test coverage
  - [x] 15.1 Achieve >= 85% line coverage and >= 80% branch coverage
    - Write unit tests for all components (ProvenanceEncoder, SemanticRelationshipExtractor, RelationshipReifier, ResearchQueryEngine)
    - Write integration tests for end-to-end workflows
    - Run coverage report and identify untested code paths
    - _Requirements: 20.1_
  
  - [x] 15.2 Run all property-based tests with >= 100 iterations
    - Configure property test framework (Hypothesis for Python) with min_iterations=100
    - Run Property 1 (Provenance Completeness) with 100+ iterations
    - Run Property 2 (Reified Claim Consistency) with 100+ iterations
    - Run Property 3 (Query Result Threshold Compliance) with 100+ iterations
    - Run Property 4 (Provenance Traceability) with 100+ iterations
    - _Requirements: 20.2, 20.4, 20.4a_
  
  - [x] 15.3 Write integration test for multi-paper aggregation
    - Test reified claim creation from 3+ papers with same (subject, predicate, object)
    - Test consensus_confidence calculation with varying individual confidences
    - Test conflicting evidence detection when papers have opposite directions
    - _Requirements: 20.3_

- [x] 16. Implement scalability optimizations
  - [x] 16.1 Implement batch processing with parallel workers
    - Configure extraction pipeline to process papers in batches of 100
    - Implement parallel workers (8-16 workers) for extraction
    - Achieve throughput of >= 100 papers/minute for regex-based extraction
    - _Requirements: 17.2, 17.3_
  
  - [x] 16.2 Implement incremental processing
    - Track processed papers in database to avoid re-extraction
    - Only extract from new papers added since last run
    - Update reified claims incrementally when new evidence is added
    - _Requirements: 17.4_


  - [x] 16.3 Validate scalability to 10,000+ papers
    - Load test with 10,000 papers and 50,000+ relationships
    - Verify query performance remains within requirements at scale
    - Monitor memory usage and optimize if needed
    - _Requirements: 17.1_

- [x] 17. Final integration and wiring
  - [x] 17.1 Wire all components together in main pipeline
    - Update `main.py` to use EnhancedKGPipeline
    - Connect SemanticRelationshipExtractor → ProvenanceEncoder → RelationshipReifier → EnhancedNeo4jLoader
    - Add configuration options for enabling/disabling features
    - _Requirements: 16.1_
  
  - [x] 17.2 Create end-to-end example workflow
    - Create example script demonstrating full pipeline: collection → enrichment → graph construction → query
    - Include example queries for all 5 research questions
    - Document expected outputs and interpretation
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [x] 17.3 Write end-to-end integration test
    - Test complete workflow from raw papers to query results
    - Verify provenance is maintained throughout pipeline
    - Verify query results match expected patterns
    - _Requirements: 20.3_

- [x] 18. Final checkpoint - Complete system validation
  - Ensure all tests pass, verify system meets all requirements, confirm documentation is complete, ask the user if questions arise.


## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at each phase boundary
- Property tests validate universal correctness properties across the system
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- The 4-phase migration strategy ensures zero downtime and safe rollback
- Phase 1 builds new components in parallel without disrupting existing pipeline
- Phase 2 incrementally replaces old components with validation at each step
- Phase 3 deploys query layer and validates research queries on production data
- Phase 4 decommissions old system only after complete validation
- All components use Python with Pydantic for data validation
- Neo4j is used for graph storage with Cypher queries
- Security controls include parameterized queries, input validation, rate limiting
- Scalability targets: 10,000+ papers, 100 papers/minute throughput, query times <5s
- Test coverage targets: >= 85% line coverage, >= 80% branch coverage, >= 100 PBT iterations


## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": ["1.1", "1.3", "1.4"]
    },
    {
      "id": 1,
      "tasks": ["1.2", "2.1", "3.1", "4.2"]
    },
    {
      "id": 2,
      "tasks": ["2.2", "2.4", "2.5", "3.2", "4.1", "4.3"]
    },
    {
      "id": 3,
      "tasks": ["2.3", "3.3", "3.4", "3.5", "4.4"]
    },
    {
      "id": 4,
      "tasks": ["4.5", "6.1"]
    },
    {
      "id": 5,
      "tasks": ["6.2", "6.3", "6.4"]
    },
    {
      "id": 6,
      "tasks": ["7.1"]
    },
    {
      "id": 7,
      "tasks": ["7.2", "7.3", "7.4"]
    },
    {
      "id": 8,
      "tasks": ["9.1", "10.1", "10.2", "10.3"]
    },
    {
      "id": 9,
      "tasks": ["9.2", "9.4", "9.5", "9.6", "10.5"]
    },
    {
      "id": 10,
      "tasks": ["9.3", "9.7", "10.4", "11.1"]
    },
    {
      "id": 11,
      "tasks": ["11.2", "11.3"]
    },
    {
      "id": 12,
      "tasks": ["11.4", "13.1"]
    },
    {
      "id": 13,
      "tasks": ["13.2", "13.3", "13.4", "13.5"]
    },
    {
      "id": 14,
      "tasks": ["13.6", "13.7", "15.1"]
    },
    {
      "id": 15,
      "tasks": ["15.2", "15.3", "16.1"]
    },
    {
      "id": 16,
      "tasks": ["16.2", "16.3"]
    },
    {
      "id": 17,
      "tasks": ["17.1"]
    },
    {
      "id": 18,
      "tasks": ["17.2", "17.3"]
    }
  ]
}
```
