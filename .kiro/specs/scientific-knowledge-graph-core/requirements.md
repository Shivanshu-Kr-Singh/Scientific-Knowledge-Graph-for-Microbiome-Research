# Requirements Document

## Introduction

This document specifies the requirements for transforming the microbiome research literature mining system from a data storage pipeline into a scientific knowledge graph that enables discovery. The system must answer three core scientific questions about disease-microbiome associations, intervention effectiveness, and data availability across the research literature. The requirements are derived from the technical design and focus on semantic relationship extraction, provenance tracking, evidence aggregation, and research query capabilities.

## Glossary

- **Knowledge_Graph**: A graph database storing scientific entities (papers, taxa, diseases, methods) and their relationships with rich semantic properties and provenance metadata
- **Provenance_Metadata**: Complete lineage information tracking a graph relationship from source text to final edge, including extraction method, timestamp, and confidence
- **Semantic_Relationship**: A graph edge carrying scientific semantics (direction, statistical measures, effect sizes) rather than simple adjacency
- **Reified_Claim**: A scientific claim represented as a first-class graph node that aggregates evidence from multiple papers
- **Consensus_Confidence**: A weighted average confidence score calculated across multiple papers supporting the same claim
- **Evidence_Strength**: Classification of relationship quality based on study design and statistical significance (strong, moderate, weak, conflicting)
- **Extraction_Method**: A registered algorithm or model used to extract relationships from text (regex_ner, biobert_ner, llm_extractor)
- **Research_Query**: A structured query that answers one of the three core scientific questions by aggregating evidence across papers
- **Entity_Normalization**: The process of grounding entity mentions to canonical ontology identifiers (NCBI Taxonomy, MeSH)
- **Direction_Consistency**: The percentage of papers agreeing on the effect direction (increased/decreased) for a given association

## Requirements

### Requirement 1: Core Scientific Questions

**User Story:** As a microbiome researcher, I want the knowledge graph to answer specific scientific questions about disease associations, interventions, and data availability, so that I can discover patterns across the literature.

#### Acceptance Criteria

1. THE Knowledge_Graph SHALL support queries for cross-study disease-microbiome associations filtered by study type and data availability
2. THE Knowledge_Graph SHALL support queries for intervention effectiveness with evidence aggregation across multiple papers
3. THE Knowledge_Graph SHALL support queries for methodology landscape and data deposition trends over time periods
4. WHEN a research query is executed with at least one matching paper, THE Knowledge_Graph SHALL return results with aggregated evidence metrics including paper count, consensus confidence, and effect direction consistency
4a. WHEN a research query executes successfully but finds no matching results, THE Knowledge_Graph SHALL return an empty result set with metadata indicating no matches were found
5. THE Knowledge_Graph SHALL support filtering query results by minimum paper count, confidence threshold, and evidence strength

### Requirement 2: Semantic Relationship Extraction

**User Story:** As a knowledge graph builder, I want to extract relationships with rich scientific semantics from papers, so that the graph captures meaningful scientific claims rather than simple adjacency.

#### Acceptance Criteria

1. WHEN extracting taxon-disease associations, THE Semantic_Relationship_Extractor SHALL capture direction (increased/decreased/no_change), comparison context, statistical measure type, effect size, and p-value
2. WHEN extracting intervention effects, THE Semantic_Relationship_Extractor SHALL capture intervention type, effect direction, duration, dosage, and sample size
3. WHEN extracting methodology usage, THE Semantic_Relationship_Extractor SHALL capture method name, sequencing platform, sample size, and data availability status
4. THE Semantic_Relationship_Extractor SHALL only create relationships with extraction confidence >= 0.5
5. WHEN multiple statistical measures are found for the same relationship, THE Semantic_Relationship_Extractor SHALL create separate edges for each distinct claim

### Requirement 3: Complete Provenance Tracking

**User Story:** As a data curator, I want every graph relationship to have complete provenance metadata, so that I can trace claims back to source text and verify extraction quality.

#### Acceptance Criteria

1. THE Provenance_Encoder SHALL capture paper identifier, section type, source sentence, and sentence offset for every relationship
2. THE Provenance_Encoder SHALL record extraction method, extractor version, extraction timestamp, and confidence score for every relationship
3. WHEN an LLM-based extraction method is used, THE Provenance_Encoder SHALL store the LLM prompt hash for reproducibility
4. THE Provenance_Encoder SHALL capture surrounding context (±2 sentences) and figure/table references when available
5. THE Knowledge_Graph SHALL reject any relationship that lacks required provenance fields (paper_id, section, source_sentence, extraction_method, timestamp, confidence) and SHALL validate that provenance values are reasonable (positive timestamps, confidence in range [0.0, 1.0])

### Requirement 4: Relationship Reification

**User Story:** As a researcher, I want scientific claims to be aggregated across multiple papers, so that I can assess consensus and identify conflicting evidence.

#### Acceptance Criteria

1. WHEN multiple papers report the same (subject, predicate, object) triple, THE Relationship_Reifier SHALL create a reified claim node aggregating all supporting evidence
2. THE Reified_Claim SHALL maintain separate lists of supporting and contradicting paper IDs with no overlap between the two sets
3. THE Reified_Claim SHALL calculate consensus confidence as a weighted average of individual relationship confidences weighted by sample size
4. THE Reified_Claim SHALL calculate effect direction consistency as the percentage of papers agreeing on the dominant direction
5. THE Reified_Claim SHALL track temporal evolution with first_reported and last_updated timestamps where first_reported <= last_updated, and SHALL allow first_reported = last_updated when a claim is first created
6. WHEN a reified claim has both supporting and contradicting evidence, THE Relationship_Reifier SHALL set evidence_strength to "conflicting" regardless of whether the >= 2 papers per direction threshold is met

### Requirement 5: Evidence Strength Classification

**User Story:** As a researcher, I want relationships classified by evidence strength, so that I can prioritize high-quality findings.

#### Acceptance Criteria

1. WHEN a relationship has p_value < 0.01 (including p_value = 0.0) and paper article_type is "original_research" or "meta_analysis", THE System SHALL classify evidence_strength as "strong"
2. WHEN a relationship has p_value < 0.05, THE System SHALL classify evidence_strength as "moderate"
3. WHEN a relationship has p_value < 0.1 or no p_value, THE System SHALL classify evidence_strength as "weak"
4. WHEN a reified claim has contradicting evidence from >= 2 papers per direction, THE System SHALL classify evidence_strength as "conflicting"
5. THE System SHALL validate that all p_values are in the range [0.0, 1.0]

### Requirement 6: Cross-Study Association Queries

**User Story:** As a researcher, I want to find taxa with consistent disease associations across multiple studies, so that I can identify robust biomarkers.

#### Acceptance Criteria

1. WHEN querying cross-study associations, THE Research_Query_Engine SHALL filter by disease entity, study type, minimum paper count, and confidence threshold
2. WHEN require_open_data is true, THE Research_Query_Engine SHALL only return results from papers with data_availability = "open" and non-empty accession_numbers
3. THE Research_Query_Engine SHALL aggregate results by taxon and calculate consensus confidence, consensus direction, and direction consistency
4. THE Research_Query_Engine SHALL only return taxa appearing in >= min_papers papers with consensus_confidence >= confidence_threshold
5. THE Research_Query_Engine SHALL sort results by consensus confidence descending, then paper count descending

### Requirement 7: Intervention Effectiveness Queries

**User Story:** As a clinician, I want to find interventions with RCT-level evidence for modifying specific taxa, so that I can make evidence-based treatment decisions.

#### Acceptance Criteria

1. WHEN querying intervention evidence, THE Research_Query_Engine SHALL filter by intervention types, minimum sample size, and evidence strength
2. THE Research_Query_Engine SHALL only return interventions from papers with article_type "original_research" or "meta_analysis"
3. THE Research_Query_Engine SHALL group results by (intervention, taxon, effect_direction) and calculate total sample size and paper count
4. THE Research_Query_Engine SHALL only return results with total_sample_size >= min_sample_size
5. THE Research_Query_Engine SHALL sort results by paper count descending, then total sample size descending

### Requirement 8: Methodology Landscape Queries

**User Story:** As a funding agency, I want to survey data availability and methodology trends over time, so that I can assess compliance with open data policies.

#### Acceptance Criteria

1. WHEN querying methodology landscape, THE Research_Query_Engine SHALL filter by year range, sequencing methods, and data deposition requirement
2. WHEN require_deposited_data is true, THE Research_Query_Engine SHALL only return papers with non-empty accession_numbers
3. THE Research_Query_Engine SHALL group results by (method, year) and calculate total papers, papers with data, and data availability percentage
4. THE Research_Query_Engine SHALL identify which papers deposited data in NCBI SRA vs ENA repositories
5. THE Research_Query_Engine SHALL sort results by year descending, then method ascending

### Requirement 9: Conflicting Evidence Detection

**User Story:** As a researcher, I want to identify taxa with conflicting associations, so that I can investigate sources of heterogeneity and design follow-up studies.

#### Acceptance Criteria

1. WHEN querying conflicting evidence, THE Research_Query_Engine SHALL identify taxa with both "increased" and "decreased" associations for the same disease
2. THE Research_Query_Engine SHALL only return taxa with >= min_papers_per_direction papers supporting each direction (using the threshold as a filter to determine which taxa are returned)
3. THE Research_Query_Engine SHALL calculate the percentage of papers supporting each direction
4. THE Research_Query_Engine SHALL return paper metadata (DOI, year, study_design) for all conflicting papers
5. THE Research_Query_Engine SHALL sort results by total paper count descending, then by direction balance (abs(increased_count - decreased_count) ascending)

### Requirement 10: Extraction Method Traceability

**User Story:** As a system maintainer, I want all extraction methods to be registered and versioned, so that I can reproduce results and roll back problematic extractors.

#### Acceptance Criteria

1. THE System SHALL maintain a registry of all extraction methods with unique identifiers (regex_ner, biobert_ner, llm_extractor_v1.2)
2. WHEN creating a relationship, THE System SHALL validate that extraction_method exists in the registered extractors list before allowing relationship creation
3. THE System SHALL record extractor_version for every relationship
4. WHEN an LLM-based extraction method is used, THE System SHALL compute and store a hash of the prompt template
5. THE System SHALL support querying relationships by extraction_method and extractor_version for auditing and rollback

### Requirement 11: Entity Normalization and Grounding

**User Story:** As a data curator, I want entity mentions to be grounded to canonical ontology identifiers, so that the same entity is consistently represented across papers.

#### Acceptance Criteria

1. WHEN creating a taxon node and entity grounding succeeds, THE Entity_Normalizer SHALL create a grounded node, ground the entity to NCBI Taxonomy, and store the canonical identifier
2. WHEN creating a disease node and entity grounding succeeds, THE Entity_Normalizer SHALL create a grounded node, ground the entity to MeSH ontology, and store the canonical identifier
3. WHEN an entity cannot be grounded to an ontology or fails the complete normalization pipeline, THE Entity_Normalizer SHALL create an "ungrounded" node with grounded=false property
4. THE Entity_Normalizer SHALL attempt fuzzy matching with edit distance <= 2 for entities that fail exact matching
5. THE Entity_Normalizer SHALL log all normalization failures to an entity_normalization_failures table for curator review

### Requirement 12: Graph Schema and Indexes

**User Story:** As a system administrator, I want the graph schema to support efficient queries, so that research queries complete within acceptable time limits.

#### Acceptance Criteria

1. THE Knowledge_Graph SHALL create indexes on paper properties (year, article_type, data_availability)
2. THE Knowledge_Graph SHALL create indexes on entity properties (taxon name, disease name)
3. THE Knowledge_Graph SHALL create indexes on relationship properties (confidence, p_value, intervention_type)
4. THE Knowledge_Graph SHALL create composite indexes for common query patterns (year+article_type, evidence_strength+consensus_confidence)
5. THE Knowledge_Graph SHALL support node types: Paper, Taxon, Disease, Method, ScientificClaim

### Requirement 13: Query Performance

**User Story:** As a researcher, I want queries to complete quickly, so that I can interactively explore the knowledge graph.

#### Acceptance Criteria

1. WHEN executing a simple query (single paper lookup), THE Research_Query_Engine SHALL complete within 50 milliseconds
2. WHEN executing an aggregation query (cross-study associations), THE Research_Query_Engine SHALL complete within 2 seconds
3. WHEN executing a complex query (conflicting evidence detection), THE Research_Query_Engine SHALL complete within 5 seconds
4. WHEN a query takes exactly 30 seconds, THE Research_Query_Engine SHALL allow it to complete normally
4a. WHEN a query exceeds 30 seconds, THE Research_Query_Engine SHALL cancel execution and return partial results with a timeout flag
4b. WHEN the timeout detection or cancellation mechanism fails, THE Research_Query_Engine SHALL implement a hard kill mechanism that forcibly terminates queries after a grace period beyond the 30-second threshold
5. THE Research_Query_Engine SHALL implement query result caching with 24-hour TTL for common research questions

### Requirement 14: Data Validation and Quality

**User Story:** As a data curator, I want the system to validate all data before loading into the graph, so that the knowledge graph maintains high quality.

#### Acceptance Criteria

1. THE System SHALL validate that all confidence scores are in the range [0.0, 1.0]
2. THE System SHALL validate that all p_values (when present) are in the range [0.0, 1.0]
3. THE System SHALL validate that direction values are in the set {"increased", "decreased", "no_change"}
4. THE System SHALL validate that evidence_strength values are in the set {"strong", "moderate", "weak", "conflicting"}
5. THE System SHALL store relationships that fail validation in a separate validation queue for manual review rather than discarding them completely

### Requirement 15: Error Handling and Recovery

**User Story:** As a system operator, I want the system to handle errors gracefully and provide recovery mechanisms, so that data quality is maintained.

#### Acceptance Criteria

1. WHEN extraction fails to capture provenance data, THE System SHALL log a warning and add the paper to an "incomplete_extraction" queue without creating a graph edge
2. WHEN multiple conflicting statistical measures are found in the same paper, THE System SHALL create separate edges for each distinct claim and flag the paper with "conflicting_statistics"
3. WHEN entity normalization fails, THE System SHALL create an "ungrounded" node with temporary ID and add to curator review queue
4. WHEN a query times out, THE System SHALL cancel execution, return partial results with timeout flag, and log the query pattern for optimization
5. WHEN attempting to create a reified claim with opposite predicate to an existing claim, THE System SHALL create separate claims and link them with CONFLICTS_WITH relationship

### Requirement 16: Migration from Current System

**User Story:** As a system maintainer, I want to migrate from the current flat relationship system to the new semantic system without data loss, so that existing work is preserved.

#### Acceptance Criteria

1. THE Migration_Process SHALL run the new extraction pipeline in parallel with the existing system writing to a separate Neo4j database instance
2. THE Migration_Process SHALL verify that the new system extracts >= 90% of entities from the old system
3. THE Migration_Process SHALL add provenance metadata retroactively to existing relationships where possible, marking others as "legacy"
4. THE Migration_Process SHALL maintain the old Neo4j database instance as a rollback option until migration is validated
5. THE Migration_Process SHALL prevent decommissioning of the old system until all five research queries are successfully completed and validated on production data

### Requirement 17: Scalability and Throughput

**User Story:** As a system operator, I want the system to scale to thousands of papers, so that it can support comprehensive literature coverage.

#### Acceptance Criteria

1. THE Knowledge_Graph SHALL support at least 10,000 papers with 50,000+ relationships
2. THE Extraction_Pipeline SHALL process papers in batches of 100 with parallel workers (8-16 workers)
3. THE Extraction_Pipeline SHALL achieve throughput of at least 100 papers/minute for regex-based extraction
4. THE System SHALL implement incremental processing to only extract from new papers
5. THE System SHALL use Neo4j batch import for bulk loading with 10,000 nodes/edges per transaction

### Requirement 18: Security and Access Control

**User Story:** As a system administrator, I want to protect the knowledge graph from unauthorized access and malicious queries, so that data integrity is maintained.

#### Acceptance Criteria

1. THE Query_API SHALL use parameterized Cypher queries exclusively to prevent injection attacks
2. THE Query_API SHALL validate and sanitize all user inputs (disease names, taxa names, thresholds)
3. THE Query_API SHALL implement query complexity limits (max depth, max result count)
4. THE Query_API SHALL rate limit query endpoints to 10 queries per minute per user
5. THE System SHALL log all data access for audit trail and comply with publisher terms of service

### Requirement 19: Reproducibility and Auditability

**User Story:** As a researcher, I want to reproduce extraction results and audit the knowledge graph, so that I can verify scientific claims.

#### Acceptance Criteria

1. THE System SHALL store extraction method source code hash for reproducibility
2. THE System SHALL store LLM prompt hash for all LLM-based extractions
3. THE System SHALL support querying all relationships extracted by a specific method version
4. THE System SHALL support rollback of extractions by method version
5. THE System SHALL maintain an audit log of all graph modifications with timestamp and user ID

### Requirement 20: Testing and Validation

**User Story:** As a developer, I want comprehensive test coverage, so that the system is reliable and correct.

#### Acceptance Criteria

1. THE System SHALL achieve >= 85% line coverage and >= 80% branch coverage for all components
2. THE System SHALL implement property-based tests for provenance completeness, reified claim consistency, and query result threshold compliance
3. THE System SHALL implement integration tests for end-to-end graph construction, multi-paper aggregation, and query performance
4. THE System SHALL validate that all property-based tests run with >= 100 iterations and SHALL prohibit running tests with insufficient iterations (zero or below the minimum threshold)
4a. THE System SHALL require both adequate code coverage (>= 85% line, >= 80% branch) and sufficient property-based test iterations (>= 100) to be met together
5. THE System SHALL implement provenance traceability tests verifying that each edge traces back to the correct section and sentence
