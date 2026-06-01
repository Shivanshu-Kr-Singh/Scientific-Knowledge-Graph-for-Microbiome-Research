# Requirements Document

## Introduction

This document specifies the requirements for the Deterministic Entity Resolution Pipeline, the second component of the Scientific Knowledge Graph system for microbiome research literature. This pipeline replaces and enhances the basic entity normalization from Spec 1 (exact matching with fuzzy fallback) with a robust, reproducible, multi-strategy resolution system.

The core problem this spec solves: the same biological entity appears under many surface forms across papers ("E. coli", "Escherichia coli", "E.coli", "ATCC 25922") and the current system creates duplicate, unlinked nodes for each variant. This pipeline ensures all surface forms for the same entity resolve to a single canonical node in the knowledge graph, with full auditability and deterministic behavior.

The pipeline must be deterministic — the same input always produces the same output — so that the knowledge graph is reproducible and curators can trust resolution decisions. It must also be efficient enough to process thousands of entities from batch paper ingestion.

## Glossary

- **Resolution_Pipeline**: The ordered sequence of resolution strategies applied to a surface form to find its canonical entity
- **Surface_Form**: A raw entity mention extracted from paper text (e.g., "E. coli", "Escherichia coli", "gut bacteria")
- **Canonical_Entity**: The authoritative representation of an entity with a stable identifier (NCBI Taxonomy ID for taxa, MeSH ID for diseases, custom ID for methods)
- **Canonical_Registry**: The persistent store mapping surface forms and synonyms to canonical entity records
- **Resolution_Strategy**: A single algorithm for matching a surface form to a canonical entity (exact match, normalized match, abbreviation expansion, synonym lookup, fuzzy match, ontology hierarchy traversal)
- **Resolution_Result**: The output of the pipeline for a single surface form, containing the canonical ID, winning strategy, confidence score, and alternatives considered
- **Resolution_Record**: The audit log entry for a single resolution attempt, capturing all strategies tried, candidates found, and the final decision
- **Manual_Override**: A curator-defined mapping that pins a surface form to a specific canonical ID, taking precedence over all automated strategies
- **Conflict_Set**: The set of candidate canonical entities produced by different strategies for the same surface form
- **Ranking_Function**: The deterministic function that selects the winning canonical entity from a conflict set
- **Entity_Merger**: The component that ensures all surface forms resolving to the same canonical ID are linked to a single graph node
- **Resolution_Cache**: The two-tier (in-memory + persistent) cache storing previously computed resolution results
- **Abbreviation_Expander**: The component that maps abbreviated forms (e.g., "E. coli") to their full forms (e.g., "Escherichia coli")
- **Synonym_Index**: The inverted index mapping all known aliases and synonyms to their canonical entity IDs
- **Ontology_Traverser**: The component that walks the NCBI Taxonomy or MeSH hierarchy to find parent/child matches when direct matching fails
- **Resolution_Metrics**: Aggregated statistics tracking resolution rates, failure rates, and per-strategy effectiveness
- **Unresolved_Entity**: A surface form that failed all resolution strategies and was assigned a temporary local ID pending curator review
- **Grounding_Confidence**: A score in [0.0, 1.0] representing the pipeline's certainty that a surface form maps to a given canonical entity

## Requirements

### Requirement 1: Multi-Strategy Resolution Pipeline

**User Story:** As a knowledge graph builder, I want entity mentions to be resolved through a sequence of increasingly flexible strategies, so that the maximum number of entities are grounded to canonical identifiers while maintaining precision.

#### Acceptance Criteria

1. WHEN a surface form is submitted for resolution, THE Resolution_Pipeline SHALL apply strategies in the following fixed order: (1) Manual_Override lookup, (2) exact match against Canonical_Registry, (3) normalized match (case-fold, strip punctuation, collapse whitespace), (4) Abbreviation_Expander lookup, (5) Synonym_Index lookup, (6) fuzzy match with edit distance ≤ 2, (7) Ontology_Traverser hierarchy search
2. WHEN a strategy produces a match with Grounding_Confidence >= 0.5, THE Resolution_Pipeline SHALL accept that match and SHALL NOT apply lower-priority strategies; IF a strategy returns multiple candidates above the 0.5 threshold, THEN THE Resolution_Pipeline SHALL select the candidate with the highest Grounding_Confidence, breaking ties by choosing the candidate whose canonical_id is lexicographically smallest
3. WHEN all strategies fail to produce a match with Grounding_Confidence >= 0.5, THE Resolution_Pipeline SHALL create an Unresolved_Entity record and add the surface form to the curator review queue
4. THE Resolution_Pipeline SHALL record which strategy produced the winning match in the Resolution_Record for every resolution attempt; IF no strategy produces a match, THE Resolution_Pipeline SHALL record the winning_strategy field as "none" in the Resolution_Record
5. WHEN the Abbreviation_Expander expands a surface form, THE Resolution_Pipeline SHALL re-enter the strategy sequence from step (2) using the expanded form, not from step (4); IF the Abbreviation_Expander returns multiple candidate expansions, THE Resolution_Pipeline SHALL try each expansion in lexicographic order through steps (2)–(6) and accept the first expansion that yields a match with Grounding_Confidence >= 0.5; THE Resolution_Pipeline SHALL perform at most one Abbreviation_Expander re-entry per resolution attempt to prevent infinite expansion cycles

### Requirement 2: Deterministic Resolution

**User Story:** As a data curator, I want the same entity mention to always resolve to the same canonical entity, so that the knowledge graph is reproducible and I can trust resolution decisions across pipeline runs.

#### Acceptance Criteria

1. THE Resolution_Pipeline SHALL produce identical Resolution_Result outputs for identical surface form inputs across separate invocations, regardless of invocation order or timing; two surface form inputs are considered identical if they are equal after whitespace trimming and Unicode NFC normalization
2. THE Resolution_Pipeline SHALL NOT use random number generation, probabilistic sampling, or non-deterministic LLM calls in any resolution strategy
3. WHEN multiple candidates share the same Grounding_Confidence score, THE Ranking_Function SHALL break ties first by selecting the candidate from the higher-priority strategy, then by lexicographic ordering of canonical IDs if strategy priority is also equal, ensuring a unique winner
4. THE Resolution_Pipeline SHALL produce the same result when resolving a surface form that was previously resolved and cached as when resolving it fresh without cache; "same result" means all fields of the Resolution_Result (canonical_id, winning_strategy, Grounding_Confidence, and Conflict_Set) are equal
5. WHEN the Canonical_Registry is updated (new synonyms added, overrides set), THE Resolution_Pipeline SHALL invalidate cached results for all surface forms whose resolution outcome could change due to the update — specifically, any surface form that was previously unresolved or that matched via a lower-priority strategy than the newly added synonym or override — and recompute them deterministically on next access

### Requirement 3: Canonical Entity Registry

**User Story:** As a system administrator, I want a persistent registry that maps all known surface forms to canonical entity identifiers, so that entity grounding is consistent across all papers processed by the system.

#### Acceptance Criteria

1. THE Canonical_Registry SHALL store canonical entity records with: canonical_id (NCBI Taxonomy ID for taxa, MeSH ID for diseases, system-assigned ID for methods), primary_name, entity_type, ontology_source, and a list of all known surface forms
2. WHEN a taxon entity is registered, THE Canonical_Registry SHALL validate that the canonical_id is a positive integer; IF validation fails, THE Canonical_Registry SHALL reject the registration and return an error identifying the invalid field without creating a partial record
3. WHEN a disease entity is registered, THE Canonical_Registry SHALL validate that the canonical_id matches the pattern of one uppercase letter followed by one or more digits (e.g., "D006262"); IF validation fails, THE Canonical_Registry SHALL reject the registration and return an error identifying the invalid field without creating a partial record
4. WHEN a method entity is registered, THE Canonical_Registry SHALL validate that the canonical_id matches the pattern "METHOD-" followed by one or more alphanumeric characters (e.g., "METHOD-16S"); IF validation fails, THE Canonical_Registry SHALL reject the registration and return an error without creating a partial record
5. WHEN a lookup is performed by surface form, THE Canonical_Registry SHALL perform a case-insensitive match against all registered surface forms and SHALL return the canonical entity record if a match is found; IF no match is found, THE Canonical_Registry SHALL return a null result without raising an exception
6. WHEN a surface form is added to an existing canonical entity record, THE Canonical_Registry SHALL update the Synonym_Index within the same transaction to maintain consistency between the registry and the index; IF the Synonym_Index update fails, THE Canonical_Registry SHALL roll back the surface form addition and return an error, leaving both the registry and the index in their prior state
7. WHEN a surface form being added to canonical entity A already exists in the Canonical_Registry as a surface form of a different canonical entity B, THE Canonical_Registry SHALL reject the addition and log a conflict record containing: the duplicate surface form, canonical entity A's ID, canonical entity B's ID, and the timestamp of the conflict

### Requirement 4: Conflict Resolution and Deterministic Ranking

**User Story:** As a knowledge graph builder, I want a deterministic mechanism to select the best canonical entity when multiple strategies produce different candidates, so that conflicts are resolved consistently without manual intervention for common cases.

#### Acceptance Criteria

1. WHEN multiple resolution strategies produce different candidate canonical entities for the same surface form, THE Ranking_Function SHALL score each candidate using: strategy priority (Manual_Override = 1.0, exact = 0.95, normalized = 0.85, abbreviation = 0.80, synonym = 0.75, fuzzy = 0.60, ontology = 0.50) multiplied by the strategy's Grounding_Confidence, where Grounding_Confidence is a value in [0.0, 1.0] inclusive
2. THE Ranking_Function SHALL select the candidate with the highest composite score as the winner
3. IF two candidates have identical composite scores, THEN THE Ranking_Function SHALL first select the candidate from the higher-priority strategy; IF strategy priority is also equal, THEN THE Ranking_Function SHALL select the candidate whose canonical_id is lexicographically smallest, ensuring a unique winner in all cases
4. THE Resolution_Record SHALL store all candidates in the Conflict_Set with their scores, not only the winner, to support curator review
5. WHEN the Conflict_Set contains candidates from three or more strategies, THE Resolution_Pipeline SHALL flag the Resolution_Record with a "high_conflict" marker for curator attention
6. WHEN only a single candidate is produced across all strategies, THE Ranking_Function SHALL return that candidate as the winner without scoring or tie-breaking

### Requirement 5: Alias and Synonym Management

**User Story:** As a data curator, I want all known surface forms for each canonical entity to be tracked and searchable, so that any variant of an entity name encountered in new papers resolves to the correct canonical node.

#### Acceptance Criteria

1. THE Synonym_Index SHALL maintain a case-insensitive mapping from every registered surface form (primary name, aliases, abbreviations, misspellings) to its canonical entity ID; each surface form SHALL be at most 500 characters in length and SHALL be stored in Unicode NFC normalized form
2. WHEN a new synonym is added to a canonical entity, THE Synonym_Index SHALL be updated atomically, blocking concurrent lookups for at most 100 milliseconds, so that lookups during the update see either the complete old state or the complete new state, never a partial state
3. THE Canonical_Registry SHALL track the provenance of each synonym: whether it was sourced from the ontology itself, extracted from paper text, or added by a curator
4. WHEN a surface form is registered as a synonym for canonical entity A and later submitted as a synonym for canonical entity B, THE Canonical_Registry SHALL reject the second registration and log a synonym conflict record containing: the duplicate surface form, canonical entity A's ID, canonical entity B's ID, the timestamp of the conflict, and the provenance source of the conflicting submission
5. THE Synonym_Index SHALL support prefix-based lookup returning all registered surface forms that begin with the given prefix, along with their canonical entity IDs, capped at 50 results per query, to enable autocomplete and partial-match queries for curator tooling

### Requirement 6: Cross-Paper Entity Merging

**User Story:** As a knowledge graph builder, I want all surface forms of the same biological entity across different papers to resolve to a single canonical graph node, so that the knowledge graph accurately represents entity co-occurrence and association patterns.

#### Acceptance Criteria

1. WHEN two surface forms from different papers resolve to the same canonical_id, THE Entity_Merger SHALL ensure both papers are linked to the same graph node rather than creating duplicate nodes
2. WHEN a curator establishes that two previously separate Unresolved_Entity graph nodes refer to the same canonical entity, THE Entity_Merger SHALL merge those nodes into the canonical node
3. WHEN merging two previously separate graph nodes, THE Entity_Merger SHALL transfer all inbound and outbound relationships from the merged node to the surviving canonical node; IF a transferred relationship would create a duplicate of an already-existing relationship on the canonical node (same type, same counterpart node, and same direction), THE Entity_Merger SHALL deduplicate by retaining the relationship with the higher confidence score and discarding the other; THEN THE Entity_Merger SHALL delete the merged node
4. THE Entity_Merger SHALL log every merge operation with: source node IDs, target canonical node ID, triggering resolution event, and timestamp
5. WHEN a merge would combine nodes of different entity types (e.g., a Taxon node and a Disease node), THE Entity_Merger SHALL reject the merge, log a type conflict error containing the source node IDs and their respective entity types, and leave both nodes unchanged
6. THE Entity_Merger SHALL execute each merge as an atomic operation; IF any step of the merge fails (relationship transfer, node deletion, or audit log write), THE Entity_Merger SHALL roll back all changes made during that merge attempt and return an error, leaving the graph in its pre-merge state
7. WHEN a merge is rolled back due to failure, THE Entity_Merger SHALL log the rollback event with: source node IDs, target canonical node ID, the step at which failure occurred, and the error message

### Requirement 7: Resolution Audit Trail

**User Story:** As a data curator, I want every resolution decision to be fully logged with the strategy that succeeded, confidence score, and alternatives considered, so that I can review, correct, and reproduce any resolution decision.

#### Acceptance Criteria

1. THE Resolution_Pipeline SHALL create a Resolution_Record for every resolution attempt, whether successful or not
2. THE Resolution_Record SHALL contain: surface_form, entity_type, timestamp (UTC ISO-8601), winning_strategy (or "unresolved"), canonical_id (or null), Grounding_Confidence (a value in [0.0, 1.0]), all candidates in the Conflict_Set with their composite scores, and the paper_id of the paper that triggered the resolution
3. WHEN a Manual_Override is applied, THE Resolution_Record SHALL record winning_strategy as "manual_override" and SHALL include a curator_override field containing the curator_id who set the override
4. THE Resolution_Pipeline SHALL persist Resolution_Records to a dedicated audit store that is logically and physically separate from the Canonical_Registry and the knowledge graph (i.e., stored in a distinct database table or file that is not co-located with registry or graph data)
5. WHEN a Resolution_Record cannot be persisted to the audit store due to a write failure, THE Resolution_Pipeline SHALL log the failure to the system error log including the surface_form and paper_id, and SHALL continue processing without blocking or retrying the failed write
6. THE Resolution_Pipeline SHALL support querying Resolution_Records by surface_form, canonical_id, winning_strategy, date range, and paper_id; query results SHALL be returned in descending timestamp order; IF no records match the query, THE Resolution_Pipeline SHALL return an empty list without raising an exception

### Requirement 8: Batch Resolution with Caching

**User Story:** As a system operator, I want the pipeline to efficiently resolve thousands of entity mentions from batch paper ingestion without redundant computation, so that Layer 3 graph construction completes within acceptable time limits.

#### Acceptance Criteria

1. THE Resolution_Pipeline SHALL accept a batch of 1 to 100,000 surface forms and resolve them in a single call, returning a list of Resolution_Result objects in the same order as the input
2. THE Resolution_Cache SHALL implement a two-tier architecture: an in-memory LRU cache (capacity configurable, default 10,000 entries) backed by a persistent cache (SQLite or equivalent)
3. WHEN a surface form is submitted for resolution and a valid cached result exists (where "valid" means the cache entry's registry_version matches the current Canonical_Registry version), THE Resolution_Cache SHALL return the cached result within 10 milliseconds for an in-memory hit and within 100 milliseconds for a persistent-tier hit, without re-executing any resolution strategy or triggering background recomputation
4. THE Resolution_Cache SHALL store cache entries with: surface_form (key), Resolution_Result (value), cache_timestamp, and registry_version at time of caching
5. WHEN the Canonical_Registry version advances (due to new synonyms or overrides), THE Resolution_Cache SHALL remove all cache entries created under the previous registry version from both the in-memory and persistent tiers before the next resolution request is processed
6. WHEN a surface form is submitted for resolution and no valid cached result exists, THE Resolution_Cache SHALL execute the full resolution strategy sequence, store the resulting Resolution_Result in both cache tiers with the current registry_version, and return the result to the caller

### Requirement 9: Manual Override Support

**User Story:** As a data curator, I want to pin specific surface forms to canonical IDs, so that I can correct automated resolution errors and handle edge cases that the automated strategies cannot resolve correctly.

#### Acceptance Criteria

1. THE Resolution_Pipeline SHALL check for a Manual_Override for the surface form before applying any automated strategy
2. IF a Manual_Override exists for the surface form, THEN THE Resolution_Pipeline SHALL use the override result as the resolution outcome without executing any automated strategy
3. WHEN a curator sets a Manual_Override for a surface form, THE Canonical_Registry SHALL record: surface_form, canonical_id, curator_id, timestamp, and an optional justification note of at most 500 characters
4. THE Resolution_Pipeline SHALL treat Manual_Override results as having Grounding_Confidence = 1.0 for ranking purposes
5. WHEN a curator removes a Manual_Override, THE Resolution_Pipeline SHALL invalidate the cached result for that surface form and recompute using automated strategies on next access
6. WHEN automated resolution strategies are updated or when cached results expire, THE Resolution_Pipeline SHALL recompute affected surface forms on next access
7. THE Resolution_Pipeline SHALL support bulk import of Manual_Overrides from a CSV file with columns: surface_form, canonical_id, entity_type, curator_id, justification; upon completion, THE Resolution_Pipeline SHALL report the count of successfully imported rows and the count of skipped rows
8. WHEN a CSV row is malformed (missing required columns, invalid canonical_id format, or a surface_form that already has a Manual_Override for a different canonical_id), THE Resolution_Pipeline SHALL skip that row, log the row number and reason for skipping, and continue processing remaining rows without aborting the import

### Requirement 10: Resolution Quality Metrics

**User Story:** As a system operator, I want aggregated metrics on resolution performance, so that I can monitor pipeline health, identify problematic entity types, and measure the impact of registry improvements.

#### Acceptance Criteria

1. THE Resolution_Metrics SHALL track, per pipeline run: total surface forms submitted, resolved count, unresolved count, resolution rate (resolved / total, reported as 0.0 when total is 0), and per-strategy resolution counts
2. THE Resolution_Metrics SHALL track, per entity type (taxon, disease, method): resolution rate, average Grounding_Confidence of resolved entities (on a scale of 0.0 to 1.0), and count of Unresolved_Entity records pending curator review
3. WHEN a pipeline run completes, THE Resolution_Metrics SHALL attempt to persist a metrics snapshot with run_id, timestamp, paper_ids processed, and all metric values; IF snapshot persistence fails, THEN THE Resolution_Metrics SHALL log the failure to the system error log and allow the pipeline run to complete normally without blocking or retrying
4. THE Resolution_Metrics SHALL support querying metric snapshots over a specified date range, returning snapshots in ascending timestamp order; a degradation in resolution quality is indicated when the resolution rate for any entity type in the most recent snapshot is more than 5 percentage points lower than the average resolution rate for that entity type across all prior snapshots in the queried range
5. WHEN at least one surface form was processed in a pipeline run and the resolution rate for any entity type falls below 70%, THEN THE Resolution_Metrics SHALL emit a warning event to the system operator log containing: the run_id, the affected entity type, the observed resolution rate, and the 70% threshold

### Requirement 11: Abbreviation Expansion

**User Story:** As a knowledge graph builder, I want abbreviated entity names to be expanded to their full forms before resolution, so that "E. coli" and "Escherichia coli" resolve to the same canonical taxon.

#### Acceptance Criteria

1. THE Abbreviation_Expander SHALL maintain a curated abbreviation table mapping abbreviated forms to their full canonical names for common microbiome taxa, diseases, and methods
2. WHEN a surface form matches an entry in the abbreviation table, THE Abbreviation_Expander SHALL return the full form and a Grounding_Confidence value in [0.0, 1.0] reflecting the specificity of the abbreviation (unambiguous abbreviations SHALL have Grounding_Confidence = 1.0; ambiguous abbreviations SHALL have Grounding_Confidence < 1.0 proportional to the number of candidate expansions)
3. WHEN a surface form matches a genus-initial abbreviation pattern (a single uppercase letter followed by a period and a species epithet, e.g., "E. coli"), THE Abbreviation_Expander SHALL look up all genera whose name begins with that letter and return each matching full binomial name as a candidate expansion; IF no genus in the abbreviation table begins with that letter, THE Abbreviation_Expander SHALL return no candidates for the genus-initial path and allow the Resolution_Pipeline to continue to the next strategy
4. WHEN a genus-initial abbreviation is ambiguous (the initial matches multiple genera), THE Abbreviation_Expander SHALL return all candidate expansions with equal Grounding_Confidence scores equal to 1.0 / number_of_candidates for the Ranking_Function to resolve
5. THE Abbreviation_Expander SHALL support adding new abbreviation mappings via the curator interface; new mappings SHALL take effect for all subsequent resolution calls without requiring a pipeline restart

### Requirement 12: Fuzzy Matching

**User Story:** As a knowledge graph builder, I want surface forms with minor spelling variations or typos to still resolve to the correct canonical entity, so that OCR errors and non-standard spellings do not create spurious unresolved entities.

#### Acceptance Criteria

1. WHEN a surface form reaches the fuzzy matching step (step 6 in the strategy sequence) and has not been resolved by any prior strategy, THE Resolution_Pipeline SHALL apply fuzzy matching using Levenshtein edit distance as the similarity metric against all canonical entity names and registered surface forms in the Canonical_Registry
2. WHEN applying fuzzy matching, THE Resolution_Pipeline SHALL only consider candidates with edit distance ≤ 2 from the normalized surface form (after case-folding, punctuation stripping, and whitespace collapsing)
3. THE Resolution_Pipeline SHALL assign Grounding_Confidence for fuzzy matches as: 1.0 - (edit_distance / max(len(normalized_surface_form), len(normalized_candidate_name))) * 0.5, where lengths are measured in Unicode code points after normalization; edit distance 0 yields Grounding_Confidence 1.0
4. WHEN fuzzy matching produces multiple candidates within the edit distance threshold, THE Resolution_Pipeline SHALL rank them by edit distance ascending, then by canonical_id lexicographic order for tie-breaking
5. WHEN a surface form is shorter than 4 Unicode code points after normalization, THE Resolution_Pipeline SHALL skip fuzzy matching for that surface form and proceed directly to the Ontology_Traverser step (step 7)
6. WHEN fuzzy matching produces no candidates within the edit distance ≤ 2 threshold, THE Resolution_Pipeline SHALL proceed to the Ontology_Traverser step (step 7) without creating an Unresolved_Entity record at this stage

### Requirement 13: Ontology Hierarchy Traversal

**User Story:** As a knowledge graph builder, I want entity mentions that cannot be matched directly to traverse the ontology hierarchy, so that species-level mentions can be resolved to genus-level canonical entities when no species-level entry exists.

#### Acceptance Criteria

1. WHEN all direct matching strategies (steps 1–6) fail to produce a match with Grounding_Confidence >= 0.5, THE Ontology_Traverser SHALL query the NCBI Taxonomy hierarchy (for taxa) or MeSH hierarchy (for diseases) to find the nearest ancestor that exists in the Canonical_Registry; IF the ontology service is unavailable, THE Ontology_Traverser SHALL log a warning and return no candidates, allowing the Resolution_Pipeline to proceed to the unresolved path
2. THE Ontology_Traverser SHALL only traverse up to 3 levels in the hierarchy
3. WHEN the Ontology_Traverser finds a match at hierarchy level N (where N=1 is direct parent), THE Ontology_Traverser SHALL assign Grounding_Confidence as: 0.50 - (N - 1) * 0.10, so parent = 0.50, grandparent = 0.40, great-grandparent = 0.30
4. WHEN the Ontology_Traverser is the winning strategy, THE Resolution_Record SHALL include a hierarchy_level field set to the integer N at which the match was found (1, 2, or 3)
5. WHEN the Ontology_Traverser is used as the winning strategy, THE Resolution_Pipeline SHALL flag the Resolution_Record with a "hierarchy_match" marker to indicate the match is at a higher taxonomic level than the original mention
6. WHEN the Ontology_Traverser traverses all 3 levels without finding any ancestor present in the Canonical_Registry, THE Ontology_Traverser SHALL return no candidates, and THE Resolution_Pipeline SHALL proceed to create an Unresolved_Entity record for the surface form

### Requirement 14: Integration with Spec 1 Entity Normalization

**User Story:** As a system maintainer, I want the new resolution pipeline to replace the basic entity normalization from Spec 1 without breaking existing graph construction, so that Layer 3 continues to function correctly with improved entity grounding.

#### Acceptance Criteria

1. THE Resolution_Pipeline SHALL expose a normalize(surface_form: str, entity_type: str) -> NormalizationResult interface where NormalizationResult contains canonical_id (str or null) and grounded (bool), making it a drop-in replacement for the Spec 1 Entity_Normalizer interface
2. WHEN the Resolution_Pipeline is enabled, THE System SHALL disable the Spec 1 fuzzy fallback (edit distance ≤ 2 with ungrounded node creation) and route all normalization calls through the new pipeline
3. IF a surface form resolves via strategies 1–6 (Manual_Override through fuzzy match) with Grounding_Confidence >= 0.5, THEN THE Resolution_Pipeline SHALL return grounded=true and the resolved canonical_id
4. IF a surface form fails all strategies or achieves Grounding_Confidence < 0.5 across all strategies, THEN THE Resolution_Pipeline SHALL return grounded=false and create an Unresolved_Entity record containing at minimum: surface_form, entity_type, paper_id, and timestamp
5. THE Resolution_Pipeline SHALL be configurable to run in shadow mode; WHEN shadow mode is enabled, THE Resolution_Pipeline SHALL execute alongside the Spec 1 normalizer, return the Spec 1 normalizer's result to the caller, and log any discrepancy where the two systems produce different canonical_ids or different grounded values for the same surface form, without affecting graph construction
6. WHILE the Spec 1 fuzzy fallback is active (shadow mode or parallel operation), THE Resolution_Pipeline SHALL log every discrepancy between the two systems regardless of shadow mode configuration; each discrepancy log entry SHALL contain: surface_form, entity_type, Spec 1 result (canonical_id, grounded), Resolution_Pipeline result (canonical_id, grounded), and timestamp

### Requirement 15: Correctness Properties and Testing

**User Story:** As a developer, I want the resolution pipeline to be verified against key correctness properties, so that I can be confident the system is free of subtle bugs in entity merging and synonym management.

#### Acceptance Criteria

1. THE Resolution_Pipeline SHALL satisfy the idempotency property: for any surface form S that resolves to a canonical_id C, resolve(C) SHALL return C as the canonical_id; for any surface form S that is unresolved, resolve(S) SHALL return an unresolved result
2. THE Resolution_Pipeline SHALL satisfy the synonym completeness property: for any canonical entity E with registered synonyms S1…Sn, resolve(Si).canonical_id == E.canonical_id for all i in 1..n
3. THE Resolution_Pipeline SHALL satisfy the no-spurious-merge property: for any two canonical entities E1 and E2 with different canonical_ids, no surface form that appears in E1's synonym list and does not appear in E2's synonym list SHALL resolve to E2's canonical_id
4. THE Resolution_Pipeline SHALL satisfy the audit completeness property: for every resolution attempt, a Resolution_Record exists in the audit store with a non-empty winning_strategy field (or "unresolved") and a non-null timestamp
5. THE System SHALL implement property-based tests using Hypothesis covering: idempotency of resolution (for surface forms of length 1–500 characters), synonym completeness (for synonym lists of size 1–100), determinism under arbitrary surface form inputs (identical inputs produce identical canonical_id and winning_strategy), and audit record existence for every resolution call; each property SHALL be tested with at least 100 generated examples
6. THE System SHALL implement property-based tests verifying that batch_resolve(forms) produces the same results as [resolve(f) for f in forms] for any list of 1 to 1,000 surface forms (each of length 1–500 characters), confirming batch consistency with single resolution; this property SHALL be tested with at least 100 generated examples

