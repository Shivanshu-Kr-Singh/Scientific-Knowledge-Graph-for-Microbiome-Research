"""
graph/enhanced_neo4j_loader.py
-------------------------------
Enhanced Neo4j loader with support for rich relationship properties.

This module implements batch loading of nodes and relationships with complete
provenance metadata and scientific semantics embedded as relationship properties.

Requirements: 12.5, 17.5
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Session
from datetime import datetime
import logging

from graph.semantic_relationships import SemanticRelationship, RelationType
from graph.reified_claims import ScientificClaim
from graph.data_validator import DataValidator
from graph.audit_log import get_audit_log, AuditLog
from graph.extractor_registry import get_registry


logger = logging.getLogger(__name__)


class EnhancedNeo4jLoader:
    """
    Neo4j loader with support for EnhancedGraphEdge (SemanticRelationship)
    and batch loading for performance.
    
    This loader embeds complete provenance metadata and scientific semantics
    as relationship properties, enabling rich queries and traceability.
    
    Requirements:
    - 12.5: Graph schema and indexes
    - 17.5: Batch import with 10,000 nodes/edges per transaction
    """
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        batch_size: int = 10000,
        validation_queue_path: Optional[str] = None,
        audit_log: Optional[AuditLog] = None,
    ):
        """
        Initialize the enhanced Neo4j loader.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            batch_size: Number of nodes/edges per transaction (default: 10,000)
            validation_queue_path: Path to store invalid relationships (default: data/validation_queue.json)
            audit_log: AuditLog instance (default: global instance)
        
        Requirements:
        - 17.5: Batch import with 10,000 nodes/edges per transaction
        - 14.5: Store invalid relationships in validation queue
        - 19.5: Audit logging for all graph modifications
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size
        self.validator = DataValidator(validation_queue_path)
        self.audit_log = audit_log or get_audit_log()
        self.extractor_registry = get_registry()
        logger.info(
            f"Initialized EnhancedNeo4jLoader with batch_size={batch_size}"
        )
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        logger.info("Closed Neo4j driver connection")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def create_node(
        self,
        session: Session,
        node_type: str,
        node_id: str,
        properties: Dict[str, Any]
    ) -> None:
        """
        Create or update a single node in Neo4j.
        
        Args:
            session: Neo4j session
            node_type: Node label (e.g., "Paper", "Taxon", "Disease")
            node_id: Unique node identifier
            properties: Node properties as dictionary
        
        Requirement 12.5: Support node types: Paper, Taxon, Disease, Method, ScientificClaim
        Requirement 19.5: Log all graph modifications
        """
        # Convert datetime objects to ISO format strings
        serialized_props = self._serialize_properties(properties)
        
        query = f"""
        MERGE (n:{node_type} {{id: $id}})
        SET n += $props
        """
        
        session.run(query, id=node_id, props=serialized_props)
        
        # Log node creation to audit log
        self.audit_log.log_node_creation(
            node_type=node_type,
            node_id=node_id,
            properties=serialized_props,
            user_id="system",
        )
    
    def create_relationship(
        self,
        session: Session,
        relationship: SemanticRelationship
    ) -> None:
        """
        Create a relationship with rich properties from SemanticRelationship.
        
        This method embeds complete provenance metadata and scientific semantics
        as relationship properties, enabling traceability and rich queries.
        
        Args:
            session: Neo4j session
            relationship: SemanticRelationship with embedded provenance
        
        Requirements:
        - 12.5: Embed provenance metadata as relationship properties
        - 17.5: Support batch loading
        - 19.1: Store extraction method source code hash
        - 19.2: Store LLM prompt hash for LLM-based extractions
        - 19.5: Log all graph modifications
        """
        # Build relationship properties from semantic properties and provenance
        rel_props = self._build_relationship_properties(relationship)
        
        # Create relationship with all properties
        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{relationship.relation_type.value}]->(target)
        SET r += $props
        """
        
        session.run(
            query,
            source_id=relationship.source_entity,
            target_id=relationship.target_entity,
            props=rel_props
        )
        
        # Get extraction method metadata for audit log
        extractor_metadata = self.extractor_registry.get_extractor(
            relationship.provenance.extraction_method
        )
        
        # Log edge creation to audit log with extraction metadata
        self.audit_log.log_edge_creation(
            relationship_type=relationship.relation_type.value,
            source_id=relationship.source_entity,
            target_id=relationship.target_entity,
            properties=rel_props,
            user_id="system",
            extraction_method=relationship.provenance.extraction_method,
            extractor_version=relationship.provenance.extractor_version,
            source_code_hash=extractor_metadata.source_code_hash if extractor_metadata else None,
            llm_prompt_hash=relationship.provenance.llm_prompt_hash,
            paper_id=relationship.provenance.paper_id,
            section=relationship.provenance.section_type,
        )
    
    def _build_relationship_properties(
        self,
        relationship: SemanticRelationship
    ) -> Dict[str, Any]:
        """
        Build relationship properties dictionary from SemanticRelationship.
        
        This combines:
        1. Scientific semantic properties (direction, p_value, effect_size, etc.)
        2. Provenance metadata (section, source_sentence, extraction_method, etc.)
        3. Quality indicators (confidence, evidence_strength)
        
        Args:
            relationship: SemanticRelationship with embedded provenance
        
        Returns:
            Dictionary of relationship properties ready for Neo4j
        
        Requirement 12.5: Embed provenance metadata as relationship properties
        """
        props = {}
        
        # Add all semantic properties
        props.update(relationship.properties)
        
        # Add provenance metadata (embedded)
        provenance = relationship.provenance
        props.update({
            "section": provenance.section_type,
            "source_sentence": provenance.source_sentence,
            "sentence_offset": provenance.sentence_offset,
            "extraction_method": provenance.extraction_method,
            "extraction_timestamp": provenance.extraction_timestamp.isoformat(),
            "extractor_version": provenance.extractor_version,
            "llm_prompt_hash": provenance.llm_prompt_hash,
            "validation_status": provenance.validation_status,
            "validator_id": provenance.validator_id,
            "surrounding_context": provenance.surrounding_context,
            "figure_table_ref": provenance.figure_table_ref,
        })
        
        # Add quality indicators
        props.update({
            "confidence": relationship.extraction_confidence,
            "evidence_strength": relationship.evidence_strength,
        })
        
        # Serialize any remaining complex types
        return self._serialize_properties(props)
    
    def _serialize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize property values for Neo4j storage.
        
        Converts datetime objects to ISO format strings and handles None values.
        
        Args:
            properties: Dictionary of properties
        
        Returns:
            Dictionary with serialized values
        """
        serialized = {}
        for key, value in properties.items():
            if value is None:
                # Skip None values
                continue
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                # Neo4j supports lists and maps
                serialized[key] = value
            else:
                serialized[key] = value
        
        return serialized
    
    def load_nodes_batch(
        self,
        nodes: List[Dict[str, Any]],
        node_type: str
    ) -> None:
        """
        Load nodes in batches for performance.
        
        Args:
            nodes: List of node dictionaries with 'id' and properties
            node_type: Node label (e.g., "Paper", "Taxon")
        
        Requirement 17.5: Batch loading with 10,000 nodes/edges per transaction
        """
        total_nodes = len(nodes)
        logger.info(f"Loading {total_nodes} {node_type} nodes in batches of {self.batch_size}")
        
        with self.driver.session() as session:
            for i in range(0, total_nodes, self.batch_size):
                batch = nodes[i:i + self.batch_size]
                
                with session.begin_transaction() as tx:
                    for node in batch:
                        node_id = node.pop('id')
                        properties = self._serialize_properties(node)
                        
                        query = f"""
                        MERGE (n:{node_type} {{id: $id}})
                        SET n += $props
                        """
                        
                        tx.run(query, id=node_id, props=properties)
                    
                    tx.commit()
                
                logger.info(
                    f"Loaded batch {i // self.batch_size + 1}: "
                    f"{min(i + self.batch_size, total_nodes)}/{total_nodes} nodes"
                )
    
    def load_relationships_batch(
        self,
        relationships: List[SemanticRelationship],
        validate: bool = True
    ) -> Dict[str, int]:
        """
        Load relationships in batches for performance with optional validation.
        
        This method processes SemanticRelationship objects and creates
        Neo4j relationships with embedded provenance and semantic properties.
        
        When validation is enabled (default), relationships are validated before
        loading. Invalid relationships are stored in a validation queue for
        manual review rather than being loaded into the graph.
        
        Args:
            relationships: List of SemanticRelationship objects
            validate: Whether to validate relationships before loading (default: True)
        
        Returns:
            Dictionary with counts: {"loaded": int, "invalid": int, "total": int}
        
        Requirements:
        - 17.5: Batch loading with 10,000 nodes/edges per transaction
        - 14.1, 14.2, 14.3, 14.4: Validate data before loading
        - 14.5: Store invalid relationships in validation queue
        """
        total_rels = len(relationships)
        logger.info(f"Loading {total_rels} relationships in batches of {self.batch_size}")
        
        # Validate relationships if requested
        if validate:
            logger.info("Validating relationships before loading...")
            validation_result = self.validator.validate_batch(relationships)
            
            # Store invalid relationships in validation queue
            if validation_result.invalid_relationships:
                self.validator.store_invalid_relationships(
                    validation_result.invalid_relationships
                )
                logger.warning(
                    f"Stored {validation_result.invalid_count} invalid relationships "
                    f"in validation queue for manual review"
                )
            
            # Only load valid relationships
            relationships_to_load = validation_result.valid_relationships
            logger.info(
                f"Validation complete: {validation_result.valid_count} valid, "
                f"{validation_result.invalid_count} invalid"
            )
        else:
            relationships_to_load = relationships
            validation_result = None
        
        # Load valid relationships
        loaded_count = 0
        with self.driver.session() as session:
            for i in range(0, len(relationships_to_load), self.batch_size):
                batch = relationships_to_load[i:i + self.batch_size]
                
                with session.begin_transaction() as tx:
                    for rel in batch:
                        rel_props = self._build_relationship_properties(rel)
                        
                        query = f"""
                        MATCH (source {{id: $source_id}})
                        MATCH (target {{id: $target_id}})
                        MERGE (source)-[r:{rel.relation_type.value}]->(target)
                        SET r += $props
                        """
                        
                        tx.run(
                            query,
                            source_id=rel.source_entity,
                            target_id=rel.target_entity,
                            props=rel_props
                        )
                        loaded_count += 1
                    
                    tx.commit()
                
                logger.info(
                    f"Loaded batch {i // self.batch_size + 1}: "
                    f"{min(i + self.batch_size, len(relationships_to_load))}/{len(relationships_to_load)} relationships"
                )
        
        # Return statistics
        result = {
            "loaded": loaded_count,
            "invalid": validation_result.invalid_count if validation_result else 0,
            "total": total_rels,
        }
        
        logger.info(
            f"Relationship loading complete: {result['loaded']} loaded, "
            f"{result['invalid']} invalid, {result['total']} total"
        )
        
        return result
    
    def load_claims_batch(
        self,
        claims: List[ScientificClaim]
    ) -> None:
        """
        Load reified scientific claims in batches for performance.
        
        This method creates ScientificClaim nodes with all consensus metrics
        as properties and creates SUPPORTED_BY and CONTRADICTED_BY relationships
        to papers.
        
        Args:
            claims: List of ScientificClaim objects
        
        Requirements:
        - 4.1: Create ScientificClaim nodes with consensus metrics
        - 4.2: Create relationships between claims and supporting/contradicting papers
        - 4.3: Store consensus metrics as node properties
        - 17.5: Batch loading with 10,000 nodes/edges per transaction
        """
        if not claims:
            logger.warning("No claims to load")
            return
        
        total_claims = len(claims)
        logger.info(f"Loading {total_claims} reified claims in batches of {self.batch_size}")
        
        with self.driver.session() as session:
            for i in range(0, total_claims, self.batch_size):
                batch = claims[i:i + self.batch_size]
                
                with session.begin_transaction() as tx:
                    for claim in batch:
                        # Create ScientificClaim node with all properties
                        # Requirement 4.1, 4.3: Store consensus metrics as node properties
                        tx.run(
                            """
                            CREATE (c:ScientificClaim {
                                claim_id: $claim_id,
                                claim_type: $claim_type,
                                subject_entity: $subject_entity,
                                predicate: $predicate,
                                object_entity: $object_entity,
                                evidence_strength: $evidence_strength,
                                consensus_confidence: $consensus_confidence,
                                effect_direction_consistency: $effect_direction_consistency,
                                total_sample_size: $total_sample_size,
                                first_reported: $first_reported,
                                last_updated: $last_updated,
                                supporting_paper_count: $supporting_paper_count,
                                contradicting_paper_count: $contradicting_paper_count,
                                pooled_effect_size: $pooled_effect_size,
                                effect_size_variance: $effect_size_variance,
                                meta_analysis_performed: $meta_analysis_performed
                            })
                            """,
                            claim_id=claim.claim_id,
                            claim_type=claim.claim_type,
                            subject_entity=claim.subject_entity,
                            predicate=claim.predicate,
                            object_entity=claim.object_entity,
                            evidence_strength=claim.evidence_strength,
                            consensus_confidence=claim.consensus_confidence,
                            effect_direction_consistency=claim.effect_direction_consistency,
                            total_sample_size=claim.total_sample_size,
                            first_reported=claim.first_reported,
                            last_updated=claim.last_updated,
                            supporting_paper_count=len(claim.supporting_papers),
                            contradicting_paper_count=len(claim.contradicting_papers),
                            pooled_effect_size=claim.pooled_effect_size,
                            effect_size_variance=claim.effect_size_variance,
                            meta_analysis_performed=claim.meta_analysis_performed
                        )
                        
                        # Requirement 4.2: Create SUPPORTED_BY relationships
                        for paper_id in claim.supporting_papers:
                            tx.run(
                                """
                                MATCH (c:ScientificClaim {claim_id: $claim_id})
                                MERGE (p:Paper {id: $paper_id})
                                CREATE (c)-[:SUPPORTED_BY]->(p)
                                """,
                                claim_id=claim.claim_id,
                                paper_id=paper_id
                            )
                        
                        # Requirement 4.2: Create CONTRADICTED_BY relationships
                        for paper_id in claim.contradicting_papers:
                            tx.run(
                                """
                                MATCH (c:ScientificClaim {claim_id: $claim_id})
                                MERGE (p:Paper {id: $paper_id})
                                CREATE (c)-[:CONTRADICTED_BY]->(p)
                                """,
                                claim_id=claim.claim_id,
                                paper_id=paper_id
                            )
                    
                    tx.commit()
                
                logger.info(
                    f"Loaded batch {i // self.batch_size + 1}: "
                    f"{min(i + self.batch_size, total_claims)}/{total_claims} claims"
                )
        
        logger.info(f"Successfully loaded {total_claims} reified claims")
    
    def load_graph(
        self,
        nodes: Dict[str, List[Dict[str, Any]]],
        relationships: List[SemanticRelationship],
        validate_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Load complete graph with nodes and relationships.
        
        This is the main entry point for loading a knowledge graph into Neo4j.
        It handles both nodes (grouped by type) and relationships with full
        provenance and semantic properties.
        
        Args:
            nodes: Dictionary mapping node_type to list of node dictionaries
                   Example: {"Paper": [...], "Taxon": [...], "Disease": [...]}
            relationships: List of SemanticRelationship objects
            validate_relationships: Whether to validate relationships before loading (default: True)
        
        Returns:
            Dictionary with loading statistics
        
        Requirements:
        - 12.5: Support node types: Paper, Taxon, Disease, Method, ScientificClaim
        - 17.5: Batch loading with 10,000 nodes/edges per transaction
        - 14.1, 14.2, 14.3, 14.4, 14.5: Validate data before loading
        """
        logger.info("Starting graph load")
        
        stats = {
            "nodes_loaded": 0,
            "relationships_loaded": 0,
            "relationships_invalid": 0,
            "relationships_total": 0,
        }
        
        # Load nodes by type
        for node_type, node_list in nodes.items():
            if node_list:
                self.load_nodes_batch(node_list, node_type)
                stats["nodes_loaded"] += len(node_list)
        
        # Load relationships with validation
        if relationships:
            rel_stats = self.load_relationships_batch(
                relationships,
                validate=validate_relationships
            )
            stats["relationships_loaded"] = rel_stats["loaded"]
            stats["relationships_invalid"] = rel_stats["invalid"]
            stats["relationships_total"] = rel_stats["total"]
        
        logger.info(
            f"Graph load completed: {stats['nodes_loaded']} nodes, "
            f"{stats['relationships_loaded']} relationships loaded, "
            f"{stats['relationships_invalid']} relationships invalid"
        )
        
        return stats
    
    def create_indexes(self) -> None:
        """
        Create indexes for efficient querying.
        
        This creates indexes on commonly queried properties to support
        the research query patterns defined in the design.
        
        Requirement 12.5: Create indexes on paper, entity, and relationship properties
        """
        logger.info("Creating Neo4j indexes")
        
        with self.driver.session() as session:
            # Paper property indexes (Requirement 12.1)
            indexes = [
                "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
                "CREATE INDEX paper_article_type IF NOT EXISTS FOR (p:Paper) ON (p.article_type)",
                "CREATE INDEX paper_data_availability IF NOT EXISTS FOR (p:Paper) ON (p.data_availability)",
                
                # Composite index for common query patterns (Requirement 12.4)
                "CREATE INDEX paper_year_article_type_composite IF NOT EXISTS FOR (p:Paper) ON (p.year, p.article_type)",
                
                # Entity property indexes (Requirement 12.2)
                "CREATE INDEX taxon_name IF NOT EXISTS FOR (t:Taxon) ON (t.name)",
                "CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)",
                "CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)",
                
                # Entity canonical identifiers
                "CREATE INDEX taxon_ncbi_id IF NOT EXISTS FOR (t:Taxon) ON (t.ncbi_id)",
                "CREATE INDEX disease_mesh_id IF NOT EXISTS FOR (d:Disease) ON (d.mesh_id)",
                
                # Relationship property indexes (Requirement 12.3)
                # Individual relationship property indexes for REPORTS_ASSOCIATION
                "CREATE INDEX rel_association_confidence IF NOT EXISTS FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.confidence)",
                "CREATE INDEX rel_association_p_value IF NOT EXISTS FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.p_value)",
                
                # Individual relationship property indexes for REPORTS_INTERVENTION_EFFECT
                "CREATE INDEX rel_intervention_confidence IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.confidence)",
                "CREATE INDEX rel_intervention_p_value IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.p_value)",
                "CREATE INDEX rel_intervention_type IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.intervention_type)",
                
                # Individual relationship property indexes for USES_METHODOLOGY
                "CREATE INDEX rel_methodology_confidence IF NOT EXISTS FOR ()-[r:USES_METHODOLOGY]-() ON (r.confidence)",
                
                # Composite index on (evidence_strength, consensus_confidence) for REPORTS_ASSOCIATION (Requirement 12.4)
                "CREATE INDEX rel_association_evidence_consensus_composite IF NOT EXISTS FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.evidence_strength, r.consensus_confidence)",
                
                # Composite index on (evidence_strength, consensus_confidence) for REPORTS_INTERVENTION_EFFECT (Requirement 12.4)
                "CREATE INDEX rel_intervention_evidence_consensus_composite IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.evidence_strength, r.consensus_confidence)",
            ]
            
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.info(f"Created index: {index_query}")
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
        
        logger.info("Index creation completed")
    
    def clear_database(self) -> None:
        """
        Clear all nodes and relationships from the database.
        
        WARNING: This is a destructive operation. Use with caution.
        """
        logger.warning("Clearing Neo4j database")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.info("Database cleared")


# Convenience function for loading from enhanced graph builder output

def load_enhanced_graph(
    uri: str,
    user: str,
    password: str,
    nodes: Dict[str, List[Dict[str, Any]]],
    relationships: List[SemanticRelationship],
    create_indexes: bool = True,
    batch_size: int = 10000,
    validate_relationships: bool = True,
    validation_queue_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load an enhanced knowledge graph into Neo4j.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        nodes: Dictionary mapping node_type to list of node dictionaries
        relationships: List of SemanticRelationship objects
        create_indexes: Whether to create indexes after loading (default: True)
        batch_size: Batch size for loading (default: 10,000)
        validate_relationships: Whether to validate relationships before loading (default: True)
        validation_queue_path: Path to store invalid relationships (default: data/validation_queue.json)
    
    Returns:
        Dictionary with loading statistics
    
    Requirements:
    - 12.5: Graph schema and indexes
    - 17.5: Batch loading with 10,000 nodes/edges per transaction
    - 14.1, 14.2, 14.3, 14.4, 14.5: Validate data before loading
    """
    with EnhancedNeo4jLoader(uri, user, password, batch_size, validation_queue_path) as loader:
        # Load graph data with validation
        stats = loader.load_graph(nodes, relationships, validate_relationships)
        
        # Create indexes for efficient querying
        if create_indexes:
            loader.create_indexes()
        
        return stats
