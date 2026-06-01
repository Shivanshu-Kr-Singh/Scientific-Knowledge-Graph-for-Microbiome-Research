#!/usr/bin/env python3
"""
scripts/migrate_to_enhanced_schema.py
--------------------------------------
Migration script for transitioning from old Neo4j schema to enhanced schema.

This script:
1. Reads existing relationships from old Neo4j database
2. Adds provenance metadata retroactively where possible
3. Marks relationships without provenance as "legacy"
4. Verifies >= 90% of entities from old system are extracted by new system

Requirements: 16.2, 16.3
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from neo4j import GraphDatabase
from collections import defaultdict
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.semantic_relationships import SemanticRelationship, RelationType
from graph.provenance import ProvenanceMetadata
from graph.enhanced_neo4j_loader import EnhancedNeo4jLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationStats:
    """Statistics for migration process."""
    
    def __init__(self):
        self.old_nodes_count: Dict[str, int] = defaultdict(int)
        self.old_relationships_count: Dict[str, int] = defaultdict(int)
        self.new_nodes_count: Dict[str, int] = defaultdict(int)
        self.new_relationships_count: Dict[str, int] = defaultdict(int)
        self.legacy_relationships: int = 0
        self.provenance_added: int = 0
        self.entities_matched: int = 0
        self.entities_missing: int = 0
        self.entity_match_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "old_system": {
                "nodes": dict(self.old_nodes_count),
                "relationships": dict(self.old_relationships_count),
                "total_nodes": sum(self.old_nodes_count.values()),
                "total_relationships": sum(self.old_relationships_count.values()),
            },
            "new_system": {
                "nodes": dict(self.new_nodes_count),
                "relationships": dict(self.new_relationships_count),
                "total_nodes": sum(self.new_nodes_count.values()),
                "total_relationships": sum(self.new_relationships_count.values()),
            },
            "migration": {
                "legacy_relationships": self.legacy_relationships,
                "provenance_added": self.provenance_added,
                "entities_matched": self.entities_matched,
                "entities_missing": self.entities_missing,
                "entity_match_percentage": self.entity_match_percentage,
            }
        }
    
    def print_summary(self):
        """Print migration summary."""
        logger.info("=" * 80)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Old System:")
        logger.info(f"  Total Nodes: {sum(self.old_nodes_count.values())}")
        for node_type, count in self.old_nodes_count.items():
            logger.info(f"    {node_type}: {count}")
        logger.info(f"  Total Relationships: {sum(self.old_relationships_count.values())}")
        for rel_type, count in self.old_relationships_count.items():
            logger.info(f"    {rel_type}: {count}")
        
        logger.info(f"\nNew System:")
        logger.info(f"  Total Nodes: {sum(self.new_nodes_count.values())}")
        for node_type, count in self.new_nodes_count.items():
            logger.info(f"    {node_type}: {count}")
        logger.info(f"  Total Relationships: {sum(self.new_relationships_count.values())}")
        for rel_type, count in self.new_relationships_count.items():
            logger.info(f"    {rel_type}: {count}")
        
        logger.info(f"\nMigration:")
        logger.info(f"  Legacy Relationships: {self.legacy_relationships}")
        logger.info(f"  Provenance Added: {self.provenance_added}")
        logger.info(f"  Entities Matched: {self.entities_matched}")
        logger.info(f"  Entities Missing: {self.entities_missing}")
        logger.info(f"  Entity Match Percentage: {self.entity_match_percentage:.2f}%")
        logger.info("=" * 80)


class SchemaEnhancementMigrator:
    """
    Migrates from old Neo4j schema to enhanced schema with provenance.
    
    Requirements: 16.2, 16.3
    """
    
    def __init__(
        self,
        old_uri: str,
        old_user: str,
        old_password: str,
        new_uri: str,
        new_user: str,
        new_password: str,
        batch_size: int = 1000
    ):
        """
        Initialize the migrator.
        
        Args:
            old_uri: URI for old Neo4j database
            old_user: Username for old database
            old_password: Password for old database
            new_uri: URI for new Neo4j database
            new_user: Username for new database
            new_password: Password for new database
            batch_size: Batch size for migration
        """
        self.old_driver = GraphDatabase.driver(old_uri, auth=(old_user, old_password))
        self.new_loader = EnhancedNeo4jLoader(new_uri, new_user, new_password, batch_size)
        self.stats = MigrationStats()
        self.batch_size = batch_size
        
        logger.info(f"Initialized migrator with batch_size={batch_size}")
    
    def close(self):
        """Close database connections."""
        self.old_driver.close()
        self.new_loader.close()
        logger.info("Closed database connections")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def read_old_nodes(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read all nodes from old Neo4j database.
        
        Returns:
            Dictionary mapping node type to list of node dictionaries
        """
        logger.info("Reading nodes from old database...")
        
        nodes_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        with self.old_driver.session() as session:
            # Get all node labels
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            
            logger.info(f"Found node labels: {labels}")
            
            # Read nodes for each label
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN n")
                
                for record in result:
                    node = record["n"]
                    node_dict = dict(node)
                    
                    # Ensure 'id' field exists
                    if 'id' not in node_dict:
                        # Use node element ID as fallback
                        node_dict['id'] = node.element_id
                    
                    nodes_by_type[label].append(node_dict)
                    self.stats.old_nodes_count[label] += 1
                
                logger.info(f"Read {self.stats.old_nodes_count[label]} {label} nodes")
        
        return nodes_by_type
    
    def read_old_relationships(self) -> List[Dict[str, Any]]:
        """
        Read all relationships from old Neo4j database.
        
        Returns:
            List of relationship dictionaries with source, target, type, and properties
        """
        logger.info("Reading relationships from old database...")
        
        relationships = []
        
        with self.old_driver.session() as session:
            # Get all relationship types
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            
            logger.info(f"Found relationship types: {rel_types}")
            
            # Read relationships for each type
            for rel_type in rel_types:
                result = session.run(
                    f"""
                    MATCH (source)-[r:{rel_type}]->(target)
                    RETURN source.id as source_id, target.id as target_id,
                           type(r) as rel_type, properties(r) as props,
                           labels(source) as source_labels, labels(target) as target_labels
                    """
                )
                
                for record in result:
                    rel_dict = {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "rel_type": record["rel_type"],
                        "properties": dict(record["props"]) if record["props"] else {},
                        "source_labels": record["source_labels"],
                        "target_labels": record["target_labels"],
                    }
                    relationships.append(rel_dict)
                    self.stats.old_relationships_count[rel_type] += 1
                
                logger.info(f"Read {self.stats.old_relationships_count[rel_type]} {rel_type} relationships")
        
        return relationships
    
    def create_legacy_provenance(
        self,
        paper_id: str,
        relationship_type: str
    ) -> ProvenanceMetadata:
        """
        Create legacy provenance metadata for relationships without provenance.
        
        Requirement 16.3: Mark relationships without provenance as "legacy"
        
        Args:
            paper_id: Paper identifier
            relationship_type: Type of relationship
        
        Returns:
            ProvenanceMetadata marked as legacy
        """
        return ProvenanceMetadata(
            paper_id=paper_id,
            section_type="other",  # Use "other" instead of "unknown"
            source_sentence="[LEGACY] Migrated from old system without source sentence",
            sentence_offset=None,
            extraction_method="legacy",  # Use registered "legacy" method
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            llm_prompt_hash=None,
            confidence_score=0.5,  # Default confidence for legacy data
            validation_status="unvalidated",  # Use "unvalidated" instead of "legacy"
            validator_id=None,
            surrounding_context=None,
            figure_table_ref=None,
        )
    
    def enhance_relationship_with_provenance(
        self,
        old_rel: Dict[str, Any]
    ) -> Optional[SemanticRelationship]:
        """
        Enhance old relationship with provenance metadata.
        
        This method attempts to add provenance metadata to existing relationships.
        If provenance cannot be determined, marks the relationship as "legacy".
        
        Requirement 16.3: Add provenance metadata retroactively where possible
        
        Args:
            old_rel: Old relationship dictionary
        
        Returns:
            SemanticRelationship with provenance, or None if cannot be converted
        """
        # Extract relationship information
        source_id = old_rel["source_id"]
        target_id = old_rel["target_id"]
        rel_type = old_rel["rel_type"]
        props = old_rel["properties"]
        
        # Determine paper ID (assume source is paper for most relationships)
        paper_id = source_id if "Paper" in old_rel["source_labels"] else "unknown"
        
        # Check if relationship already has provenance metadata
        has_provenance = all(
            key in props for key in [
                "section", "source_sentence", "extraction_method"
            ]
        )
        
        if has_provenance:
            # Use existing provenance
            provenance = ProvenanceMetadata(
                paper_id=paper_id,
                section_type=props.get("section", "unknown"),
                source_sentence=props.get("source_sentence", ""),
                sentence_offset=props.get("sentence_offset"),
                extraction_method=props.get("extraction_method", "unknown"),
                extraction_timestamp=datetime.fromisoformat(props["extraction_timestamp"])
                    if "extraction_timestamp" in props
                    else datetime.now(timezone.utc),
                extractor_version=props.get("extractor_version", "1.0"),
                llm_prompt_hash=props.get("llm_prompt_hash"),
                confidence_score=props.get("confidence", 0.5),
                validation_status=props.get("validation_status", "unvalidated"),
                validator_id=props.get("validator_id"),
                surrounding_context=props.get("surrounding_context"),
                figure_table_ref=props.get("figure_table_ref"),
            )
            self.stats.provenance_added += 1
        else:
            # Create legacy provenance
            provenance = self.create_legacy_provenance(paper_id, rel_type)
            self.stats.legacy_relationships += 1
        
        # Map old relationship type to new RelationType
        relation_type = self._map_relationship_type(rel_type)
        if relation_type is None:
            logger.warning(f"Cannot map relationship type: {rel_type}")
            return None
        
        # Extract semantic properties
        semantic_props = self._extract_semantic_properties(props, relation_type)
        
        # Determine evidence strength
        evidence_strength = self._determine_evidence_strength(props)
        
        # Create SemanticRelationship
        try:
            semantic_rel = SemanticRelationship(
                source_entity=source_id,
                target_entity=target_id,
                relation_type=relation_type,
                properties=semantic_props,
                provenance=provenance,
                evidence_strength=evidence_strength,
                extraction_confidence=provenance.confidence_score,
            )
            return semantic_rel
        except Exception as e:
            logger.error(f"Failed to create SemanticRelationship: {e}")
            return None
    
    def _map_relationship_type(self, old_type: str) -> Optional[RelationType]:
        """
        Map old relationship type to new RelationType enum.
        
        Args:
            old_type: Old relationship type string
        
        Returns:
            RelationType enum value, or None if cannot be mapped
        """
        # Map common relationship types
        type_mapping = {
            "HAS_TAXON": RelationType.REPORTS_ASSOCIATION,
            "HAS_DISEASE": RelationType.REPORTS_ASSOCIATION,
            "ASSOCIATED_WITH": RelationType.REPORTS_ASSOCIATION,
            "REPORTS_ASSOCIATION": RelationType.REPORTS_ASSOCIATION,
            "INTERVENTION_EFFECT": RelationType.REPORTS_INTERVENTION_EFFECT,
            "REPORTS_INTERVENTION_EFFECT": RelationType.REPORTS_INTERVENTION_EFFECT,
            "USES_METHOD": RelationType.USES_METHODOLOGY,
            "USES_METHODOLOGY": RelationType.USES_METHODOLOGY,
            "CANDIDATE_REL": RelationType.REPORTS_ASSOCIATION,  # Treat candidate as association
        }
        
        return type_mapping.get(old_type)
    
    def _extract_semantic_properties(
        self,
        props: Dict[str, Any],
        relation_type: RelationType
    ) -> Dict[str, Any]:
        """
        Extract semantic properties from old relationship properties.
        
        Args:
            props: Old relationship properties
            relation_type: New relationship type
        
        Returns:
            Dictionary of semantic properties (excludes None values)
        """
        semantic_props = {}
        
        if relation_type == RelationType.REPORTS_ASSOCIATION:
            # Extract association properties
            semantic_props["direction"] = props.get("direction", "no_change")
            semantic_props["comparison"] = props.get("comparison", "unknown")
            semantic_props["statistical_measure"] = props.get("statistical_measure", "unknown")
            
            # Only include optional fields if they have non-None values
            if "effect_size" in props and props["effect_size"] is not None:
                semantic_props["effect_size"] = props["effect_size"]
            if "p_value" in props and props["p_value"] is not None:
                semantic_props["p_value"] = props["p_value"]
            if "adjusted_p_value" in props and props["adjusted_p_value"] is not None:
                semantic_props["adjusted_p_value"] = props["adjusted_p_value"]
        
        elif relation_type == RelationType.REPORTS_INTERVENTION_EFFECT:
            # Extract intervention properties
            semantic_props["intervention_type"] = props.get("intervention_type", "other")
            semantic_props["effect_direction"] = props.get("effect_direction", "no_change")
            
            # Only include optional fields if they have non-None values
            if "duration" in props and props["duration"] is not None:
                semantic_props["duration"] = props["duration"]
            if "dosage" in props and props["dosage"] is not None:
                semantic_props["dosage"] = props["dosage"]
            if "sample_size" in props and props["sample_size"] is not None:
                semantic_props["sample_size"] = props["sample_size"]
        
        elif relation_type == RelationType.USES_METHODOLOGY:
            # Extract methodology properties
            semantic_props["method_name"] = props.get("method_name", "unknown")
            
            # Only include optional fields if they have non-None values
            if "sequencing_platform" in props and props["sequencing_platform"] is not None:
                semantic_props["sequencing_platform"] = props["sequencing_platform"]
            if "sample_size" in props and props["sample_size"] is not None:
                semantic_props["sample_size"] = props["sample_size"]
        
        return semantic_props
    
    def _determine_evidence_strength(self, props: Dict[str, Any]) -> str:
        """
        Determine evidence strength from relationship properties.
        
        Args:
            props: Relationship properties
        
        Returns:
            Evidence strength: "strong" | "moderate" | "weak"
        """
        p_value = props.get("p_value")
        
        if p_value is not None:
            if p_value < 0.01:
                return "strong"
            elif p_value < 0.05:
                return "moderate"
            else:
                return "weak"
        
        # Default to weak if no p-value
        return "weak"
    
    def migrate_nodes(self, old_nodes: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Migrate nodes from old database to new database.
        
        Args:
            old_nodes: Dictionary mapping node type to list of node dictionaries
        """
        logger.info("Migrating nodes to new database...")
        
        for node_type, nodes in old_nodes.items():
            if not nodes:
                continue
            
            logger.info(f"Migrating {len(nodes)} {node_type} nodes...")
            self.new_loader.load_nodes_batch(nodes, node_type)
            self.stats.new_nodes_count[node_type] = len(nodes)
    
    def migrate_relationships(self, old_relationships: List[Dict[str, Any]]) -> None:
        """
        Migrate relationships from old database to new database with enhanced provenance.
        
        Args:
            old_relationships: List of old relationship dictionaries
        """
        logger.info("Migrating relationships to new database...")
        
        enhanced_relationships = []
        
        for old_rel in old_relationships:
            semantic_rel = self.enhance_relationship_with_provenance(old_rel)
            if semantic_rel:
                enhanced_relationships.append(semantic_rel)
        
        logger.info(f"Enhanced {len(enhanced_relationships)} relationships with provenance")
        
        # Load enhanced relationships in batches
        if enhanced_relationships:
            result = self.new_loader.load_relationships_batch(
                enhanced_relationships,
                validate=True
            )
            
            for rel in enhanced_relationships:
                self.stats.new_relationships_count[rel.relation_type.value] += 1
            
            logger.info(
                f"Loaded {result['loaded']} relationships, "
                f"{result['invalid']} invalid"
            )
    
    def verify_entity_extraction(
        self,
        old_nodes: Dict[str, List[Dict[str, Any]]],
        new_nodes: Dict[str, List[Dict[str, Any]]]
    ) -> bool:
        """
        Verify that >= 90% of entities from old system are extracted by new system.
        
        Requirement 16.2: Verify >= 90% of entities from old system are extracted
        
        Args:
            old_nodes: Nodes from old database
            new_nodes: Nodes from new database
        
        Returns:
            True if >= 90% of entities are matched, False otherwise
        """
        logger.info("Verifying entity extraction...")
        
        # Count entities by type
        for node_type in old_nodes.keys():
            old_entities = set(node["id"] for node in old_nodes.get(node_type, []))
            new_entities = set(node["id"] for node in new_nodes.get(node_type, []))
            
            matched = old_entities.intersection(new_entities)
            missing = old_entities - new_entities
            
            self.stats.entities_matched += len(matched)
            self.stats.entities_missing += len(missing)
            
            match_percentage = (len(matched) / len(old_entities) * 100) if old_entities else 0
            
            logger.info(
                f"{node_type}: {len(matched)}/{len(old_entities)} matched "
                f"({match_percentage:.2f}%)"
            )
            
            if missing:
                logger.warning(f"Missing {len(missing)} {node_type} entities: {list(missing)[:10]}")
        
        # Calculate overall match percentage
        total_old_entities = sum(len(nodes) for nodes in old_nodes.values())
        if total_old_entities > 0:
            self.stats.entity_match_percentage = (
                self.stats.entities_matched / total_old_entities * 100
            )
        
        logger.info(
            f"Overall entity match: {self.stats.entities_matched}/{total_old_entities} "
            f"({self.stats.entity_match_percentage:.2f}%)"
        )
        
        # Requirement 16.2: Verify >= 90%
        if self.stats.entity_match_percentage >= 90.0:
            logger.info("✓ Entity extraction verification PASSED (>= 90%)")
            return True
        else:
            logger.error(
                f"✗ Entity extraction verification FAILED "
                f"({self.stats.entity_match_percentage:.2f}% < 90%)"
            )
            return False
    
    def run_migration(self, verify_entities: bool = True) -> MigrationStats:
        """
        Run the complete migration process.
        
        This method:
        1. Reads nodes and relationships from old database
        2. Migrates nodes to new database
        3. Enhances relationships with provenance metadata
        4. Migrates enhanced relationships to new database
        5. Verifies entity extraction (optional)
        
        Requirements: 16.2, 16.3
        
        Args:
            verify_entities: Whether to verify entity extraction (default: True)
        
        Returns:
            MigrationStats with migration statistics
        """
        logger.info("Starting migration process...")
        
        # Step 1: Read old database
        old_nodes = self.read_old_nodes()
        old_relationships = self.read_old_relationships()
        
        # Step 2: Migrate nodes
        self.migrate_nodes(old_nodes)
        
        # Step 3: Migrate relationships with enhanced provenance
        self.migrate_relationships(old_relationships)
        
        # Step 4: Create indexes
        logger.info("Creating indexes in new database...")
        self.new_loader.create_indexes()
        
        # Step 5: Verify entity extraction
        if verify_entities:
            new_nodes = self.read_new_nodes()
            verification_passed = self.verify_entity_extraction(old_nodes, new_nodes)
            
            if not verification_passed:
                logger.warning(
                    "Entity extraction verification failed. "
                    "Consider investigating missing entities."
                )
        
        # Print summary
        self.stats.print_summary()
        
        logger.info("Migration completed successfully!")
        
        return self.stats
    
    def read_new_nodes(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read all nodes from new Neo4j database for verification.
        
        Returns:
            Dictionary mapping node type to list of node dictionaries
        """
        logger.info("Reading nodes from new database for verification...")
        
        nodes_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        with self.new_loader.driver.session() as session:
            # Get all node labels
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            
            # Read nodes for each label
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN n.id as id")
                
                for record in result:
                    nodes_by_type[label].append({"id": record["id"]})
        
        return nodes_by_type
    
    def save_migration_report(self, output_path: str) -> None:
        """
        Save migration statistics to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        report = {
            "migration_timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": self.stats.to_dict(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Migration report saved to {output_path}")


def main():
    """Main entry point for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate from old Neo4j schema to enhanced schema with provenance"
    )
    parser.add_argument(
        "--old-uri",
        default=os.getenv("NEO4J_OLD_URI", "bolt://localhost:7687"),
        help="URI for old Neo4j database"
    )
    parser.add_argument(
        "--old-user",
        default=os.getenv("NEO4J_OLD_USER", "neo4j"),
        help="Username for old database"
    )
    parser.add_argument(
        "--old-password",
        default=os.getenv("NEO4J_OLD_PASSWORD", "your_password"),
        help="Password for old database"
    )
    parser.add_argument(
        "--new-uri",
        default=os.getenv("NEO4J_NEW_URI", "bolt://localhost:7688"),
        help="URI for new Neo4j database"
    )
    parser.add_argument(
        "--new-user",
        default=os.getenv("NEO4J_NEW_USER", "neo4j"),
        help="Username for new database"
    )
    parser.add_argument(
        "--new-password",
        default=os.getenv("NEO4J_NEW_PASSWORD", "your_password"),
        help="Password for new database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for migration (default: 1000)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip entity extraction verification"
    )
    parser.add_argument(
        "--report",
        default="data/migration_report.json",
        help="Path to save migration report (default: data/migration_report.json)"
    )
    
    args = parser.parse_args()
    
    # Run migration
    with SchemaEnhancementMigrator(
        old_uri=args.old_uri,
        old_user=args.old_user,
        old_password=args.old_password,
        new_uri=args.new_uri,
        new_user=args.new_user,
        new_password=args.new_password,
        batch_size=args.batch_size
    ) as migrator:
        stats = migrator.run_migration(verify_entities=not args.no_verify)
        migrator.save_migration_report(args.report)
    
    # Exit with appropriate code
    if stats.entity_match_percentage >= 90.0:
        logger.info("Migration completed successfully with >= 90% entity match")
        sys.exit(0)
    else:
        logger.error(f"Migration completed but entity match < 90% ({stats.entity_match_percentage:.2f}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
