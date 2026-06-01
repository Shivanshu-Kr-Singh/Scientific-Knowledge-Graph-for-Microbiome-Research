"""
graph/rollback_manager.py
--------------------------
Rollback functionality for knowledge graph extractions.

This module implements rollback capabilities to remove all extractions
created by a specific extraction method version. This is useful for
removing problematic extractions or reverting to a previous state.

Requirements: 10.5, 19.4
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import logging

from graph.audit_log import get_audit_log, AuditLog

logger = logging.getLogger(__name__)


class RollbackManager:
    """
    Manager for rolling back extractions by method version.
    
    This class provides functionality to remove all relationships and nodes
    created by a specific extraction method version, using the audit log
    to identify what needs to be removed.
    
    Requirements:
    - 10.5: Support rollback of extractions by method version
    - 19.4: Support rollback of extractions by method version
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        audit_log: Optional[AuditLog] = None,
    ):
        """
        Initialize the rollback manager.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            audit_log: AuditLog instance (default: global instance)
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.audit_log = audit_log or get_audit_log()
        logger.info("Initialized RollbackManager")
    
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
    
    def get_relationships_to_rollback(
        self,
        extraction_method: str,
        extractor_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships that would be removed by a rollback.
        
        This is a dry-run operation that shows what would be removed
        without actually removing anything.
        
        Args:
            extraction_method: Extraction method identifier
            extractor_version: Version of extraction method (optional)
        
        Returns:
            List of relationship dictionaries
        
        Requirement 19.3: Query relationships by extraction method version
        """
        relationships = self.audit_log.get_relationships_by_method_version(
            extraction_method,
            extractor_version
        )
        
        logger.info(
            f"Found {len(relationships)} relationships to rollback for "
            f"extraction_method={extraction_method}, version={extractor_version}"
        )
        
        return relationships
    
    def rollback_by_method_version(
        self,
        extraction_method: str,
        extractor_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Rollback all extractions by a specific method version.
        
        This removes all relationships created by the specified extraction
        method version from the Neo4j database. The audit log entries are
        preserved for historical tracking.
        
        Args:
            extraction_method: Extraction method identifier
            extractor_version: Version of extraction method (optional)
            dry_run: If True, only report what would be removed without actually removing
        
        Returns:
            Dictionary with rollback statistics
        
        Requirements:
        - 10.5: Support rollback of extractions by method version
        - 19.4: Support rollback of extractions by method version
        """
        # Get relationships to rollback from audit log
        relationships = self.get_relationships_to_rollback(
            extraction_method,
            extractor_version
        )
        
        if not relationships:
            logger.warning(
                f"No relationships found for extraction_method={extraction_method}, "
                f"version={extractor_version}"
            )
            return {
                "relationships_removed": 0,
                "dry_run": dry_run,
                "extraction_method": extraction_method,
                "extractor_version": extractor_version,
            }
        
        if dry_run:
            logger.info(
                f"DRY RUN: Would remove {len(relationships)} relationships for "
                f"extraction_method={extraction_method}, version={extractor_version}"
            )
            return {
                "relationships_removed": len(relationships),
                "dry_run": True,
                "extraction_method": extraction_method,
                "extractor_version": extractor_version,
                "relationships": relationships,
            }
        
        # Remove relationships from Neo4j
        removed_count = 0
        with self.driver.session() as session:
            for rel in relationships:
                try:
                    # Delete relationship by source, target, and type
                    query = f"""
                    MATCH (source {{id: $source_id}})-[r:{rel['relationship_type']}]->(target {{id: $target_id}})
                    WHERE r.extraction_method = $extraction_method
                    """
                    
                    if extractor_version:
                        query += " AND r.extractor_version = $extractor_version"
                    
                    query += " DELETE r"
                    
                    params = {
                        "source_id": rel["source_id"],
                        "target_id": rel["target_id"],
                        "extraction_method": extraction_method,
                    }
                    
                    if extractor_version:
                        params["extractor_version"] = extractor_version
                    
                    result = session.run(query, params)
                    
                    # Log the deletion to audit log
                    self.audit_log.log_modification(
                        operation_type="delete_edge",
                        entity_type=rel["relationship_type"],
                        entity_id=f"{rel['source_id']}->{rel['target_id']}",
                        user_id="rollback_manager",
                        modification_details={
                            "reason": "rollback",
                            "extraction_method": extraction_method,
                            "extractor_version": extractor_version,
                        },
                    )
                    
                    removed_count += 1
                    
                except Exception as e:
                    logger.error(
                        f"Failed to remove relationship {rel['source_id']}->{rel['target_id']}: {e}"
                    )
        
        logger.info(
            f"Rollback complete: Removed {removed_count} relationships for "
            f"extraction_method={extraction_method}, version={extractor_version}"
        )
        
        return {
            "relationships_removed": removed_count,
            "dry_run": False,
            "extraction_method": extraction_method,
            "extractor_version": extractor_version,
        }
    
    def rollback_by_paper(
        self,
        paper_id: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Rollback all extractions from a specific paper.
        
        This removes all relationships extracted from the specified paper
        from the Neo4j database.
        
        Args:
            paper_id: Paper ID to rollback
            dry_run: If True, only report what would be removed without actually removing
        
        Returns:
            Dictionary with rollback statistics
        """
        # Query audit log for all extractions from this paper
        with self.audit_log._lock:
            import sqlite3
            with sqlite3.connect(self.audit_log.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT entity_type, entity_id, modification_details
                    FROM audit_log
                    WHERE paper_id = ? AND operation_type = 'create_edge'
                    ORDER BY timestamp DESC
                """, (paper_id,))
                rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"No relationships found for paper_id={paper_id}")
            return {
                "relationships_removed": 0,
                "dry_run": dry_run,
                "paper_id": paper_id,
            }
        
        if dry_run:
            logger.info(
                f"DRY RUN: Would remove {len(rows)} relationships for paper_id={paper_id}"
            )
            return {
                "relationships_removed": len(rows),
                "dry_run": True,
                "paper_id": paper_id,
            }
        
        # Remove relationships from Neo4j
        removed_count = 0
        with self.driver.session() as session:
            for row in rows:
                try:
                    import json
                    entity_type = row[0]
                    modification_details = json.loads(row[2])
                    source_id = modification_details.get("source_id")
                    target_id = modification_details.get("target_id")
                    
                    if not source_id or not target_id:
                        continue
                    
                    # Delete relationship
                    query = f"""
                    MATCH (source {{id: $source_id}})-[r:{entity_type}]->(target {{id: $target_id}})
                    WHERE r.paper_id = $paper_id
                    DELETE r
                    """
                    
                    session.run(query, {
                        "source_id": source_id,
                        "target_id": target_id,
                        "paper_id": paper_id,
                    })
                    
                    # Log the deletion
                    self.audit_log.log_modification(
                        operation_type="delete_edge",
                        entity_type=entity_type,
                        entity_id=f"{source_id}->{target_id}",
                        user_id="rollback_manager",
                        modification_details={
                            "reason": "rollback_by_paper",
                            "paper_id": paper_id,
                        },
                    )
                    
                    removed_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to remove relationship: {e}")
        
        logger.info(
            f"Rollback complete: Removed {removed_count} relationships for paper_id={paper_id}"
        )
        
        return {
            "relationships_removed": removed_count,
            "dry_run": False,
            "paper_id": paper_id,
        }
    
    def get_extraction_method_statistics(
        self,
        extraction_method: str,
        extractor_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics about extractions by a specific method version.
        
        Args:
            extraction_method: Extraction method identifier
            extractor_version: Version of extraction method (optional)
        
        Returns:
            Dictionary with statistics
        """
        relationships = self.get_relationships_to_rollback(
            extraction_method,
            extractor_version
        )
        
        # Count by relationship type
        type_counts = {}
        for rel in relationships:
            rel_type = rel["relationship_type"]
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        
        # Count by paper
        paper_counts = {}
        for rel in relationships:
            paper_id = rel.get("paper_id")
            if paper_id:
                paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
        
        return {
            "total_relationships": len(relationships),
            "relationships_by_type": type_counts,
            "unique_papers": len(paper_counts),
            "relationships_by_paper": paper_counts,
            "extraction_method": extraction_method,
            "extractor_version": extractor_version,
        }


def rollback_extraction_method(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    extraction_method: str,
    extractor_version: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to rollback extractions by method version.
    
    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        extraction_method: Extraction method identifier
        extractor_version: Version of extraction method (optional)
        dry_run: If True, only report what would be removed without actually removing
    
    Returns:
        Dictionary with rollback statistics
    
    Requirements:
    - 10.5: Support rollback of extractions by method version
    - 19.4: Support rollback of extractions by method version
    """
    with RollbackManager(neo4j_uri, neo4j_user, neo4j_password) as manager:
        return manager.rollback_by_method_version(
            extraction_method,
            extractor_version,
            dry_run
        )
