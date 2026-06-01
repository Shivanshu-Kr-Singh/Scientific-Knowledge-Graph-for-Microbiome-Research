#!/usr/bin/env python3
"""
scripts/backup_neo4j.py
-----------------------
Backup script for Neo4j database before migration.

This script creates a snapshot of the old Neo4j database instance
before migration to the enhanced schema, enabling rollback if needed.

Requirements: 16.4
"""

import os
import sys
import logging
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import json
from neo4j import GraphDatabase

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Neo4jBackupManager:
    """
    Manages Neo4j database backups for migration rollback.
    
    Requirements: 16.4
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        backup_dir: str = "data/backups"
    ):
        """
        Initialize the backup manager.
        
        Args:
            neo4j_uri: URI for Neo4j database to backup
            neo4j_user: Username for database
            neo4j_password: Password for database
            backup_dir: Directory to store backups (default: data/backups)
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.backup_dir = Path(backup_dir)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized backup manager for {neo4j_uri}")
        logger.info(f"Backup directory: {self.backup_dir.absolute()}")
    
    def close(self):
        """Close database connection."""
        self.driver.close()
        logger.info("Closed database connection")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database to be backed up.
        
        Returns:
            Dictionary with node counts, relationship counts, and labels
        """
        logger.info("Gathering database statistics...")
        
        stats = {
            "nodes": {},
            "relationships": {},
            "total_nodes": 0,
            "total_relationships": 0,
        }
        
        with self.driver.session() as session:
            # Get node counts by label
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            
            for label in labels:
                count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = count_result.single()["count"]
                stats["nodes"][label] = count
                stats["total_nodes"] += count
            
            # Get relationship counts by type
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            
            for rel_type in rel_types:
                count_result = session.run(
                    f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                )
                count = count_result.single()["count"]
                stats["relationships"][rel_type] = count
                stats["total_relationships"] += count
        
        logger.info(f"Database contains {stats['total_nodes']} nodes and {stats['total_relationships']} relationships")
        
        return stats
    
    def export_to_cypher(self, backup_name: str) -> Path:
        """
        Export database to Cypher statements.
        
        This method exports all nodes and relationships as Cypher CREATE statements
        that can be used to restore the database.
        
        Args:
            backup_name: Name for this backup
        
        Returns:
            Path to the exported Cypher file
        """
        logger.info("Exporting database to Cypher statements...")
        
        cypher_file = self.backup_dir / f"{backup_name}_export.cypher"
        
        with open(cypher_file, 'w') as f:
            # Write header
            f.write(f"// Neo4j Database Backup\n")
            f.write(f"// Created: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"// Source: {self.neo4j_uri}\n\n")
            
            with self.driver.session() as session:
                # Export nodes
                f.write("// ========== NODES ==========\n\n")
                
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                
                for label in labels:
                    logger.info(f"Exporting {label} nodes...")
                    f.write(f"// {label} nodes\n")
                    
                    node_result = session.run(f"MATCH (n:{label}) RETURN n")
                    
                    for record in node_result:
                        node = record["n"]
                        props = dict(node)
                        
                        # Format properties for Cypher
                        props_str = ", ".join(
                            f"{k}: {self._format_value(v)}" 
                            for k, v in props.items()
                        )
                        
                        f.write(f"CREATE (:{label} {{{props_str}}});\n")
                    
                    f.write("\n")
                
                # Export relationships
                f.write("// ========== RELATIONSHIPS ==========\n\n")
                
                result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in result]
                
                for rel_type in rel_types:
                    logger.info(f"Exporting {rel_type} relationships...")
                    f.write(f"// {rel_type} relationships\n")
                    
                    rel_result = session.run(
                        f"""
                        MATCH (source)-[r:{rel_type}]->(target)
                        RETURN source.id as source_id, target.id as target_id,
                               properties(r) as props,
                               labels(source)[0] as source_label,
                               labels(target)[0] as target_label
                        """
                    )
                    
                    for record in rel_result:
                        source_id = record["source_id"]
                        target_id = record["target_id"]
                        props = dict(record["props"]) if record["props"] else {}
                        source_label = record["source_label"]
                        target_label = record["target_label"]
                        
                        # Format properties for Cypher
                        props_str = ""
                        if props:
                            props_str = " {" + ", ".join(
                                f"{k}: {self._format_value(v)}" 
                                for k, v in props.items()
                            ) + "}"
                        
                        f.write(
                            f"MATCH (source:{source_label} {{id: {self._format_value(source_id)}}}), "
                            f"(target:{target_label} {{id: {self._format_value(target_id)}}}) "
                            f"CREATE (source)-[:{rel_type}{props_str}]->(target);\n"
                        )
                    
                    f.write("\n")
        
        logger.info(f"Exported database to {cypher_file}")
        return cypher_file
    
    def _format_value(self, value: Any) -> str:
        """
        Format a Python value as a Cypher literal.
        
        Args:
            value: Python value to format
        
        Returns:
            Cypher literal string
        """
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape quotes and backslashes
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(value, list):
            items = ", ".join(self._format_value(item) for item in value)
            return f"[{items}]"
        elif isinstance(value, dict):
            items = ", ".join(
                f"{k}: {self._format_value(v)}" 
                for k, v in value.items()
            )
            return f"{{{items}}}"
        else:
            # Fallback: convert to string
            return f'"{str(value)}"'
    
    def create_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a complete backup of the Neo4j database.
        
        This method:
        1. Gathers database statistics
        2. Exports database to Cypher statements
        3. Creates a backup metadata file
        
        Requirements: 16.4
        
        Args:
            backup_name: Optional name for this backup. If not provided,
                        uses timestamp-based name.
        
        Returns:
            Dictionary with backup information
        """
        # Generate backup name if not provided
        if backup_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"neo4j_backup_{timestamp}"
        
        logger.info(f"Creating backup: {backup_name}")
        
        # Get database statistics
        stats = self.get_database_stats()
        
        # Export to Cypher
        cypher_file = self.export_to_cypher(backup_name)
        
        # Create backup metadata
        metadata = {
            "backup_name": backup_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "neo4j_uri": self.neo4j_uri,
            "statistics": stats,
            "cypher_file": str(cypher_file.name),
            "backup_method": "cypher_export",
        }
        
        metadata_file = self.backup_dir / f"{backup_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Backup metadata saved to {metadata_file}")
        
        # Create a "latest" symlink for easy access
        latest_link = self.backup_dir / "latest_backup.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        # Write latest backup info
        with open(latest_link, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Backup completed successfully: {backup_name}")
        logger.info(f"  Cypher file: {cypher_file}")
        logger.info(f"  Metadata file: {metadata_file}")
        logger.info(f"  Total nodes: {stats['total_nodes']}")
        logger.info(f"  Total relationships: {stats['total_relationships']}")
        
        return metadata
    
    def list_backups(self) -> list[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup metadata dictionaries
        """
        backups = []
        
        for metadata_file in self.backup_dir.glob("*_metadata.json"):
            if metadata_file.name == "latest_backup.json":
                continue
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                backups.append(metadata)
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return backups
    
    def verify_backup(self, backup_name: str) -> bool:
        """
        Verify that a backup is complete and valid.
        
        Args:
            backup_name: Name of the backup to verify
        
        Returns:
            True if backup is valid, False otherwise
        """
        logger.info(f"Verifying backup: {backup_name}")
        
        # Check metadata file exists
        metadata_file = self.backup_dir / f"{backup_name}_metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check Cypher file exists
        cypher_file = self.backup_dir / metadata["cypher_file"]
        if not cypher_file.exists():
            logger.error(f"Cypher file not found: {cypher_file}")
            return False
        
        # Check Cypher file is not empty
        if cypher_file.stat().st_size == 0:
            logger.error(f"Cypher file is empty: {cypher_file}")
            return False
        
        logger.info(f"✓ Backup verification passed: {backup_name}")
        return True


def main():
    """Main entry point for backup script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Backup Neo4j database before migration"
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="URI for Neo4j database to backup"
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Username for database"
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "your_password"),
        help="Password for database"
    )
    parser.add_argument(
        "--backup-dir",
        default="data/backups",
        help="Directory to store backups (default: data/backups)"
    )
    parser.add_argument(
        "--name",
        help="Optional name for this backup (default: timestamp-based)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available backups"
    )
    parser.add_argument(
        "--verify",
        help="Verify a specific backup by name"
    )
    
    args = parser.parse_args()
    
    # Create backup manager
    with Neo4jBackupManager(
        neo4j_uri=args.uri,
        neo4j_user=args.user,
        neo4j_password=args.password,
        backup_dir=args.backup_dir
    ) as manager:
        
        if args.list:
            # List backups
            backups = manager.list_backups()
            
            if not backups:
                logger.info("No backups found")
            else:
                logger.info(f"Found {len(backups)} backup(s):")
                for backup in backups:
                    logger.info(f"  - {backup['backup_name']} ({backup['timestamp']})")
                    logger.info(f"    Nodes: {backup['statistics']['total_nodes']}, "
                              f"Relationships: {backup['statistics']['total_relationships']}")
        
        elif args.verify:
            # Verify backup
            is_valid = manager.verify_backup(args.verify)
            sys.exit(0 if is_valid else 1)
        
        else:
            # Create backup
            metadata = manager.create_backup(backup_name=args.name)
            
            # Verify the backup
            is_valid = manager.verify_backup(metadata["backup_name"])
            
            if is_valid:
                logger.info("✓ Backup created and verified successfully")
                sys.exit(0)
            else:
                logger.error("✗ Backup verification failed")
                sys.exit(1)


if __name__ == "__main__":
    main()
