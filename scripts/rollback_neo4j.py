#!/usr/bin/env python3
"""
scripts/rollback_neo4j.py
--------------------------
Rollback script to restore old Neo4j database from backup.

This script restores the old Neo4j database instance from a backup
if migration issues are encountered.

Requirements: 16.4
"""

import os
import sys
import logging
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


class Neo4jRollbackManager:
    """
    Manages Neo4j database rollback from backups.
    
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
        Initialize the rollback manager.
        
        Args:
            neo4j_uri: URI for Neo4j database to restore to
            neo4j_user: Username for database
            neo4j_password: Password for database
            backup_dir: Directory containing backups (default: data/backups)
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.backup_dir = Path(backup_dir)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        if not self.backup_dir.exists():
            raise FileNotFoundError(f"Backup directory not found: {self.backup_dir}")
        
        logger.info(f"Initialized rollback manager for {neo4j_uri}")
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
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                backups.append(metadata)
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return backups
    
    def get_latest_backup(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest backup metadata.
        
        Returns:
            Latest backup metadata, or None if no backups exist
        """
        latest_file = self.backup_dir / "latest_backup.json"
        
        if not latest_file.exists():
            logger.warning("No latest backup found")
            return None
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
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
        with open(metadata_file, 'r', encoding='utf-8') as f:
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
    
    def clear_database(self) -> None:
        """
        Clear all nodes and relationships from the database.
        
        WARNING: This is a destructive operation!
        """
        logger.warning("Clearing database (this may take a while)...")
        
        with self.driver.session() as session:
            # Delete all relationships first
            logger.info("Deleting all relationships...")
            session.run("MATCH ()-[r]->() DELETE r")
            
            # Delete all nodes
            logger.info("Deleting all nodes...")
            session.run("MATCH (n) DELETE n")
            
            # Drop all indexes
            logger.info("Dropping indexes...")
            result = session.run("SHOW INDEXES")
            for record in result:
                index_name = record.get("name")
                if index_name:
                    try:
                        session.run(f"DROP INDEX {index_name} IF EXISTS")
                    except Exception as e:
                        logger.warning(f"Failed to drop index {index_name}: {e}")
            
            # Drop all constraints
            logger.info("Dropping constraints...")
            result = session.run("SHOW CONSTRAINTS")
            for record in result:
                constraint_name = record.get("name")
                if constraint_name:
                    try:
                        session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                    except Exception as e:
                        logger.warning(f"Failed to drop constraint {constraint_name}: {e}")
        
        logger.info("✓ Database cleared")
    
    def restore_from_cypher(self, cypher_file: Path) -> None:
        """
        Restore database from Cypher export file.
        
        Args:
            cypher_file: Path to Cypher export file
        """
        logger.info(f"Restoring database from {cypher_file}...")
        
        # Read Cypher file
        with open(cypher_file, 'r', encoding='utf-8') as f:
            cypher_content = f.read()
        
        # Split into individual statements
        statements = [
            stmt.strip() 
            for stmt in cypher_content.split(';') 
            if stmt.strip() and not stmt.strip().startswith('//')
        ]
        
        logger.info(f"Executing {len(statements)} Cypher statements...")
        
        # Execute statements in batches
        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(statements), batch_size):
                batch = statements[i:i + batch_size]
                
                logger.info(f"Executing batch {i // batch_size + 1}/{(len(statements) + batch_size - 1) // batch_size}...")
                
                for stmt in batch:
                    try:
                        session.run(stmt)
                    except Exception as e:
                        logger.error(f"Failed to execute statement: {stmt[:100]}...")
                        logger.error(f"Error: {e}")
                        raise
        
        logger.info("✓ Database restored from Cypher file")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current database.
        
        Returns:
            Dictionary with node counts and relationship counts
        """
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
        
        return stats
    
    def rollback(
        self,
        backup_name: Optional[str] = None,
        skip_verification: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Rollback database to a previous backup.
        
        This method:
        1. Verifies the backup is valid
        2. Clears the current database
        3. Restores from the backup
        4. Verifies the restoration
        
        Requirements: 16.4
        
        Args:
            backup_name: Name of backup to restore. If None, uses latest backup.
            skip_verification: Skip backup verification (not recommended)
            dry_run: If True, only verify backup without restoring
        
        Returns:
            Dictionary with rollback information
        """
        # Get backup metadata
        if backup_name is None:
            logger.info("No backup name specified, using latest backup...")
            metadata = self.get_latest_backup()
            if metadata is None:
                raise ValueError("No backups available")
            backup_name = metadata["backup_name"]
        else:
            metadata_file = self.backup_dir / f"{backup_name}_metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        logger.info(f"Rolling back to backup: {backup_name}")
        logger.info(f"Backup timestamp: {metadata['timestamp']}")
        
        # Verify backup
        if not skip_verification:
            if not self.verify_backup(backup_name):
                raise ValueError(f"Backup verification failed: {backup_name}")
        
        if dry_run:
            logger.info("Dry run mode: backup verification passed, skipping restoration")
            return {
                "backup_name": backup_name,
                "dry_run": True,
                "verification_passed": True,
            }
        
        # Get current database stats (for comparison)
        logger.info("Getting current database statistics...")
        current_stats = self.get_database_stats()
        logger.info(f"Current database: {current_stats['total_nodes']} nodes, "
                   f"{current_stats['total_relationships']} relationships")
        
        # Clear database
        self.clear_database()
        
        # Restore from backup
        cypher_file = self.backup_dir / metadata["cypher_file"]
        self.restore_from_cypher(cypher_file)
        
        # Verify restoration
        logger.info("Verifying restoration...")
        restored_stats = self.get_database_stats()
        
        expected_stats = metadata["statistics"]
        
        # Compare statistics
        nodes_match = restored_stats["total_nodes"] == expected_stats["total_nodes"]
        rels_match = restored_stats["total_relationships"] == expected_stats["total_relationships"]
        
        if nodes_match and rels_match:
            logger.info("✓ Restoration verified successfully")
            logger.info(f"  Restored {restored_stats['total_nodes']} nodes")
            logger.info(f"  Restored {restored_stats['total_relationships']} relationships")
        else:
            logger.warning("⚠ Restoration statistics do not match backup")
            logger.warning(f"  Expected nodes: {expected_stats['total_nodes']}, "
                         f"Got: {restored_stats['total_nodes']}")
            logger.warning(f"  Expected relationships: {expected_stats['total_relationships']}, "
                         f"Got: {restored_stats['total_relationships']}")
        
        # Create rollback report
        rollback_info = {
            "backup_name": backup_name,
            "backup_timestamp": metadata["timestamp"],
            "rollback_timestamp": datetime.now(timezone.utc).isoformat(),
            "neo4j_uri": self.neo4j_uri,
            "previous_state": current_stats,
            "restored_state": restored_stats,
            "expected_state": expected_stats,
            "verification_passed": nodes_match and rels_match,
        }
        
        # Save rollback report
        report_file = self.backup_dir / f"rollback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(rollback_info, f, indent=2)
        
        logger.info(f"Rollback report saved to {report_file}")
        
        return rollback_info


def main():
    """Main entry point for rollback script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rollback Neo4j database from backup"
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="URI for Neo4j database to restore to"
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
        help="Directory containing backups (default: data/backups)"
    )
    parser.add_argument(
        "--backup",
        help="Name of backup to restore (default: latest)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available backups"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify backup without restoring"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip backup verification (not recommended)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rollback without confirmation"
    )
    
    args = parser.parse_args()
    
    # Create rollback manager
    with Neo4jRollbackManager(
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
        
        else:
            # Perform rollback
            if not args.force and not args.dry_run:
                # Confirm with user
                backup_name = args.backup or "latest"
                response = input(
                    f"\n⚠️  WARNING: This will DELETE all data in the database and restore from backup '{backup_name}'.\n"
                    f"Are you sure you want to continue? (yes/no): "
                )
                
                if response.lower() != "yes":
                    logger.info("Rollback cancelled by user")
                    sys.exit(0)
            
            # Execute rollback
            try:
                rollback_info = manager.rollback(
                    backup_name=args.backup,
                    skip_verification=args.skip_verification,
                    dry_run=args.dry_run
                )
                
                if rollback_info.get("verification_passed", False):
                    logger.info("✓ Rollback completed successfully")
                    sys.exit(0)
                else:
                    logger.warning("⚠ Rollback completed with warnings")
                    sys.exit(1)
            
            except Exception as e:
                logger.error(f"✗ Rollback failed: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
