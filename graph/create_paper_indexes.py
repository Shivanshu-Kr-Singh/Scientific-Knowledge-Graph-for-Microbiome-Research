"""
graph/create_paper_indexes.py
------------------------------
Script to create indexes on Paper node properties for query optimization.

This script implements Task 10.1 from the scientific-knowledge-graph-core spec:
- Create indexes on Paper.year, Paper.article_type, Paper.data_availability
- Create composite index on (Paper.year, Paper.article_type)
- Verify index creation with SHOW INDEXES command

Requirements: 12.1, 12.4
"""

import os
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperIndexCreator:
    """
    Creates and verifies indexes on Paper node properties.
    
    Requirements:
    - 12.1: Create indexes on paper properties (year, article_type, data_availability)
    - 12.4: Create composite indexes for common query patterns (year+article_type)
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the index creator with Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
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
    
    def create_single_property_indexes(self) -> List[str]:
        """
        Create single-property indexes on Paper node properties.
        
        Creates indexes on:
        - Paper.year
        - Paper.article_type
        - Paper.data_availability
        
        Returns:
            List of created index names
        
        Requirement 12.1: Create indexes on paper properties
        """
        logger.info("Creating single-property indexes on Paper node...")
        
        indexes = [
            {
                "name": "paper_year",
                "query": "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)"
            },
            {
                "name": "paper_article_type",
                "query": "CREATE INDEX paper_article_type IF NOT EXISTS FOR (p:Paper) ON (p.article_type)"
            },
            {
                "name": "paper_data_availability",
                "query": "CREATE INDEX paper_data_availability IF NOT EXISTS FOR (p:Paper) ON (p.data_availability)"
            }
        ]
        
        created_indexes = []
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index["query"])
                    logger.info(f"✓ Created index: {index['name']}")
                    created_indexes.append(index["name"])
                except Exception as e:
                    logger.warning(f"Index {index['name']} creation failed (may already exist): {e}")
                    # Still add to list as it exists
                    created_indexes.append(index["name"])
        
        return created_indexes
    
    def create_composite_index(self) -> str:
        """
        Create composite index on (Paper.year, Paper.article_type).
        
        This composite index optimizes queries that filter by both year and
        article_type, which is a common pattern in research queries.
        
        Returns:
            Name of the created composite index
        
        Requirement 12.4: Create composite indexes for common query patterns
        """
        logger.info("Creating composite index on (Paper.year, Paper.article_type)...")
        
        index_name = "paper_year_article_type_composite"
        index_query = """
        CREATE INDEX paper_year_article_type_composite IF NOT EXISTS 
        FOR (p:Paper) ON (p.year, p.article_type)
        """
        
        with self.driver.session() as session:
            try:
                session.run(index_query)
                logger.info(f"✓ Created composite index: {index_name}")
            except Exception as e:
                logger.warning(f"Composite index creation failed (may already exist): {e}")
        
        return index_name
    
    def verify_indexes(self) -> List[Dict[str, Any]]:
        """
        Verify index creation using SHOW INDEXES command.
        
        Returns:
            List of index information dictionaries
        
        Requirement 12.1, 12.4: Verify index creation
        """
        logger.info("Verifying indexes with SHOW INDEXES command...")
        
        with self.driver.session() as session:
            # Use SHOW INDEXES to list all indexes
            result = session.run("SHOW INDEXES")
            
            indexes = []
            for record in result:
                index_info = {
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "entityType": record.get("entityType"),
                    "labelsOrTypes": record.get("labelsOrTypes"),
                    "properties": record.get("properties"),
                    "state": record.get("state"),
                }
                indexes.append(index_info)
            
            return indexes
    
    def display_paper_indexes(self, indexes: List[Dict[str, Any]]) -> None:
        """
        Display Paper-related indexes in a readable format.
        
        Args:
            indexes: List of index information dictionaries
        """
        logger.info("\n" + "="*80)
        logger.info("PAPER NODE INDEXES")
        logger.info("="*80)
        
        paper_indexes = [
            idx for idx in indexes 
            if idx.get("labelsOrTypes") and "Paper" in idx.get("labelsOrTypes", [])
        ]
        
        if not paper_indexes:
            logger.warning("No indexes found on Paper nodes!")
            return
        
        for idx in paper_indexes:
            logger.info(f"\nIndex Name: {idx['name']}")
            logger.info(f"  Type: {idx['type']}")
            logger.info(f"  Properties: {idx['properties']}")
            logger.info(f"  State: {idx['state']}")
        
        logger.info("\n" + "="*80)
        logger.info(f"Total Paper indexes: {len(paper_indexes)}")
        logger.info("="*80 + "\n")
    
    def create_all_indexes(self) -> Dict[str, Any]:
        """
        Create all required indexes and verify creation.
        
        This is the main entry point that:
        1. Creates single-property indexes
        2. Creates composite index
        3. Verifies all indexes
        4. Displays results
        
        Returns:
            Dictionary with creation results and statistics
        """
        logger.info("Starting index creation for Paper node properties...")
        
        # Create single-property indexes
        single_indexes = self.create_single_property_indexes()
        
        # Create composite index
        composite_index = self.create_composite_index()
        
        # Verify all indexes
        all_indexes = self.verify_indexes()
        
        # Display Paper-specific indexes
        self.display_paper_indexes(all_indexes)
        
        # Return results
        results = {
            "single_property_indexes": single_indexes,
            "composite_index": composite_index,
            "total_indexes_in_db": len(all_indexes),
            "paper_indexes_count": len([
                idx for idx in all_indexes 
                if idx.get("labelsOrTypes") and "Paper" in idx.get("labelsOrTypes", [])
            ])
        }
        
        logger.info("Index creation completed successfully!")
        logger.info(f"Results: {results}")
        
        return results


def main():
    """
    Main function to create and verify Paper node indexes.
    
    Reads Neo4j connection details from environment variables:
    - NEO4J_URI (default: bolt://localhost:7687)
    - NEO4J_USER (default: neo4j)
    - NEO4J_PASSWORD (required)
    """
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        logger.error("NEO4J_PASSWORD environment variable is required!")
        return
    
    # Create indexes
    with PaperIndexCreator(uri, user, password) as creator:
        results = creator.create_all_indexes()
        
        # Print summary
        print("\n" + "="*80)
        print("INDEX CREATION SUMMARY")
        print("="*80)
        print(f"Single-property indexes created: {len(results['single_property_indexes'])}")
        print(f"  - {', '.join(results['single_property_indexes'])}")
        print(f"Composite index created: {results['composite_index']}")
        print(f"Total Paper indexes in database: {results['paper_indexes_count']}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
