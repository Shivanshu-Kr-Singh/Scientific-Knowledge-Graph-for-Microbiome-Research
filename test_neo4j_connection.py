#!/usr/bin/env python3
"""Test Neo4j connection for both old and enhanced databases"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

def test_connection(uri, user, password, db_name):
    """Test connection to a Neo4j instance"""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' AS message")
            message = result.single()["message"]
            print(f"✓ {db_name}: {message}")
            print(f"  URI: {uri}")
            
            # Get Neo4j version
            version_result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version")
            for record in version_result:
                print(f"  Version: {record['name']} {record['version']}")
        
        driver.close()
        return True
    except Exception as e:
        print(f"✗ {db_name}: Connection failed")
        print(f"  URI: {uri}")
        print(f"  Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Neo4j Connections\n" + "="*50)
    
    # Test old database
    old_uri = os.getenv("NEO4J_URI")
    old_user = os.getenv("NEO4J_USER")
    old_password = os.getenv("NEO4J_PASSWORD")
    
    print("\n1. Old Database (Rollback)")
    old_success = test_connection(old_uri, old_user, old_password, "Old Database")
    
    # Test enhanced database
    new_uri = os.getenv("NEO4J_NEW_URI")
    new_user = os.getenv("NEO4J_NEW_USER")
    new_password = os.getenv("NEO4J_NEW_PASSWORD")
    
    print("\n2. Enhanced Database (New Schema)")
    new_success = test_connection(new_uri, new_user, new_password, "Enhanced Database")
    
    print("\n" + "="*50)
    if old_success and new_success:
        print("✓ All Neo4j instances are running and accessible!")
        print("\nYou can now:")
        print("  - Access old database at: http://localhost:7474")
        print("  - Access enhanced database at: http://localhost:7475")
        print("  - Run task 10.1: python graph/create_paper_indexes.py")
    else:
        print("✗ Some connections failed. Please check the errors above.")
