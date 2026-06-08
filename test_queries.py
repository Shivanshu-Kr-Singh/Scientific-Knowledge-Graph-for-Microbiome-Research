#!/usr/bin/env python3
"""
Test Queries Script
Runs all 5 research queries to verify the knowledge graph is working.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from graph.research_query_engine import ResearchQueryEngine
    from neo4j import GraphDatabase
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def test_connection():
    """Test Neo4j connection"""
    print_header("Testing Neo4j Connection")
    
    try:
        uri = os.getenv("NEO4J_ENHANCED_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_ENHANCED_USER", "neo4j")
        password = os.getenv("NEO4J_ENHANCED_PASSWORD", "password")
        database = os.getenv("NEO4J_ENHANCED_DATABASE", "neo4j_enhanced")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session(database=database) as session:
            # Count nodes
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            counts = {record["type"]: record["count"] for record in result}
        
        print(f"✅ Connected to Neo4j: {uri}")
        print(f"✅ Database: {database}\n")
        print("Node counts:")
        for node_type, count in counts.items():
            print(f"  {node_type}: {count}")
        
        if not counts:
            print("\n⚠️  Database is empty! Run Layer 3 to populate it:")
            print("   RUN_LAYER=3 python main.py")
            return None
        
        return driver
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker-compose -f docker-compose.neo4j-dual.yml up -d")
        return None

def test_query_1(engine):
    """Test Query 1: Cross-Study Associations"""
    print_header("QUERY 1: Cross-Study Disease-Microbiome Associations")
    
    try:
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="any",
            min_papers=1,
            confidence_threshold=0.5,
            require_open_data=False
        )
        
        print(f"Query: {result.query_description}")
        print(f"Execution time: {result.execution_time_ms:.1f}ms")
        print(f"Results: {result.result_count} taxa\n")
        
        if result.result_count == 0:
            print("⚠️  No results found. Try:")
            print("   - Lower confidence_threshold (e.g., 0.3)")
            print("   - Different disease name")
            print("   - Check if data was loaded into Neo4j")
        else:
            print("Top 5 taxa:")
            for i, taxon in enumerate(result.results[:5], 1):
                print(f"{i}. {taxon['taxon_name']}")
                print(f"   Papers: {taxon['paper_count']}, "
                      f"Confidence: {taxon['consensus_confidence']:.2f}, "
                      f"Direction: {taxon['consensus_direction']}")
        
        return result.result_count > 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

def test_query_2(engine):
    """Test Query 2: Intervention Evidence"""
    print_header("QUERY 2: Intervention Effectiveness Evidence")
    
    try:
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic", "FMT", "diet"],
            min_sample_size=10,
            evidence_strength="any"
        )
        
        print(f"Query: {result.query_description}")
        print(f"Execution time: {result.execution_time_ms:.1f}ms")
        print(f"Results: {result.result_count} interventions\n")
        
        if result.result_count == 0:
            print("⚠️  No results found. Try:")
            print("   - Lower min_sample_size (e.g., 5)")
            print("   - Add more intervention types")
        else:
            print("Top 5 interventions:")
            for i, intervention in enumerate(result.results[:5], 1):
                print(f"{i}. {intervention['intervention_type']} → {intervention['taxon_name']}")
                print(f"   Effect: {intervention['effect_direction']}, "
                      f"Papers: {intervention['paper_count']}, "
                      f"Sample size: {intervention['total_sample_size']}")
        
        return result.result_count > 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

def test_query_3(engine):
    """Test Query 3: Methodology Landscape"""
    print_header("QUERY 3: Methodology Landscape and Data Availability")
    
    try:
        result = engine.query_methodology_landscape(
            year_start=2024,
            year_end=2026,
            sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
            require_deposited_data=False
        )
        
        print(f"Query: {result.query_description}")
        print(f"Execution time: {result.execution_time_ms:.1f}ms")
        print(f"Results: {result.result_count} method-year combinations\n")
        
        if result.result_count == 0:
            print("⚠️  No results found. Try:")
            print("   - Wider year range (e.g., 2020-2026)")
            print("   - Different sequencing methods")
        else:
            print("Top 5 method-year combinations:")
            for i, row in enumerate(result.results[:5], 1):
                print(f"{i}. {row['year']} - {row['method']}")
                print(f"   Papers: {row['total_papers']}, "
                      f"With data: {row['papers_with_data']} "
                      f"({row['data_availability_pct']:.1f}%)")
        
        return result.result_count > 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

def test_query_4(engine):
    """Test Query 4: Top Associations"""
    print_header("QUERY 4: Top Associations by Evidence Quality")
    
    try:
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=10,
            min_confidence=0.5
        )
        
        print(f"Query: {result.query_description}")
        print(f"Execution time: {result.execution_time_ms:.1f}ms")
        print(f"Results: {result.result_count} taxa\n")
        
        if result.result_count == 0:
            print("⚠️  No results found. Try:")
            print("   - Lower min_confidence (e.g., 0.3)")
            print("   - Different disease name")
        else:
            print("Top 10 taxa:")
            for i, taxon in enumerate(result.results, 1):
                print(f"{i}. {taxon['taxon_name']}")
                print(f"   Papers: {taxon['paper_count']}, "
                      f"Confidence: {taxon['avg_confidence']:.2f}, "
                      f"Direction: {taxon['consensus_direction']}")
        
        return result.result_count > 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

def test_query_5(engine):
    """Test Query 5: Conflicting Evidence"""
    print_header("QUERY 5: Conflicting Evidence Detection")
    
    try:
        result = engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=1
        )
        
        print(f"Query: {result.query_description}")
        print(f"Execution time: {result.execution_time_ms:.1f}ms")
        print(f"Results: {result.result_count} taxa with conflicts\n")
        
        if result.result_count == 0:
            print("ℹ️  No conflicting evidence found (this is actually good!)")
            print("   It means all taxa have consistent direction across papers.")
        else:
            print("Taxa with conflicting evidence:")
            for i, taxon in enumerate(result.results[:5], 1):
                print(f"{i}. ⚠️  {taxon['taxon_name']}")
                print(f"   Total: {taxon['total_paper_count']} papers")
                print(f"   Increased: {taxon['increased_count']} "
                      f"({taxon['increased_percentage']:.1f}%), "
                      f"Decreased: {taxon['decreased_count']} "
                      f"({taxon['decreased_percentage']:.1f}%)")
        
        return True  # No conflicts is also a valid result
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

def main():
    """Main test function"""
    print_header("SCIENTIFIC KNOWLEDGE GRAPH - QUERY TESTS")
    
    # Test connection
    driver = test_connection()
    if not driver:
        print("\n❌ Cannot proceed without database connection")
        return 1
    
    # Create query engine
    try:
        engine = ResearchQueryEngine(driver)
    except Exception as e:
        print(f"❌ Failed to create query engine: {e}")
        driver.close()
        return 1
    
    # Run all queries
    tests = [
        ("Query 1: Cross-Study Associations", test_query_1),
        ("Query 2: Intervention Evidence", test_query_2),
        ("Query 3: Methodology Landscape", test_query_3),
        ("Query 4: Top Associations", test_query_4),
        ("Query 5: Conflicting Evidence", test_query_5)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func(engine)
            results.append((name, result))
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Close connection
    driver.close()
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nOverall: {passed}/{total} queries passed")
    
    if passed == total:
        print("\n🎉 All queries working successfully!")
        print("\nYour Scientific Knowledge Graph is fully operational!")
        print("\nNext steps:")
        print("1. Explore queries in Neo4j Browser: http://localhost:7474")
        print("2. Start REST API: python -m api.query_api")
        print("3. Run full test suite: pytest")
        print("4. See QUERY_EXAMPLES.md for more query examples")
    else:
        print("\n⚠️  Some queries failed or returned no results.")
        print("\nPossible causes:")
        print("- Database is empty (run Layer 3 to populate)")
        print("- Query parameters too strict (lower thresholds)")
        print("- Data doesn't match query criteria (try different diseases)")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
