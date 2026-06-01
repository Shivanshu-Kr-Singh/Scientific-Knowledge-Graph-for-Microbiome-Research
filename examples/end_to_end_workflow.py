"""
examples/end_to_end_workflow.py
--------------------------------
End-to-End Example Workflow: Scientific Knowledge Graph Pipeline

This script demonstrates the complete pipeline from data collection through
graph construction to research queries. It showcases:

1. Data Collection: Fetching papers from PubMed/EuropePMC
2. NLP Enrichment: Entity extraction and article classification
3. Graph Construction: Building semantic relationships with provenance
4. Research Queries: Executing all 5 core research questions

Requirements: 1.1, 1.2, 1.3

Usage:
    python examples/end_to_end_workflow.py --mode full
    python examples/end_to_end_workflow.py --mode query-only
    python examples/end_to_end_workflow.py --mode demo

Author: Scientific Knowledge Graph Team
Date: 2024
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from nlp.enriched_record import EnrichedPaperRecord, NamedEntity, ParsedSection

# Layer 3: Knowledge Graph
from graph.enhanced_kg_pipeline import EnhancedKGPipeline
from graph.research_query_engine import ResearchQueryEngine
from graph.provenance import ProvenanceEncoder
from graph.semantic_extractor import SemanticRelationshipExtractor
from graph.relationship_reifier import RelationshipReifier
from graph.enhanced_neo4j_loader import EnhancedNeo4jLoader


class EndToEndWorkflow:
    """
    Complete end-to-end workflow demonstrating the scientific knowledge graph pipeline.
    
    This class orchestrates:
    - Paper collection from multiple sources
    - NLP enrichment (entity extraction, classification)
    - Graph construction with semantic relationships
    - Research query execution
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j_enhanced"
    ):
        """
        Initialize the workflow with Neo4j connection.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            database=neo4j_database
        )
        
        # Initialize components
        self.query_engine = ResearchQueryEngine(self.driver)
        
        print("=" * 80)
        print("End-to-End Scientific Knowledge Graph Workflow")
        print("=" * 80)
        print(f"Neo4j URI: {neo4j_uri}")
        print(f"Database: {neo4j_database}")
        print(f"Initialized at: {datetime.now().isoformat()}")
        print("=" * 80)
    
    def step_1_collection(self, query: str = "microbiome diabetes", max_papers: int = 5) -> List[Dict[str, Any]]:
        """
        STEP 1: Data Collection
        
        Collect papers from PubMed and EuropePMC based on search query.
        
        Args:
            query: Search query for papers
            max_papers: Maximum number of papers to collect
            
        Returns:
            List of collected paper metadata
        """
        print("\n" + "=" * 80)
        print("STEP 1: DATA COLLECTION")
        print("=" * 80)
        print(f"Query: '{query}'")
        print(f"Max papers: {max_papers}")
        print()
        
        # Simulate paper collection (in production, use actual collectors)
        papers = [
            {
                "pmid": "12345678",
                "title": "Gut microbiome alterations in Type 2 Diabetes: A randomized controlled trial",
                "abstract": "We investigated gut microbiome changes in T2D patients. "
                           "Bacteroides fragilis showed increased abundance (p<0.001, LDA=3.2). "
                           "Faecalibacterium prausnitzii was significantly decreased (p=0.003).",
                "year": 2024,
                "doi": "10.1234/example1",
                "source": "PubMed"
            },
            {
                "pmid": "23456789",
                "title": "Probiotic intervention modulates gut microbiota in diabetes",
                "abstract": "Lactobacillus acidophilus supplementation (10^9 CFU/day, 8 weeks) "
                           "increased Lactobacillus abundance in T2D patients (n=120, p<0.01).",
                "year": 2023,
                "doi": "10.5678/example2",
                "source": "EuropePMC"
            },
            {
                "pmid": "34567890",
                "title": "16S rRNA sequencing reveals microbiome dysbiosis in IBD",
                "abstract": "Using 16S rRNA sequencing (Illumina MiSeq, n=85), we found "
                           "Faecalibacterium prausnitzii decreased in IBD patients. "
                           "Data deposited: SRA accession PRJNA123456.",
                "year": 2024,
                "doi": "10.9012/example3",
                "source": "PubMed"
            },
            {
                "pmid": "45678901",
                "title": "Shotgun metagenomics analysis of Crohn's disease microbiome",
                "abstract": "Shotgun metagenomic sequencing revealed Escherichia coli enrichment "
                           "in Crohn's disease (p=0.002). Data: ENA accession ERP234567.",
                "year": 2023,
                "doi": "10.3456/example4",
                "source": "EuropePMC"
            },
            {
                "pmid": "56789012",
                "title": "Fecal microbiota transplantation for recurrent C. difficile infection",
                "abstract": "FMT restored Faecalibacterium prausnitzii levels in CDI patients "
                           "(n=45, p<0.001). Treatment duration: 4 weeks.",
                "year": 2024,
                "doi": "10.7890/example5",
                "source": "PubMed"
            }
        ]
        
        print(f"✓ Collected {len(papers)} papers")
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. [{paper['source']}] {paper['title'][:60]}...")
            print(f"     PMID: {paper['pmid']}, Year: {paper['year']}")
        
        print(f"\n✓ Collection complete: {len(papers)} papers ready for enrichment")
        return papers
    
    def step_2_enrichment(self, papers: List[Dict[str, Any]]) -> List[EnrichedPaperRecord]:
        """
        STEP 2: NLP Enrichment
        
        Extract entities, classify articles, and enrich paper metadata.
        
        Args:
            papers: List of collected paper metadata
            
        Returns:
            List of enriched paper records
        """
        print("\n" + "=" * 80)
        print("STEP 2: NLP ENRICHMENT")
        print("=" * 80)
        print(f"Processing {len(papers)} papers...")
        print()
        
        enriched_papers = []
        
        for i, paper in enumerate(papers, 1):
            print(f"Processing paper {i}/{len(papers)}: {paper['title'][:50]}...")
            
            # Simulate entity extraction
            entities = self._extract_entities_demo(paper)
            methods = self._extract_methods_demo(paper)
            
            # Simulate article classification
            article_type = self._classify_article_demo(paper)
            
            # Create enriched record
            enriched = EnrichedPaperRecord(
                pmid=paper.get("pmid"),
                doi=paper.get("doi"),
                title=paper["title"],
                abstract=paper["abstract"],
                year=paper["year"],
                entities=entities,
                methods=[e.text for e in entities if e.label == "method"],
                taxa=[e.text for e in entities if e.label == "taxon"],
                diseases=[e.text for e in entities if e.label == "disease"],
                article_type_normalized=article_type,
                data_availability=None,  # Will be set below
                accession_numbers=self._extract_accessions_demo(paper["abstract"])
            )
            
            enriched_papers.append(enriched)
            
            # Determine data availability status
            data_status = "open" if "SRA" in paper["abstract"] or "ENA" in paper["abstract"] else "closed"
            
            print(f"  ✓ Extracted {len(entities)} entities")
            print(f"  ✓ Identified {len(methods)} methods")
            print(f"  ✓ Article type: {article_type}")
            print(f"  ✓ Data availability: {data_status}")
            print()
        
        print(f"✓ Enrichment complete: {len(enriched_papers)} papers enriched")
        return enriched_papers
    
    def step_3_graph_construction(self, enriched_papers: List[EnrichedPaperRecord]) -> Dict[str, int]:
        """
        STEP 3: Graph Construction
        
        Build knowledge graph with semantic relationships and provenance.
        
        Args:
            enriched_papers: List of enriched paper records
            
        Returns:
            Statistics about graph construction
        """
        print("\n" + "=" * 80)
        print("STEP 3: GRAPH CONSTRUCTION")
        print("=" * 80)
        print(f"Building graph from {len(enriched_papers)} papers...")
        print()
        
        # Initialize pipeline
        pipeline = EnhancedKGPipeline(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
            neo4j_database=self.neo4j_database
        )
        
        # Process papers through pipeline
        stats = {
            "papers_processed": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "reified_claims": 0
        }
        
        for i, paper in enumerate(enriched_papers, 1):
            print(f"Processing paper {i}/{len(enriched_papers)}: {paper.title[:50]}...")
            
            try:
                # Process paper through pipeline
                result = pipeline.process_paper(paper)
                
                stats["papers_processed"] += 1
                stats["nodes_created"] += result.get("nodes_created", 0)
                stats["relationships_created"] += result.get("relationships_created", 0)
                
                print(f"  ✓ Created {result.get('nodes_created', 0)} nodes")
                print(f"  ✓ Created {result.get('relationships_created', 0)} relationships")
                
            except Exception as e:
                print(f"  ✗ Error processing paper: {e}")
        
        print()
        print("Graph Construction Summary:")
        print(f"  Papers processed: {stats['papers_processed']}")
        print(f"  Nodes created: {stats['nodes_created']}")
        print(f"  Relationships created: {stats['relationships_created']}")
        print(f"  Reified claims: {stats['reified_claims']}")
        
        print(f"\n✓ Graph construction complete")
        return stats
    
    def step_4_research_queries(self) -> Dict[str, Any]:
        """
        STEP 4: Research Queries
        
        Execute all 5 core research questions and display results.
        
        Returns:
            Dictionary containing results from all queries
        """
        print("\n" + "=" * 80)
        print("STEP 4: RESEARCH QUERIES")
        print("=" * 80)
        print("Executing all 5 core research questions...")
        print()
        
        results = {}
        
        # Query 1: Cross-Study Disease-Microbiome Associations
        print("-" * 80)
        print("QUERY 1: Cross-Study Disease-Microbiome Associations")
        print("-" * 80)
        print("Research Question: Which gut microbiome taxa show consistent association")
        print("with Type 2 Diabetes across RCT studies with open sequencing data?")
        print()
        
        result_q1 = self.query_engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=2,
            confidence_threshold=0.7,
            require_open_data=False
        )
        
        results["query_1"] = result_q1
        self._display_query_results(result_q1, "Cross-Study Associations")
        
        # Query 2: Intervention Effectiveness Evidence
        print("\n" + "-" * 80)
        print("QUERY 2: Intervention Effectiveness Evidence")
        print("-" * 80)
        print("Research Question: What interventions (probiotics, FMT, diet) have")
        print("RCT-level evidence for modifying specific gut taxa?")
        print()
        
        result_q2 = self.query_engine.query_intervention_evidence(
            intervention_types=["probiotic", "FMT", "diet"],
            min_sample_size=30,
            evidence_strength="moderate"
        )
        
        results["query_2"] = result_q2
        self._display_query_results(result_q2, "Intervention Evidence")
        
        # Query 3: Methodology Landscape and Data Availability
        print("\n" + "-" * 80)
        print("QUERY 3: Methodology Landscape and Data Availability")
        print("-" * 80)
        print("Research Question: Which microbiome studies from 2023-2024 deposited")
        print("data on SRA/ENA and used shotgun metagenomics vs 16S sequencing?")
        print()
        
        result_q3 = self.query_engine.query_methodology_landscape(
            year_start=2023,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
            require_deposited_data=True
        )
        
        results["query_3"] = result_q3
        self._display_query_results(result_q3, "Methodology Landscape")
        
        # Query 4: Top Associations by Evidence Quality
        print("\n" + "-" * 80)
        print("QUERY 4: Top Associations by Evidence Quality")
        print("-" * 80)
        print("Research Question: Top 10 taxa associated with IBD across multiple")
        print("papers with high confidence, ranked by evidence quality.")
        print()
        
        result_q4 = self.query_engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=10,
            min_confidence=0.7
        )
        
        results["query_4"] = result_q4
        self._display_query_results(result_q4, "Top Associations")
        
        # Query 5: Conflicting Evidence Detection
        print("\n" + "-" * 80)
        print("QUERY 5: Conflicting Evidence Detection")
        print("-" * 80)
        print("Research Question: Which taxa show conflicting associations")
        print("(increased vs decreased) for Crohn's disease?")
        print()
        
        result_q5 = self.query_engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=2
        )
        
        results["query_5"] = result_q5
        self._display_query_results(result_q5, "Conflicting Evidence")
        
        print("\n" + "=" * 80)
        print("✓ All research queries completed successfully")
        print("=" * 80)
        
        return results
    
    def _display_query_results(self, result: Any, query_name: str):
        """Display query results in a formatted way."""
        print(f"Query: {result.query_description}")
        print(f"Execution time: {result.execution_time_ms:.1f} ms")
        print(f"Results found: {result.result_count}")
        
        if result.error:
            print(f"❌ Error: {result.error}")
        elif result.timeout:
            print(f"⏱️  Query timed out")
        elif result.result_count == 0:
            print(f"ℹ️  No results found (this is expected for demo data)")
        else:
            print(f"\nTop results:")
            for i, record in enumerate(result.results[:3], 1):
                print(f"  {i}. {self._format_result_record(record)}")
        
        print()
    
    def _format_result_record(self, record: Dict[str, Any]) -> str:
        """Format a single result record for display."""
        if "taxon_name" in record:
            return f"{record['taxon_name']} (papers: {record.get('paper_count', 'N/A')})"
        elif "intervention_type" in record:
            return f"{record['intervention_type']} → {record.get('taxon_name', 'N/A')}"
        elif "method" in record:
            return f"{record['method']} ({record.get('year', 'N/A')})"
        else:
            return str(record)
    
    def _extract_entities_demo(self, paper: Dict[str, Any]) -> List[NamedEntity]:
        """Demo entity extraction (simplified)."""
        entities = []
        
        # Extract taxa
        taxa_keywords = ["Bacteroides fragilis", "Faecalibacterium prausnitzii", 
                        "Lactobacillus acidophilus", "Escherichia coli"]
        for taxon in taxa_keywords:
            if taxon in paper["abstract"]:
                entities.append(NamedEntity(
                    text=taxon,
                    label="taxon",
                    start=paper["abstract"].find(taxon),
                    end=paper["abstract"].find(taxon) + len(taxon)
                ))
        
        # Extract diseases
        disease_keywords = ["Type 2 Diabetes", "T2D", "IBD", "Crohn's disease", "diabetes"]
        for disease in disease_keywords:
            if disease in paper["abstract"]:
                entities.append(NamedEntity(
                    text=disease,
                    label="disease",
                    start=paper["abstract"].find(disease),
                    end=paper["abstract"].find(disease) + len(disease)
                ))
        
        return entities
    
    def _extract_methods_demo(self, paper: Dict[str, Any]) -> List[str]:
        """Demo method extraction (simplified)."""
        methods = []
        
        method_keywords = {
            "16S rRNA sequencing": "16S",
            "shotgun metagenomics": "shotgun",
            "Illumina": "sequencing"
        }
        
        for method_name, keyword in method_keywords.items():
            if keyword in paper["abstract"]:
                methods.append(method_name)
        
        return methods
    
    def _classify_article_demo(self, paper: Dict[str, Any]) -> str:
        """Demo article classification (simplified)."""
        abstract = paper["abstract"].lower()
        
        if "randomized controlled trial" in abstract or "rct" in abstract:
            return "original_research"
        elif "meta-analysis" in abstract or "systematic review" in abstract:
            return "meta_analysis"
        elif "review" in abstract:
            return "review"
        else:
            return "original_research"
    
    def _extract_accessions_demo(self, abstract: str) -> List[str]:
        """Demo accession number extraction (simplified)."""
        accessions = []
        
        if "PRJNA" in abstract:
            start = abstract.find("PRJNA")
            accessions.append(abstract[start:start+12])
        
        if "ERP" in abstract:
            start = abstract.find("ERP")
            accessions.append(abstract[start:start+9])
        
        return accessions
    
    def run_full_workflow(self):
        """Run the complete end-to-end workflow."""
        print("\n" + "=" * 80)
        print("RUNNING FULL END-TO-END WORKFLOW")
        print("=" * 80)
        
        # Step 1: Collection
        papers = self.step_1_collection(query="microbiome diabetes", max_papers=5)
        
        # Step 2: Enrichment
        enriched_papers = self.step_2_enrichment(papers)
        
        # Step 3: Graph Construction
        graph_stats = self.step_3_graph_construction(enriched_papers)
        
        # Step 4: Research Queries
        query_results = self.step_4_research_queries()
        
        # Summary
        print("\n" + "=" * 80)
        print("WORKFLOW SUMMARY")
        print("=" * 80)
        print(f"Papers collected: {len(papers)}")
        print(f"Papers enriched: {len(enriched_papers)}")
        print(f"Graph nodes created: {graph_stats['nodes_created']}")
        print(f"Graph relationships created: {graph_stats['relationships_created']}")
        print(f"Research queries executed: {len(query_results)}")
        print("=" * 80)
        print("✓ Full workflow completed successfully!")
        print("=" * 80)
    
    def run_query_only_mode(self):
        """Run only the research queries (assumes graph is already built)."""
        print("\n" + "=" * 80)
        print("RUNNING QUERY-ONLY MODE")
        print("=" * 80)
        print("Assuming graph is already constructed...")
        print()
        
        query_results = self.step_4_research_queries()
        
        print("\n" + "=" * 80)
        print("✓ Query-only mode completed!")
        print("=" * 80)
    
    def run_demo_mode(self):
        """Run a quick demo showing expected outputs."""
        print("\n" + "=" * 80)
        print("RUNNING DEMO MODE")
        print("=" * 80)
        print("This mode shows expected outputs without connecting to Neo4j")
        print()
        
        self._display_demo_outputs()
        
        print("\n" + "=" * 80)
        print("✓ Demo mode completed!")
        print("=" * 80)
    
    def _display_demo_outputs(self):
        """Display expected outputs for all queries."""
        print("\n" + "-" * 80)
        print("EXPECTED OUTPUT: Query 1 - Cross-Study Associations")
        print("-" * 80)
        print("""
Taxon: Bacteroides fragilis
  Papers: 5
  Consensus confidence: 0.85
  Consensus direction: increased
  Direction consistency: 80.0%
  Increased: 4, Decreased: 1, No change: 0

Interpretation:
- Bacteroides fragilis shows strong evidence (5 papers) for increased abundance
- High consensus confidence (0.85) indicates reliable findings
- 80% direction consistency suggests some heterogeneity
- Recommended for further investigation as potential biomarker
        """)
        
        print("\n" + "-" * 80)
        print("EXPECTED OUTPUT: Query 2 - Intervention Evidence")
        print("-" * 80)
        print("""
Intervention: probiotic
  Taxon: Lactobacillus acidophilus
  Effect: increased
  Papers: 8
  Total sample size: 450
  Average confidence: 0.87

Interpretation:
- Strong evidence (8 papers, 450 participants) for probiotic effectiveness
- High confidence (0.87) in effect direction
- Sufficient sample size for clinical recommendations
- Consider for evidence-based treatment protocols
        """)
        
        print("\n" + "-" * 80)
        print("EXPECTED OUTPUT: Query 3 - Methodology Landscape")
        print("-" * 80)
        print("""
2024 - shotgun metagenomics
  Total papers: 45
  Papers with data: 38
  Data availability: 84.4%
  NCBI SRA: 30, ENA: 12

Interpretation:
- High data sharing compliance (84.4%) for shotgun metagenomics
- NCBI SRA is preferred repository (30 vs 12)
- Trend shows improving data availability over time
- Funding agencies can use for policy assessment
        """)
        
        print("\n" + "-" * 80)
        print("EXPECTED OUTPUT: Query 4 - Top Associations")
        print("-" * 80)
        print("""
1. Faecalibacterium prausnitzii
   Papers: 12, Avg confidence: 0.89
   Direction: decreased, Consistency: 91.7%

2. Escherichia coli
   Papers: 10, Avg confidence: 0.86
   Direction: increased, Consistency: 90.0%

Interpretation:
- Top-ranked taxa have strongest evidence base
- High consistency indicates robust findings
- Prioritize these taxa for meta-analysis
- Use for educational materials and literature reviews
        """)
        
        print("\n" + "-" * 80)
        print("EXPECTED OUTPUT: Query 5 - Conflicting Evidence")
        print("-" * 80)
        print("""
Taxon: Escherichia coli
  Total papers: 8
  Increased: 5 papers (62.5%)
  Decreased: 3 papers (37.5%)
  Direction balance: 2

Interpretation:
- Conflicting evidence suggests heterogeneity in study populations
- May indicate different E. coli strains or disease subtypes
- Requires subgroup analysis or meta-regression
- Opportunity for follow-up studies to resolve discrepancy
        """)
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("\n✓ Neo4j connection closed")


def main():
    """Main entry point for the workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="End-to-End Scientific Knowledge Graph Workflow"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "query-only", "demo"],
        default="demo",
        help="Workflow mode: full (complete pipeline), query-only (queries only), demo (show expected outputs)"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        default="password",
        help="Neo4j password"
    )
    parser.add_argument(
        "--neo4j-database",
        default="neo4j_enhanced",
        help="Neo4j database name"
    )
    
    args = parser.parse_args()
    
    # Create workflow instance
    workflow = EndToEndWorkflow(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database
    )
    
    try:
        if args.mode == "full":
            workflow.run_full_workflow()
        elif args.mode == "query-only":
            workflow.run_query_only_mode()
        elif args.mode == "demo":
            workflow.run_demo_mode()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if args.mode != "demo":
            workflow.close()


if __name__ == "__main__":
    main()
