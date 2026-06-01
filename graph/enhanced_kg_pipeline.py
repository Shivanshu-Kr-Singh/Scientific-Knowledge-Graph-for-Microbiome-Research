"""
graph/enhanced_kg_pipeline.py
------------------------------
Enhanced knowledge graph pipeline that runs in parallel with the existing system.

This module implements the migration strategy by:
1. Running the new extraction pipeline in parallel with the existing system
2. Writing to a separate Neo4j database instance (neo4j_enhanced)
3. Processing papers in batches with parallel workers
4. Supporting configuration flags to enable/disable the enhanced pipeline

Requirements: 16.1, 17.2
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from nlp.enriched_record import EnrichedPaperRecord
from graph.enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphEdge
from graph.reified_claims import ScientificClaim
from neo4j import GraphDatabase


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for the enhanced knowledge graph pipeline.
    
    Requirements: 16.1 (parallel execution with separate database)
    """
    # Enable/disable enhanced pipeline
    enabled: bool = True
    
    # Neo4j connection for enhanced database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j_enhanced"  # Separate database instance
    
    # Batch processing configuration (Requirement 17.2)
    batch_size: int = 100  # Papers per batch
    num_workers: int = 8   # Parallel workers (8-16 recommended)
    
    # Extraction configuration
    extraction_method: str = "regex_ner"
    extractor_version: str = "1.0"
    
    # Output configuration
    output_dir: Path = Path("data/processed")
    save_intermediate: bool = True  # Save edges/claims to JSON
    
    # Neo4j batch loading (Requirement 17.5)
    neo4j_batch_size: int = 10000  # Nodes/edges per transaction
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            PipelineConfig instance with values from environment
        """
        import os
        
        return cls(
            enabled=os.getenv("ENHANCED_PIPELINE_ENABLED", "true").lower() == "true",
            neo4j_uri=os.getenv("NEO4J_ENHANCED_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_ENHANCED_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_ENHANCED_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_ENHANCED_DATABASE", "neo4j_enhanced"),
            batch_size=int(os.getenv("ENHANCED_BATCH_SIZE", "100")),
            num_workers=int(os.getenv("ENHANCED_NUM_WORKERS", "8")),
            extraction_method=os.getenv("ENHANCED_EXTRACTION_METHOD", "regex_ner"),
            extractor_version=os.getenv("ENHANCED_EXTRACTOR_VERSION", "1.0"),
        )


class EnhancedNeo4jLoader:
    """
    Neo4j loader for enhanced graph edges and reified claims.
    
    This loader writes to a separate Neo4j database instance to avoid
    interfering with the existing system during migration.
    
    Requirements: 16.1 (separate database), 17.5 (batch loading)
    """
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        batch_size: int = 10000
    ):
        """
        Initialize Neo4j loader.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
            batch_size: Number of nodes/edges per transaction
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.batch_size = batch_size
        logger.info(f"Connected to Neo4j at {uri}, database: {database}")
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        logger.info("Closed Neo4j connection")
    
    def create_indexes(self):
        """
        Create indexes for efficient querying.
        
        Requirement 12.1, 12.2, 12.3, 12.4: Create indexes on key properties
        """
        with self.driver.session(database=self.database) as session:
            # Paper indexes (Requirement 12.1)
            session.run("CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)")
            session.run("CREATE INDEX paper_article_type IF NOT EXISTS FOR (p:Paper) ON (p.article_type)")
            session.run("CREATE INDEX paper_data_availability IF NOT EXISTS FOR (p:Paper) ON (p.data_availability)")
            
            # Entity indexes (Requirement 12.2)
            session.run("CREATE INDEX taxon_name IF NOT EXISTS FOR (t:Taxon) ON (t.name)")
            session.run("CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)")
            session.run("CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)")
            
            # Composite indexes for common query patterns (Requirement 12.4)
            session.run(
                "CREATE INDEX paper_year_type IF NOT EXISTS "
                "FOR (p:Paper) ON (p.year, p.article_type)"
            )
            
            # Relationship property indexes (Requirement 12.3)
            # Individual relationship property indexes for REPORTS_ASSOCIATION
            session.run("CREATE INDEX rel_association_confidence IF NOT EXISTS FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.confidence)")
            session.run("CREATE INDEX rel_association_p_value IF NOT EXISTS FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.p_value)")
            
            # Individual relationship property indexes for REPORTS_INTERVENTION_EFFECT
            session.run("CREATE INDEX rel_intervention_confidence IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.confidence)")
            session.run("CREATE INDEX rel_intervention_p_value IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.p_value)")
            session.run("CREATE INDEX rel_intervention_type IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.intervention_type)")
            
            # Individual relationship property indexes for USES_METHODOLOGY
            session.run("CREATE INDEX rel_methodology_confidence IF NOT EXISTS FOR ()-[r:USES_METHODOLOGY]-() ON (r.confidence)")
            
            # Composite index on (evidence_strength, consensus_confidence) for REPORTS_ASSOCIATION (Requirement 12.4)
            session.run("CREATE INDEX rel_association_evidence_consensus_composite IF NOT EXISTS FOR ()-[r:REPORTS_ASSOCIATION]-() ON (r.evidence_strength, r.consensus_confidence)")
            
            # Composite index on (evidence_strength, consensus_confidence) for REPORTS_INTERVENTION_EFFECT (Requirement 12.4)
            session.run("CREATE INDEX rel_intervention_evidence_consensus_composite IF NOT EXISTS FOR ()-[r:REPORTS_INTERVENTION_EFFECT]-() ON (r.evidence_strength, r.consensus_confidence)")
            
            logger.info("Created Neo4j indexes")
    
    def load_edges(self, edges: List[EnhancedGraphEdge]):
        """
        Load enhanced graph edges into Neo4j in batches.
        
        Requirement 17.5: Batch import with 10,000 nodes/edges per transaction
        
        Args:
            edges: List of EnhancedGraphEdge objects to load
        """
        if not edges:
            logger.warning("No edges to load")
            return
        
        logger.info(f"Loading {len(edges)} edges into Neo4j...")
        
        # Process in batches
        for i in range(0, len(edges), self.batch_size):
            batch = edges[i:i + self.batch_size]
            self._load_edge_batch(batch)
            logger.info(f"Loaded batch {i // self.batch_size + 1} ({len(batch)} edges)")
        
        logger.info(f"Successfully loaded {len(edges)} edges")
    
    def _load_edge_batch(self, edges: List[EnhancedGraphEdge]):
        """
        Load a batch of edges in a single transaction.
        
        Args:
            edges: Batch of edges to load
        """
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for edge in edges:
                    edge_dict = edge.to_dict()

                    # Determine target label based on relation type
                    if edge.relation == "REPORTS_ASSOCIATION":
                        target_label = "Taxon"
                    elif edge.relation == "REPORTS_INTERVENTION_EFFECT":
                        target_label = "Taxon"
                    elif edge.relation == "USES_METHODOLOGY":
                        target_label = "Method"
                    else:
                        target_label = "Entity"

                    # Create Paper source node with label and paper metadata
                    tx.run(
                        """
                        MERGE (source:Paper {id: $source_id})
                        ON CREATE SET source.created_at = datetime(),
                                      source.year = $year,
                                      source.article_type = $article_type,
                                      source.data_availability = $data_availability,
                                      source.accession_numbers = $accession_numbers
                        ON MATCH SET  source.year = $year,
                                      source.article_type = $article_type,
                                      source.data_availability = $data_availability,
                                      source.accession_numbers = $accession_numbers
                        """,
                        source_id=edge.source,
                        year=edge_dict.get("year"),
                        article_type=edge_dict.get("article_type"),
                        data_availability=edge_dict.get("data_availability"),
                        accession_numbers=edge_dict.get("accession_numbers", [])
                    )

                    # Create target node with correct label
                    tx.run(
                        f"""
                        MERGE (target:{target_label} {{id: $target_id}})
                        ON CREATE SET target.created_at = datetime(),
                                      target.name = $target_name
                        """,
                        target_id=edge.target,
                        target_name=edge.target
                    )

                    # Create relationship with all properties
                    tx.run(
                        f"""
                        MATCH (source:Paper {{id: $source_id}})
                        MATCH (target:{target_label} {{id: $target_id}})
                        CREATE (source)-[r:{edge.relation}]->(target)
                        SET r = $properties
                        """,
                        source_id=edge.source,
                        target_id=edge.target,
                        properties=edge_dict
                    )

                tx.commit()
    
    def load_claims(self, claims: List[ScientificClaim]):
        """
        Load reified claims into Neo4j as first-class nodes.
        
        Requirement 4.1: Create reified claim nodes
        
        Args:
            claims: List of ScientificClaim objects to load
        """
        if not claims:
            logger.warning("No claims to load")
            return
        
        logger.info(f"Loading {len(claims)} reified claims into Neo4j...")
        
        # Process in batches
        for i in range(0, len(claims), self.batch_size):
            batch = claims[i:i + self.batch_size]
            self._load_claim_batch(batch)
            logger.info(f"Loaded batch {i // self.batch_size + 1} ({len(batch)} claims)")
        
        logger.info(f"Successfully loaded {len(claims)} claims")
    
    def _load_claim_batch(self, claims: List[ScientificClaim]):
        """
        Load a batch of claims in a single transaction.
        
        Args:
            claims: Batch of claims to load
        """
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for claim in claims:
                    # Create claim node
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
                            contradicting_paper_count: $contradicting_paper_count
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
                        first_reported=claim.first_reported.isoformat() if isinstance(claim.first_reported, datetime) else claim.first_reported,
                        last_updated=claim.last_updated.isoformat() if isinstance(claim.last_updated, datetime) else claim.last_updated,
                        supporting_paper_count=len(claim.supporting_papers),
                        contradicting_paper_count=len(claim.contradicting_papers)
                    )
                    
                    # Link claim to supporting papers
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
                    
                    # Link claim to contradicting papers
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


class EnhancedKGPipeline:
    """
    Enhanced knowledge graph pipeline with parallel execution.
    
    This pipeline runs in parallel with the existing system, writing to a
    separate Neo4j database instance. It processes papers in batches with
    parallel workers for improved throughput.
    
    Requirements: 16.1 (parallel execution), 17.2 (batch processing with parallel workers)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the enhanced pipeline.
        
        Args:
            config: Pipeline configuration (defaults to environment-based config)
        """
        self.config = config or PipelineConfig.from_env()
        
        if not self.config.enabled:
            logger.info("Enhanced pipeline is disabled")
            return
        
        # Initialize Neo4j loader
        self.neo4j_loader = EnhancedNeo4jLoader(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database,
            batch_size=self.config.neo4j_batch_size
        )
        
        # Create indexes
        self.neo4j_loader.create_indexes()
        
        logger.info(
            f"Initialized enhanced pipeline: "
            f"batch_size={self.config.batch_size}, "
            f"num_workers={self.config.num_workers}"
        )
    
    def run(
        self,
        enriched_papers: List[Any],
        load_to_neo4j: bool = True
    ) -> Dict[str, Any]:
        """
        Run the enhanced knowledge graph pipeline.
        
        This method:
        1. Converts input to EnrichedPaperRecord objects
        2. Processes papers in batches with parallel workers
        3. Creates reified claims from aggregated evidence
        4. Loads edges and claims into Neo4j
        5. Saves intermediate results to JSON files
        
        Requirements: 16.1, 17.2
        
        Args:
            enriched_papers: List of enriched paper records (dicts or objects)
            load_to_neo4j: Whether to load results into Neo4j (default: True)
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        if not self.config.enabled:
            logger.warning("Enhanced pipeline is disabled, skipping execution")
            return {"status": "disabled"}
        
        start_time = datetime.now()
        logger.info(f"Starting enhanced pipeline with {len(enriched_papers)} papers")
        
        # Convert to EnrichedPaperRecord objects
        records = [
            EnrichedPaperRecord(**x) if isinstance(x, dict) else x
            for x in enriched_papers
        ]
        
        # Process papers in batches with parallel workers
        all_edges = []
        all_builders = []
        
        # Split into batches (Requirement 17.2: batches of 100)
        batches = [
            records[i:i + self.config.batch_size]
            for i in range(0, len(records), self.config.batch_size)
        ]
        
        logger.info(f"Processing {len(batches)} batches with {self.config.num_workers} workers")
        
        # Process batches in parallel (Requirement 17.2: 8-16 parallel workers)
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, batch_idx): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_edges, builder = future.result()
                    all_edges.extend(batch_edges)
                    all_builders.append(builder)
                    logger.info(
                        f"Completed batch {batch_idx + 1}/{len(batches)} "
                        f"({len(batch_edges)} edges)"
                    )
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
        
        # Merge all builders to create reified claims
        logger.info("Creating reified claims from aggregated evidence...")
        merged_builder = self._merge_builders(all_builders)
        claims = merged_builder.create_reified_claims()
        
        # Get statistics
        stats = merged_builder.get_statistics()
        stats["total_claims"] = len(claims)
        stats["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Pipeline statistics: {stats}")
        
        # Save intermediate results
        if self.config.save_intermediate:
            self._save_results(all_edges, claims, stats)
        
        # Load into Neo4j
        if load_to_neo4j:
            logger.info("Loading results into Neo4j...")
            self.neo4j_loader.load_edges(all_edges)
            self.neo4j_loader.load_claims(claims)
            logger.info("Successfully loaded results into Neo4j")
        
        return {
            "status": "success",
            "statistics": stats,
            "edges_count": len(all_edges),
            "claims_count": len(claims),
            "processing_time_seconds": stats["processing_time_seconds"]
        }
    
    def _process_batch(
        self,
        batch: List[EnrichedPaperRecord],
        batch_idx: int
    ) -> tuple[List[EnhancedGraphEdge], EnhancedGraphBuilder]:
        """
        Process a batch of papers with a dedicated builder.
        
        Args:
            batch: List of papers to process
            batch_idx: Batch index for logging
        
        Returns:
            Tuple of (edges, builder)
        """
        builder = EnhancedGraphBuilder(
            extraction_method=self.config.extraction_method,
            extractor_version=self.config.extractor_version
        )
        
        edges = builder.process_papers(batch)
        
        return edges, builder
    
    def _merge_builders(
        self,
        builders: List[EnhancedGraphBuilder]
    ) -> EnhancedGraphBuilder:
        """
        Merge multiple builders into a single builder for reification.
        
        This allows creating reified claims from relationships extracted
        across all batches.
        
        Args:
            builders: List of builders from parallel processing
        
        Returns:
            Merged builder with all relationships
        """
        merged = EnhancedGraphBuilder(
            extraction_method=self.config.extraction_method,
            extractor_version=self.config.extractor_version
        )
        
        # Merge relationships and edges
        for builder in builders:
            merged.relationships.extend(builder.relationships)
            merged.edges.extend(builder.edges)
            
            # Merge relationship index
            for key, rels in builder.relationship_index.items():
                merged.relationship_index[key].extend(rels)
        
        return merged
    
    def _save_results(
        self,
        edges: List[EnhancedGraphEdge],
        claims: List[ScientificClaim],
        stats: Dict[str, Any]
    ):
        """
        Save edges, claims, and statistics to JSON files.
        
        Args:
            edges: List of enhanced graph edges
            claims: List of reified claims
            stats: Pipeline statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save edges
        edges_path = self.config.output_dir / f"enhanced_edges_{timestamp}.json"
        with open(edges_path, "w") as f:
            json.dump(
                [edge.to_dict() for edge in edges],
                f,
                indent=2,
                default=str
            )
        logger.info(f"Saved edges to {edges_path}")
        
        # Save claims
        claims_path = self.config.output_dir / f"enhanced_claims_{timestamp}.json"
        with open(claims_path, "w") as f:
            json.dump(
                [claim.model_dump() for claim in claims],
                f,
                indent=2,
                default=str
            )
        logger.info(f"Saved claims to {claims_path}")
        
        # Save statistics
        stats_path = self.config.output_dir / f"enhanced_stats_{timestamp}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Saved statistics to {stats_path}")
    
    def close(self):
        """Close the pipeline and cleanup resources."""
        if self.config.enabled:
            self.neo4j_loader.close()
            logger.info("Closed enhanced pipeline")


# Convenience function for running the pipeline
def run_enhanced_pipeline(
    enriched_papers: List[Any],
    config: Optional[PipelineConfig] = None,
    load_to_neo4j: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run the enhanced pipeline.
    
    Args:
        enriched_papers: List of enriched paper records
        config: Pipeline configuration (optional)
        load_to_neo4j: Whether to load results into Neo4j
    
    Returns:
        Dictionary with pipeline results
    """
    pipeline = EnhancedKGPipeline(config)
    try:
        return pipeline.run(enriched_papers, load_to_neo4j)
    finally:
        pipeline.close()
