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
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from nlp.enriched_record import EnrichedPaperRecord
from graph.enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphEdge
from graph.reified_claims import ScientificClaim
from graph.triple_promotion_models import PromotedTriple, OpenWorldClaim
from graph.triple_promoter import TriplePromoter
from graph.evidence_strength_classifier import EvidenceStrengthClassifier
from graph.entity_normalizer import EntityNormalizer
from graph.predicate_registry import PredicateRegistry
from neo4j import GraphDatabase


logger = logging.getLogger(__name__)


def _to_neo4j_label(entity_type: str) -> str:
    """Convert entity type string to CamelCase Neo4j node label.
    Strips any characters invalid in Neo4j labels (e.g. | from LLM responses).
    """
    # Take only the first type if LLM returns "disease|condition" style
    clean = (entity_type or "entity").split("|")[0].split("/")[0].strip()
    # Remove any remaining non-alphanumeric/underscore/space characters
    clean = re.sub(r'[^a-zA-Z0-9_ ]', '', clean)
    return "".join(word.capitalize() for word in clean.replace("_", " ").split()) or "Entity"


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
    neo4j_database: str = "neo4j"  # Neo4j Community Edition only supports "neo4j"
    
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
            
            # Indexes for common open-world triple entity types
            common_types = ["Metabolite", "Gene", "Protein", "Pathway", "ImmuneCell",
                            "Biomarker", "Population", "DietaryComponent", "ClinicalOutcome"]
            for node_type in common_types:
                try:
                    session.run(f"CREATE INDEX {node_type.lower()}_name IF NOT EXISTS FOR (n:{node_type}) ON (n.name)")
                except Exception:
                    pass  # Index may already exist or node type not yet present
            
            logger.info("Created Neo4j indexes")
    
    def load_edges(self, edges: List[EnhancedGraphEdge], title_lookup: Dict[str, str] = None):
        """
        Load enhanced graph edges into Neo4j in batches.
        
        Requirement 17.5: Batch import with 10,000 nodes/edges per transaction
        
        Args:
            edges: List of EnhancedGraphEdge objects to load
            title_lookup: Mapping of DOI / doi:DOI keys to paper title strings
        """
        if not edges:
            logger.warning("No edges to load")
            return
        
        if title_lookup is None:
            title_lookup = {}

        logger.info(f"Loading {len(edges)} edges into Neo4j...")
        
        # Process in batches
        for i in range(0, len(edges), self.batch_size):
            batch = edges[i:i + self.batch_size]
            self._load_edge_batch(batch, title_lookup)
            logger.info(f"Loaded batch {i // self.batch_size + 1} ({len(batch)} edges)")
        
        logger.info(f"Successfully loaded {len(edges)} edges")
    
    def _load_edge_batch(self, edges: List[EnhancedGraphEdge], title_lookup: Dict[str, str] = None):
        """
        Load a batch of edges in a single transaction.
        
        Args:
            edges: Batch of edges to load
            title_lookup: Mapping of DOI / doi:DOI keys to paper title strings
        """
        if title_lookup is None:
            title_lookup = {}

        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for edge in edges:
                    edge_dict = edge.to_dict()

                    # Resolve paper title: prefer title_lookup (keyed by doi or doi:doi),
                    # fall back to edge.source so the node always gets a meaningful name.
                    paper_title = title_lookup.get(edge.source, edge.source)

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
                                      source.name = $title,
                                      source.title = $title,
                                      source.year = $year,
                                      source.article_type = $article_type,
                                      source.data_availability = $data_availability,
                                      source.accession_numbers = $accession_numbers
                        ON MATCH SET  source.name = $title,
                                      source.title = $title,
                                      source.year = $year,
                                      source.article_type = $article_type,
                                      source.data_availability = $data_availability,
                                      source.accession_numbers = $accession_numbers
                        """,
                        source_id=edge.source,
                        title=paper_title,
                        year=edge_dict.get("year"),
                        article_type=edge_dict.get("article_type"),
                        data_availability=edge_dict.get("data_availability"),
                        accession_numbers=edge_dict.get("accession_numbers", [])
                    )

                    # Create target node with correct label and canonical name
                    tx.run(
                        f"""
                        MERGE (target:{target_label} {{id: $target_id}})
                        ON CREATE SET target.created_at = datetime(),
                                      target.name = $target_name,
                                      target.canonical_name = $canonical_name,
                                      target.ontology = $ontology,
                                      target.grounded = $grounded
                        ON MATCH SET  target.canonical_name = $canonical_name,
                                      target.ontology = $ontology,
                                      target.grounded = $grounded
                        """,
                        target_id=edge.target,
                        target_name=edge_dict.get("target_canonical", edge.target),
                        canonical_name=edge_dict.get("target_canonical", edge.target),
                        ontology=edge_dict.get("target_ontology"),
                        grounded=edge_dict.get("target_grounded", False)
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
        """Load reified claims into Neo4j as first-class nodes."""
        if not claims:
            logger.warning("No claims to load")
            return
        
        logger.info(f"Loading {len(claims)} reified claims into Neo4j...")
        for i in range(0, len(claims), self.batch_size):
            batch = claims[i:i + self.batch_size]
            self._load_claim_batch(batch)
            logger.info(f"Loaded batch {i // self.batch_size + 1} ({len(batch)} claims)")
        logger.info(f"Successfully loaded {len(claims)} claims")

    def _load_claim_batch(self, claims: List[ScientificClaim]):
        """Load a batch of ScientificClaim nodes in a single transaction."""
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for claim in claims:
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
                        first_reported=claim.first_reported.isoformat() if hasattr(claim.first_reported, 'isoformat') else claim.first_reported,
                        last_updated=claim.last_updated.isoformat() if hasattr(claim.last_updated, 'isoformat') else claim.last_updated,
                        supporting_paper_count=len(claim.supporting_papers),
                        contradicting_paper_count=len(claim.contradicting_papers)
                    )

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

    def load_open_world_triples(self, triples: List[Dict]) -> None:
        """
        Load open-world (subject, predicate, object) triples into Neo4j.

        Each triple is stored as a RELATES_TO (or canonical_predicate) relationship
        between two Entity nodes. Subject/object entity types are stored as node labels.
        Novel predicates are preserved as raw_predicate property.
        """
        if not triples:
            return

        logger.info(f"Loading {len(triples)} open-world triples into Neo4j...")

        for i in range(0, len(triples), self.batch_size):
            batch = triples[i:i + self.batch_size]
            with self.driver.session(database=self.database) as session:
                with session.begin_transaction() as tx:
                    for t in batch:
                        subject = t.get("subject", "").strip()
                        object_ = t.get("object", "").strip()
                        if not subject or not object_:
                            continue

                        canonical_pred = t.get("canonical_predicate", "RELATES_TO")
                        # Use canonical predicate as relationship type if it's a known type
                        # otherwise use RELATES_TO
                        rel_type = canonical_pred if canonical_pred else "RELATES_TO"
                        # Sanitize: keep only uppercase letters, digits, underscores (valid Cypher rel type chars)
                        rel_type = re.sub(r'[^A-Z0-9_]', '_', rel_type.replace("-", "_").replace(" ", "_").upper())
                        # Relationship types cannot start with a digit
                        if rel_type and rel_type[0].isdigit():
                            rel_type = "REL_" + rel_type
                        # Fallback for empty
                        if not rel_type:
                            rel_type = "RELATES_TO"

                        subj_label = _to_neo4j_label(t.get("subject_type") or "entity")
                        obj_label = _to_neo4j_label(t.get("object_type") or "entity")

                        tx.run(
                            f"""
                            MERGE (s:{subj_label} {{name: $subject}})
                            MERGE (o:{obj_label} {{name: $object}})
                            CREATE (s)-[r:{rel_type} {{
                                raw_predicate: $raw_predicate,
                                canonical_predicate: $canonical_predicate,
                                predicate_category: $predicate_category,
                                is_novel_predicate: $is_novel_predicate,
                                confidence: $confidence,
                                evidence: $evidence,
                                paper_id: $paper_id,
                                section_type: $section_type,
                                extraction_method: 'llm_triple_extractor',
                                extracted_at: $extracted_at
                            }}]->(o)
                            """,
                            subject=subject,
                            object=object_,
                            raw_predicate=t.get("predicate", ""),
                            canonical_predicate=canonical_pred,
                            predicate_category=t.get("predicate_category", "generic"),
                            is_novel_predicate=t.get("is_novel_predicate", False),
                            confidence=t.get("confidence", 0.7),
                            evidence=t.get("evidence", "")[:500],
                            paper_id=t.get("paper_id", ""),
                            section_type=t.get("section_type", "unknown"),
                            extracted_at=t.get("extracted_at", ""),
                        )
                    tx.commit()

        logger.info(f"Successfully loaded {len(triples)} open-world triples")
    
    def load_promoted_triples(self, promoted_triples: List[PromotedTriple]) -> None:
        """
        Load fully promoted TriplePromoter output into Neo4j.

        Each PromotedTriple carries normalized entity IDs, canonical predicate,
        full provenance, and evidence strength. Stored as a typed relationship
        (canonical form or RELATES_TO) between subject and object nodes.

        Args:
            promoted_triples: List of PromotedTriple objects from TriplePromoter
        """
        if not promoted_triples:
            return

        logger.info(f"Loading {len(promoted_triples)} promoted triples into Neo4j...")

        for i in range(0, len(promoted_triples), self.batch_size):
            batch = promoted_triples[i:i + self.batch_size]
            with self.driver.session(database=self.database) as session:
                with session.begin_transaction() as tx:
                    for pt in batch:
                        subject_label = _to_neo4j_label(pt.subject_type)
                        object_label = _to_neo4j_label(pt.object_type)

                        # Sanitize relationship type
                        rel_type = re.sub(
                            r'[^A-Z0-9_]', '_',
                            pt.relationship_type.replace("-", "_").replace(" ", "_").upper()
                        )
                        if rel_type and rel_type[0].isdigit():
                            rel_type = "REL_" + rel_type
                        if not rel_type:
                            rel_type = "RELATES_TO"

                        prov = pt.provenance
                        tx.run(
                            f"""
                            MERGE (s:{subject_label} {{id: $subject_id}})
                            ON CREATE SET s.name = $subject_name,
                                          s.ontology = $subject_ontology,
                                          s.grounded = $subject_grounded
                            MERGE (o:{object_label} {{id: $object_id}})
                            ON CREATE SET o.name = $object_name,
                                          o.ontology = $object_ontology,
                                          o.grounded = $object_grounded
                            CREATE (s)-[r:{rel_type} {{
                                raw_predicate: $raw_predicate,
                                canonical_predicate: $canonical_predicate,
                                predicate_category: $predicate_category,
                                is_novel_predicate: $is_novel_predicate,
                                relationship_type: $relationship_type,
                                confidence: $confidence,
                                evidence_strength: $evidence_strength,
                                paper_id: $paper_id,
                                section_type: $section_type,
                                extraction_method: $extraction_method,
                                extraction_timestamp: $extraction_timestamp,
                                extractor_version: $extractor_version,
                                source_sentence: $source_sentence,
                                sentence_offset: $sentence_offset,
                                surrounding_context: $surrounding_context,
                                validation_status: $validation_status,
                                subject_grounded: $subject_grounded,
                                object_grounded: $object_grounded,
                                subject_ontology: $subject_ontology,
                                object_ontology: $object_ontology,
                                extracted_at: $extracted_at
                            }}]->(o)
                            """,
                            subject_id=pt.subject_id,
                            subject_name=pt.subject_name,
                            subject_ontology=pt.subject_ontology,
                            subject_grounded=pt.subject_grounded,
                            object_id=pt.object_id,
                            object_name=pt.object_name,
                            object_ontology=pt.object_ontology,
                            object_grounded=pt.object_grounded,
                            raw_predicate=pt.raw_predicate,
                            canonical_predicate=pt.canonical_predicate,
                            predicate_category=pt.predicate_category,
                            is_novel_predicate=pt.is_novel_predicate,
                            relationship_type=pt.relationship_type,
                            confidence=pt.confidence,
                            evidence_strength=pt.evidence_strength,
                            paper_id=pt.paper_id,
                            section_type=pt.section_type,
                            extraction_method=prov.extraction_method,
                            extraction_timestamp=prov.extraction_timestamp.isoformat(),
                            extractor_version=prov.extractor_version,
                            source_sentence=prov.source_sentence or "",
                            sentence_offset=prov.sentence_offset,
                            surrounding_context=prov.surrounding_context or "",
                            validation_status=prov.validation_status or "unvalidated",
                            extracted_at=pt.extracted_at,
                        )
                    tx.commit()

        logger.info(f"Successfully loaded {len(promoted_triples)} promoted triples")

    def load_open_world_claims(self, claims: List[OpenWorldClaim]) -> None:
        """
        Load OpenWorldClaim aggregation nodes into Neo4j.

        Creates an ``OpenWorldClaim`` node for each claim and attaches
        ``SUPPORTED_BY_TRIPLE`` relationships to each supporting paper.

        Args:
            claims: List of OpenWorldClaim objects from TriplePromoter.aggregate_claims
        """
        if not claims:
            return

        logger.info(f"Loading {len(claims)} open-world claims into Neo4j...")

        for i in range(0, len(claims), self.batch_size):
            batch = claims[i:i + self.batch_size]
            with self.driver.session(database=self.database) as session:
                with session.begin_transaction() as tx:
                    for claim in batch:
                        tx.run(
                            """
                            CREATE (c:OpenWorldClaim {
                                claim_id: $claim_id,
                                claim_type: $claim_type,
                                subject_id: $subject_id,
                                subject_name: $subject_name,
                                canonical_predicate: $canonical_predicate,
                                object_id: $object_id,
                                object_name: $object_name,
                                paper_count: $paper_count,
                                consensus_confidence: $consensus_confidence,
                                evidence_strength: $evidence_strength,
                                first_reported: $first_reported,
                                last_updated: $last_updated
                            })
                            """,
                            claim_id=claim.claim_id,
                            claim_type=claim.claim_type,
                            subject_id=claim.subject_id,
                            subject_name=claim.subject_name,
                            canonical_predicate=claim.canonical_predicate,
                            object_id=claim.object_id,
                            object_name=claim.object_name,
                            paper_count=claim.paper_count,
                            consensus_confidence=claim.consensus_confidence,
                            evidence_strength=claim.evidence_strength,
                            first_reported=claim.first_reported,
                            last_updated=claim.last_updated,
                        )

                        # Link claim to each supporting paper via SUPPORTED_BY_TRIPLE
                        for evidence_item in claim.evidence_items:
                            tx.run(
                                """
                                MATCH (c:OpenWorldClaim {claim_id: $claim_id})
                                MERGE (p:Paper {id: $paper_id})
                                CREATE (c)-[:SUPPORTED_BY_TRIPLE {
                                    confidence: $confidence,
                                    evidence_strength: $evidence_strength,
                                    section_type: $section_type,
                                    source_sentence: $source_sentence
                                }]->(p)
                                """,
                                claim_id=claim.claim_id,
                                paper_id=evidence_item.paper_id,
                                confidence=evidence_item.confidence,
                                evidence_strength=evidence_item.evidence_strength,
                                section_type=evidence_item.section_type,
                                source_sentence=evidence_item.source_sentence[:500],
                            )

                    tx.commit()

        logger.info(f"Successfully loaded {len(claims)} open-world claims")


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

        # Wire TriplePromoter for evidence strength classification of LLM triples.
        # Shared across all batch workers — EntityNormalizer and PredicateRegistry
        # both use SQLite-backed caches that are safe for concurrent reads.
        import os
        promotion_threshold = int(os.getenv("PREDICATE_PROMOTION_THRESHOLD", "5"))
        self.triple_promoter = TriplePromoter(
            entity_normalizer=EntityNormalizer(),
            predicate_registry=PredicateRegistry(),
            evidence_classifier=EvidenceStrengthClassifier(),
            promotion_threshold=promotion_threshold,
        )
        logger.info(
            f"Initialized TriplePromoter (promotion_threshold={promotion_threshold})"
        )

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

        # ── Entity cache pre-warming ──────────────────────────────────────────
        # Collect every unique (entity_text, entity_type) pair from NER output
        # and normalize them all before the parallel batch loop starts.
        #
        # Why this helps:
        #   Each batch worker calls entity_normalizer.normalize() for every extracted
        #   entity. On a cache miss, this costs one NCBI API call + 0.12–0.34s sleep.
        #   By pre-warming the SQLite cache here — using a small thread pool that
        #   respects the NCBI rate limit — every batch worker subsequently hits
        #   cache (microseconds) instead of the API (hundreds of milliseconds).
        #
        # Thread safety: EntityNormalizer._cache_store uses SQLite with
        # INSERT OR REPLACE, which is safe for concurrent writers on the same DB.
        self._prewarm_entity_cache(records)

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
        # Count open-world triples across all builders
        all_ow_triples_count = sum(len(b.get_open_world_triples()) for b in all_builders)
        stats["open_world_triples"] = all_ow_triples_count
        # Count promoted triples and open-world claims across all builders
        all_promoted_count = sum(len(b.promoted_triples) for b in all_builders)
        all_ow_claims_count = sum(len(b.open_world_claims) for b in all_builders)
        stats["promoted_triples"] = all_promoted_count
        stats["open_world_claims"] = all_ow_claims_count
        stats["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Pipeline statistics: {stats}")
        
        # Build title_lookup once from records: maps doi, doi:doi -> paper title
        title_lookup: Dict[str, str] = {}
        for paper in records:
            doi = getattr(paper, "doi", None) or ""
            pmid = getattr(paper, "pmid", None) or ""
            title = getattr(paper, "title", "") or ""
            if doi and title:
                title_lookup[doi] = title
                title_lookup[f"doi:{doi}"] = title
            if pmid and title:
                title_lookup[str(pmid)] = title
                title_lookup[f"pmid:{pmid}"] = title

        # Save intermediate results
        if self.config.save_intermediate:
            self._save_results(all_edges, claims, stats, title_lookup)
        
        # Load into Neo4j
        if load_to_neo4j:
            logger.info("Loading results into Neo4j...")
            self.neo4j_loader.load_edges(all_edges, title_lookup)
            self.neo4j_loader.load_claims(claims)
            # Load open-world triples from all builders
            all_ow_triples = []
            for builder in all_builders:
                all_ow_triples.extend(builder.get_open_world_triples())
            if all_ow_triples:
                self.neo4j_loader.load_open_world_triples(all_ow_triples)
                logger.info(f"Loaded {len(all_ow_triples)} open-world triples into Neo4j")

            # Load promoted triples (PromotedTriple objects) from all builders
            all_promoted_triples = []
            for builder in all_builders:
                all_promoted_triples.extend(builder.promoted_triples)
            if all_promoted_triples:
                self.neo4j_loader.load_promoted_triples(all_promoted_triples)
                logger.info(f"Loaded {len(all_promoted_triples)} promoted triples into Neo4j")

            # Load open-world claims (OpenWorldClaim objects) from all builders
            all_ow_claims = []
            for builder in all_builders:
                all_ow_claims.extend(builder.open_world_claims)
            if all_ow_claims:
                self.neo4j_loader.load_open_world_claims(all_ow_claims)
                logger.info(f"Loaded {len(all_ow_claims)} open-world claims into Neo4j")

            logger.info("Successfully loaded results into Neo4j")
        
        return {
            "status": "success",
            "statistics": stats,
            "edges_count": len(all_edges),
            "claims_count": len(claims),
            "promoted_triples_count": all_promoted_count,
            "open_world_claims_count": all_ow_claims_count,
            "processing_time_seconds": stats["processing_time_seconds"]
        }
    
    def _prewarm_entity_cache(self, records: List[Any]) -> None:
        """
        Pre-warm the EntityNormalizer SQLite cache before parallel batch processing.

        Collects every unique (entity_text, entity_type) pair from the NER output
        of all enriched papers and normalizes them sequentially — respecting the
        NCBI rate limit — so that all batch workers subsequently get instant cache
        hits instead of blocking on NCBI API calls.

        Uses a dedicated EntityNormalizer instance (not shared with batch workers)
        to avoid SQLite write contention during pre-warming.

        Args:
            records: All EnrichedPaperRecord objects to be processed.
        """
        # Collect unique (text, type) pairs from all papers
        seen: set = set()
        entity_pairs: List[tuple] = []

        ENTITY_TYPE_MAP = {
            "taxon": "taxon",
            "disease": "disease",
            "method": "method",
            "gene": "gene",
            "protein": "protein",
            "metabolite": "metabolite",
            "treatment": "treatment",
        }

        for record in records:
            entities = getattr(record, "entities", []) or []
            for ent in entities:
                raw_type = getattr(ent, "label", "unknown") or "unknown"
                entity_type = ENTITY_TYPE_MAP.get(raw_type.lower(), raw_type.lower())
                text = getattr(ent, "text", "") or ""
                if not text:
                    continue
                key = (text.lower(), entity_type)
                if key not in seen:
                    seen.add(key)
                    entity_pairs.append((text, entity_type))

        if not entity_pairs:
            return

        logger.info(
            f"Pre-warming entity cache for {len(entity_pairs)} unique entities "
            f"(eliminates NCBI wait time in batch workers)..."
        )

        # Use a dedicated normalizer so pre-warming writes to the shared SQLite
        # cache without conflicting with batch-worker reads (SQLite WAL handles this).
        prewarm_normalizer = EntityNormalizer()
        warmed = 0
        skipped = 0

        for text, entity_type in entity_pairs:
            try:
                # Cache lookup first — skip API call if already cached
                cached = prewarm_normalizer._cache_lookup(text, entity_type)
                if cached is not None:
                    skipped += 1
                    continue
                prewarm_normalizer.normalize(text, entity_type)
                warmed += 1
            except Exception as exc:
                logger.debug(f"Pre-warm failed for {text!r} ({entity_type}): {exc}")

        logger.info(
            f"Entity cache pre-warm complete: {warmed} new entries, "
            f"{skipped} already cached. "
            f"Batch workers will now use cache hits only."
        )

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

        # Wire the shared TriplePromoter so LLM triples get evidence strength
        # classification (strong/moderate/weak) instead of always landing as "weak".
        if hasattr(self, "triple_promoter") and self.triple_promoter is not None:
            builder.set_triple_promoter(self.triple_promoter)

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
        stats: Dict[str, Any],
        title_lookup: Dict[str, str] = None
    ):
        """
        Save edges, claims, statistics, entities, and relationships to JSON files.

        Args:
            edges: All extracted graph edges.
            claims: All reified scientific claims.
            stats: Pipeline statistics dictionary.
            title_lookup: Mapping of DOI / doi:DOI / pmid keys to paper title strings.
        """
        if title_lookup is None:
            title_lookup = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save edges
        edges_path = self.config.output_dir / f"enhanced_edges_{timestamp}.json"
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(
                [edge.to_dict() for edge in edges],
                f,
                indent=2,
                default=str
            )
        logger.info(f"Saved edges to {edges_path}")
        
        # Save claims
        claims_path = self.config.output_dir / f"enhanced_claims_{timestamp}.json"
        with open(claims_path, "w", encoding="utf-8") as f:
            json.dump(
                [claim.model_dump() for claim in claims],
                f,
                indent=2,
                default=str
            )
        logger.info(f"Saved claims to {claims_path}")
        
        # Save statistics
        stats_path = self.config.output_dir / f"enhanced_stats_{timestamp}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Saved statistics to {stats_path}")

        # ── Export entities (nodes) ──────────────────────────────────────────
        # Collect all unique entities from edges
        entities: Dict[str, Dict] = {}
        for edge in edges:
            d = edge.to_dict()
            # Source = Paper node
            src_id = edge.source
            if src_id not in entities:
                paper_title = title_lookup.get(src_id, src_id)
                entities[src_id] = {
                    "id": src_id,
                    "type": "Paper",
                    "name": paper_title,
                    "doi": src_id,
                    "year": d.get("year"),
                    "article_type": d.get("article_type"),
                    "data_availability": d.get("data_availability"),
                }
            # Target = Taxon / Method / Entity node
            tgt_id = edge.target
            if tgt_id not in entities:
                if edge.relation == "USES_METHODOLOGY":
                    node_type = "Method"
                elif edge.relation in ("REPORTS_ASSOCIATION", "REPORTS_INTERVENTION_EFFECT"):
                    node_type = "Taxon"
                else:
                    node_type = "Entity"
                entities[tgt_id] = {
                    "id": tgt_id,
                    "type": node_type,
                    "name": d.get("target_canonical", tgt_id),
                    "ontology": d.get("target_ontology"),
                    "ontology_id": d.get("target_ontology_id"),
                    "grounded": d.get("target_grounded", False),
                }

        entities_path = self.config.output_dir / f"entities_{timestamp}.json"
        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(list(entities.values()), f, indent=2, default=str)
        logger.info(f"Saved {len(entities)} entities to {entities_path}")

        # ── Export relationships ─────────────────────────────────────────────
        relationships = []
        for edge in edges:
            d = edge.to_dict()
            src_id = edge.source
            paper_title = title_lookup.get(src_id, src_id)
            relationships.append({
                "from_id": src_id,
                "from_name": paper_title,
                "relationship_type": edge.relation,
                "to_id": edge.target,
                "to_name": d.get("target_canonical", edge.target),
                "confidence": edge.confidence,
                "evidence_strength": edge.evidence_strength,
                "source_sentence": d.get("source_sentence", ""),
                "extraction_method": d.get("extraction_method", ""),
                "year": d.get("year"),
            })

        relationships_path = self.config.output_dir / f"relationships_{timestamp}.json"
        with open(relationships_path, "w", encoding="utf-8") as f:
            json.dump(relationships, f, indent=2, default=str)
        logger.info(f"Saved {len(relationships)} relationships to {relationships_path}")
    
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
