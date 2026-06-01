"""
Entry point for the Microbiome Literature Miner.

This script provides entry points for all three layers of the pipeline:

Layer 1 (Collection): Fetch papers from PubMed, Europe PMC, Semantic Scholar, bioRxiv
  - Run with: RUN_LAYER=1 python main.py
  - Output: data/processed/collected_YYYYMMDD_HHMMSS.json

Layer 2 (NLP Enrichment): Extract entities, classify articles, parse sections
  - Run with: RUN_LAYER=2 python main.py
  - Output: data/processed/enriched_YYYYMMDD_HHMMSS.json

Layer 3 (Knowledge Graph): Build enhanced knowledge graph with semantic relationships
  - Run with: RUN_LAYER=3 python main.py
  - Output: Neo4j database (neo4j_enhanced) + JSON files
  - Components: SemanticRelationshipExtractor → ProvenanceEncoder → 
                RelationshipReifier → EnhancedNeo4jLoader

Configuration:
  Layer 1: MAX_PER_SOURCE (default: 50)
  Layer 2: USE_NER_MODEL (default: false)
  Layer 3: ENHANCED_PIPELINE_ENABLED (default: true)
           LOAD_TO_NEO4J (default: true)
           ENHANCED_BATCH_SIZE (default: 100)
           ENHANCED_NUM_WORKERS (default: 8)

During development, set MAX_RESULTS_PER_SOURCE small (e.g. 20) so you're
not waiting 10 minutes for a full run while testing.
"""
import sys
from loguru import logger
from config import LOG_FILE, LOG_LEVEL
from collectors.orchestrator import CollectionOrchestrator

# ─── Configure Logging ───
# Loguru lets us log to both the console AND a file simultaneously.
# The file keeps a permanent record; the console gives live feedback.

logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
)
logger.add(
    LOG_FILE,
    level="DEBUG",
    rotation="10 MB",          # New log file after 10 MB
    retention="30 days",       # Keep logs for 30 days
    format="{time} | {level} | {message}"
)


def run_layer1(max_per_source: int = 100):
    """
    Runs the full Layer 1 data collection pipeline.
    WHAT HAPPENS:
      1. PubMedCollector.collect()        → searches PubMed, parses XML records
      2. EuropePMCCollector.collect()     → searches Europe PMC, parses JSON
      3. SemanticScholarCollector.collect()→ searches S2 for citation data
      4. BioRxivCollector.collect()       → fetches recent preprints
      5. Orchestrator merges and deduplicates all four result sets
      6. Saves the clean merged list to data/processed/collected_YYYYMMDD_HHMMSS.json
      7. Prints a summary report

    OUTPUT FILE FORMAT:
      JSON array of PaperRecord objects. Each looks like:
      {
        "doi": "10.1038/s41586-024-07999-z",
        "pmid": "38765432",
        "title": "Gut microbiome composition...",
        "abstract": "Background: ...",
        "authors": ["Smith J", "Jones K"],
        "journal": "Nature",
        "publication_year": 2024,
        "article_types": ["Journal Article"],
        "mesh_terms": ["Microbiota", "Gastrointestinal Microbiome"],
        "is_open_access": true,
        "content_hash": "a3f8b2c1...",
        "fetched_at": "2024-05-21T02:00:00"
      }
    This file is what Layer 2 (NLP pipeline) will read and process.
    """

    logger.info("Starting Microbiome Literature Miner — Layer 1")
    
    # ── Reset audit files for this run ──-
    from pathlib import Path
    import json

    audit = Path(
        "data/audit")

    audit.mkdir(
        parents=True,
        exist_ok=True)

    for f in ["kept.json", "rejected.json", "review.json", "llm_verified.json"]:

        with open(audit / f,"w") as fp: json.dump([], fp)

    logger.info("[audit] Audit files reset")
    # ─────────────────────────────────────────────────────

    orchestrator = CollectionOrchestrator()
    papers = orchestrator.collect_all(max_per_source=max_per_source)

    logger.success(f"Layer 1 complete. {len(papers)} papers ready for NLP processing.")
    logger.info("Next step: run layer2_nlp.py to process these papers.")

    # Preview the first paper so you can see the data format
    if papers:
        first = papers[0]
        logger.info(f"\nSample record:")
        logger.info(f"  Title:   {first.title[:80]}...")
        logger.info(f"  Journal: {first.journal}")
        logger.info(f"  Year:    {first.publication_year}")
        logger.info(f"  DOI:     {first.doi}")
        logger.info(f"  Authors: {', '.join(first.authors[:3])}{'...' if len(first.authors) > 3 else ''}")
        logger.info(f"  Open access: {first.is_open_access}")

    return papers

def run_layer2(use_ner_model: bool = False, use_llm: bool = False):
    """
    Runs the NLP pipeline on the latest Layer 1 collected data.

    WHAT HAPPENS:
      1. Loads latest collected_YYYYMMDD.json from data/processed/
      2. Runs all 5 NLP modules on each PaperRecord
      3. Saves enriched_YYYYMMDD.json to data/processed/
      4. Returns list of EnrichedPaperRecords for Layer 3

    set USE_NER_MODEL=true for BioBERT NER (needs: pip install transformers torch)
    set USE_LLM=true for Ollama LLM extraction (needs Ollama running with a model)
    """
    from collectors.orchestrator import CollectionOrchestrator
    from nlp.pipeline import NLPPipeline

    logger.info("Starting Layer 2 — NLP Processing Pipeline")
    orchestrator = CollectionOrchestrator()
    papers = orchestrator.load_latest()
    logger.info(f"Loaded {len(papers)} papers from Layer 1")

    pipeline = NLPPipeline(use_ner_model=use_ner_model, use_llm=use_llm)
    enriched = pipeline.process_all(papers)
    logger.success(f"Layer 2 complete. {len(enriched)} enriched records ready for Layer 3.")
    return enriched


def run_layer3(
    enable_enhanced_pipeline: bool = True,
    load_to_neo4j: bool = True,
    batch_size: int = 100,
    num_workers: int = 8
):
    """
    Runs the enhanced knowledge graph pipeline (Layer 3).
    
    WHAT HAPPENS:
      1. Loads latest enriched papers from Layer 2
      2. Extracts semantic relationships with provenance tracking
      3. Reifies claims by aggregating evidence across papers
      4. Loads edges and claims into Neo4j (separate database)
      5. Saves intermediate results to JSON files
    
    This pipeline runs in parallel with the existing system, writing to
    a separate Neo4j database instance (neo4j_enhanced) for safe migration.
    
    Configuration:
      - enable_enhanced_pipeline: Enable/disable the enhanced pipeline
      - load_to_neo4j: Whether to load results into Neo4j
      - batch_size: Papers per batch (default: 100)
      - num_workers: Parallel workers (default: 8, recommended: 8-16)
    
    Requirements: 16.1 (parallel execution), 17.1 (component wiring)
    """
    from nlp.pipeline import NLPPipeline
    from graph.enhanced_kg_pipeline import EnhancedKGPipeline, PipelineConfig
    
    logger.info("Starting Layer 3 — Enhanced Knowledge Graph Pipeline")
    
    # Load enriched papers from Layer 2
    nlp_pipeline = NLPPipeline()
    enriched_papers = nlp_pipeline.load_latest()
    logger.info(f"Loaded {len(enriched_papers)} enriched papers from Layer 2")
    
    # Create pipeline configuration from environment variables
    config = PipelineConfig.from_env()
    config.enabled = enable_enhanced_pipeline
    config.batch_size = batch_size
    config.num_workers = num_workers
    config.save_intermediate = True
    
    # Initialize and run the enhanced pipeline
    # This wires together: SemanticRelationshipExtractor → ProvenanceEncoder → 
    # RelationshipReifier → EnhancedNeo4jLoader
    pipeline = EnhancedKGPipeline(config)
    
    try:
        results = pipeline.run(enriched_papers, load_to_neo4j=load_to_neo4j)
        
        if results["status"] == "success":
            logger.success(
                f"Layer 3 complete. "
                f"Extracted {results['edges_count']} relationships, "
                f"created {results['claims_count']} reified claims. "
                f"Processing time: {results['processing_time_seconds']:.2f}s"
            )
            
            # Print statistics
            stats = results["statistics"]
            logger.info("\nPipeline Statistics:")
            logger.info(f"  Total relationships: {stats.get('total_relationships', 0)}")
            logger.info(f"  Associations: {stats.get('associations', 0)}")
            logger.info(f"  Interventions: {stats.get('interventions', 0)}")
            logger.info(f"  Methodologies: {stats.get('methodologies', 0)}")
            logger.info(f"  Reified claims: {results['claims_count']}")
            logger.info(f"  Processing time: {results['processing_time_seconds']:.2f}s")
            
            if load_to_neo4j:
                logger.info("\nResults loaded into Neo4j database: neo4j_enhanced")
                logger.info("Next step: Run research queries to validate the knowledge graph")
        else:
            logger.warning(f"Pipeline status: {results['status']}")
        
        return results
        
    finally:
        pipeline.close()


def train_relevance_model():
    """
    Trains the Stage 3 ML classifier from collected papers.
    Run this after your first large collection (MAX_PER_SOURCE=500+).

    WHAT HAPPENS:
      1. Loads latest collected JSON from data/processed/
      2. Runs Stage 2 rules on all papers to generate pseudo-labels
         (high-confidence keeps → label 1, high-confidence rejects → label 0)
      3. Encodes papers with sentence-transformers (all-MiniLM-L6-v2)
      4. Trains LogisticRegression with 5-fold cross-validation
      5. Saves model to config/relevance_model.pkl
      6. Reports F1 score

    Re-run monthly as more papers accumulate.
    """
    from collectors.orchestrator import CollectionOrchestrator
    from collectors.relevance_filter import RelevanceFilter

    logger.info("Training ML relevance classifier...")
    orchestrator = CollectionOrchestrator()
    papers = orchestrator.load_latest()
    logger.info(f"Loaded {len(papers)} papers for training")

    rf = RelevanceFilter()
    rf.train_ml_model(papers)


if __name__ == "__main__":
    import os
    mode = os.getenv("RUN_LAYER", "1")
    if mode == "1":
        MAX = int(os.getenv("MAX_PER_SOURCE", "50"))
        run_layer1(max_per_source=MAX)
    elif mode == "2":
        USE_MODEL = os.getenv("USE_NER_MODEL", "false").lower() == "true"
        USE_LLM   = os.getenv("USE_LLM", "false").lower() == "true"
        run_layer2(use_ner_model=USE_MODEL, use_llm=USE_LLM)
    elif mode == "3":
        # Layer 3: Enhanced Knowledge Graph Pipeline
        ENABLE_ENHANCED = os.getenv("ENHANCED_PIPELINE_ENABLED", "true").lower() == "true"
        LOAD_TO_NEO4J = os.getenv("LOAD_TO_NEO4J", "true").lower() == "true"
        BATCH_SIZE = int(os.getenv("ENHANCED_BATCH_SIZE", "100"))
        NUM_WORKERS = int(os.getenv("ENHANCED_NUM_WORKERS", "8"))
        run_layer3(
            enable_enhanced_pipeline=ENABLE_ENHANCED,
            load_to_neo4j=LOAD_TO_NEO4J,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
    elif mode == "train_filter":
        train_relevance_model()
