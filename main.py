"""
Entry point for the Microbiome Literature Miner.
Run this to kick off Layer 1:
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

def run_layer2(use_ner_model: bool = False):
    """
    Runs the NLP pipeline on the latest Layer 1 collected data.

    WHAT HAPPENS:
      1. Loads latest collected_YYYYMMDD.json from data/processed/
      2. Runs all 5 NLP modules on each PaperRecord
      3. Saves enriched_YYYYMMDD.json to data/processed/
      4. Returns list of EnrichedPaperRecords for Layer 3

    set use_ner_model=True for BioBERT NER (needs: pip install transformers torch)
    """
    from collectors.orchestrator import CollectionOrchestrator
    from nlp.pipeline import NLPPipeline

    logger.info("Starting Layer 2 — NLP Processing Pipeline")
    orchestrator = CollectionOrchestrator()
    papers = orchestrator.load_latest()
    logger.info(f"Loaded {len(papers)} papers from Layer 1")

    pipeline = NLPPipeline(use_ner_model=use_ner_model)
    enriched = pipeline.process_all(papers)
    logger.success(f"Layer 2 complete. {len(enriched)} enriched records ready for Layer 3.")
    return enriched


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
        run_layer2(use_ner_model=USE_MODEL)
    elif mode == "train_filter":
        train_relevance_model()
