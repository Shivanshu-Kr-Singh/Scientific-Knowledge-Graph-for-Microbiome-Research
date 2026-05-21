"""
main.py
--------
Entry point for the Microbiome Literature Miner.

Run this to kick off Layer 1:
  python main.py

During development, set MAX_RESULTS_PER_SOURCE small (e.g. 20) so you're
not waiting 10 minutes for a full run while testing.
"""

import sys
from loguru import logger
from config import LOG_FILE, LOG_LEVEL
from collectors.orchestrator import CollectionOrchestrator


# ─── Configure Logging ────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    # Use a small number during development to test quickly
    # Change to 500+ for production runs
    import os
    MAX = int(os.getenv("MAX_PER_SOURCE", "50"))
    run_layer1(max_per_source=MAX)
