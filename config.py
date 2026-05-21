"""
config.py
---------
Single source of truth for all project settings.

WHY A SEPARATE CONFIG?
  Every other file imports from here. If you ever need to change an API key,
  a rate limit, or a date range, you change it in ONE place, not scattered
  across 10 files. This is standard production practice.

HOW TO USE:
  Copy .env.example → .env and fill in your actual keys.
  Then anywhere in the project: from config import settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (won't override real environment variables)
load_dotenv()

# ─── Project Paths ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent          # /microbiome_miner/
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"              # Where raw API responses are cached
PROC_DIR = DATA_DIR / "processed"        # Where NLP-processed records go
LOG_DIR  = BASE_DIR / "logs"

# Create directories if they don't exist yet (safe to call repeatedly)
for d in [RAW_DIR, PROC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Search Parameters ────────────────────────────────────────────────────────
# This is what we're actually searching for across all sources.

SEARCH_QUERY = "human microbiome"        # Core topic

# Date range for the project (2024–2026)
DATE_FROM = "2024/01/01"
DATE_TO   = "2026/12/31"

# MeSH (Medical Subject Headings) terms — PubMed's controlled vocabulary.
# Using MeSH gives you MORE precise results than keyword search alone because
# it catches synonyms (e.g. "gut flora" and "intestinal microbiome" both
# map to the same MeSH term).
PUBMED_MESH_TERMS = [
    "Microbiota",
    "Gastrointestinal Microbiome",
    "RNA, Ribosomal, 16S",
    "Metagenomics",
    "Bacteria",
]

# Maximum papers to fetch per source per run (set lower during development)
MAX_RESULTS_PER_SOURCE = 500


# ─── API Credentials ─────────────────────────────────────────────────────────
# Never hardcode secrets in source code. They come from the .env file.

NCBI_EMAIL    = os.getenv("NCBI_EMAIL", "your_email@example.com")
NCBI_API_KEY  = os.getenv("NCBI_API_KEY", "")    # Free at NCBI; gives 10 req/sec vs 3
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Google Scholar — choose one mode:
#   "scholarly" (free, may get blocked) or "serpapi" (paid, reliable)
GOOGLE_SCHOLAR_MODE = os.getenv("GOOGLE_SCHOLAR_MODE", "scholarly")
SERPAPI_KEY         = os.getenv("SERPAPI_KEY", "")        # https://serpapi.com/
SCRAPER_API_KEY     = os.getenv("SCRAPER_API_KEY", "")    # https://www.scraperapi.com/ (proxy for scholarly mode)

# Google Scholar — no official API. Two backend options:
#   Option A (free):  scholarly library with rotating free proxies
#   Option B (paid):  SerpAPI — $50/month, much more reliable
#   If SERPAPI_KEY is set, Option B is used automatically.
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# ScraperAPI proxy service — works well with scholarly (free tier: 5000 req/month)
# Sign up: https://www.scraperapi.com/
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY", "")

# Neo4j connection (Layer 4)
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# PostgreSQL connection (Layer 4)
POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://postgres:password@localhost:5432/microbiome_miner"
)


# ─── Rate Limits ──────────────────────────────────────────────────────────────
# Academic APIs are strict about rate limits. Violating them gets your IP
# blocked. These values are conservative — well within each API's limits.
#
# Rule of thumb:
#   Without API key → 3 requests/second for NCBI
#   With NCBI API key → 10 requests/second
#   Semantic Scholar → 100 requests/minute for registered users

RATE_LIMITS = {
    "pubmed":           0.4,   # seconds between requests (= 2.5 req/sec)
    "europepmc":        0.5,   # 2 req/sec
    "semantic_scholar": 1.0,   # 1 req/sec (conservative)
    "biorxiv":          0.5,   # 2 req/sec
    "google_scholar":   8.0,   # 8 sec between requests — Scholar blocks aggressively
                               # With free proxies: rotate every 5 requests
                               # With ScraperAPI: can go faster (their problem to handle)
}

# How many times to retry a failed request before giving up
MAX_RETRIES = 3

# Seconds to wait before first retry (doubles each attempt: 2, 4, 8...)
RETRY_BACKOFF_BASE = 2


# ─── Scheduler Settings (Layer 4b) ────────────────────────────────────────────
# Cron-style schedule for automatic updates

SCHEDULE = {
    "daily_new_papers": {
        "hour": 2, "minute": 0,    # Runs at 2:00 AM every day
        "description": "Fetch papers added in last 24 hours"
    },
    "weekly_full_scan": {
        "day_of_week": "sun", "hour": 4, "minute": 0,
        "description": "Re-scan all sources for updated metadata"
    },
}


# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")   # DEBUG | INFO | WARNING | ERROR
LOG_FILE  = LOG_DIR / "miner.log"
