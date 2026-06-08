"""
config.py
WHY A SEPARATE CONFIG?
Every other file imports from here. If you ever need to change an API key,
a rate limit, or a date range, you change it in ONE place, not scattered
across 10 files. This is standard production practice.

"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

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
NCBI_API_KEY  = os.getenv("NCBI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


# ─── Enhanced Knowledge Graph Configuration ──────────────────────────────────
# The enhanced knowledge graph system uses semantic relationships with provenance
# tracking, evidence aggregation, and entity normalization.
#
# MIGRATION NOTE: The old flat relationship system has been decommissioned.
# All new data should use the enhanced pipeline.

# Neo4j Enhanced Knowledge Graph (primary system)
NEO4J_ENHANCED_URI      = os.getenv("NEO4J_ENHANCED_URI", "bolt://localhost:7687")
NEO4J_ENHANCED_USER     = os.getenv("NEO4J_ENHANCED_USER", "neo4j")
NEO4J_ENHANCED_PASSWORD = os.getenv("NEO4J_ENHANCED_PASSWORD", "password")
NEO4J_ENHANCED_DATABASE = os.getenv("NEO4J_ENHANCED_DATABASE", "neo4j_enhanced")

# Enhanced Pipeline Settings
ENHANCED_PIPELINE_ENABLED = os.getenv("ENHANCED_PIPELINE_ENABLED", "true").lower() == "true"
ENHANCED_BATCH_SIZE = int(os.getenv("ENHANCED_BATCH_SIZE", "100"))
ENHANCED_NUM_WORKERS = int(os.getenv("ENHANCED_NUM_WORKERS", "8"))

# Query Engine Settings
QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
QUERY_CACHE_TTL_HOURS = int(os.getenv("QUERY_CACHE_TTL_HOURS", "24"))
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))

# Entity Normalization Settings
ENTITY_NORMALIZATION_ENABLED = os.getenv("ENTITY_NORMALIZATION_ENABLED", "true").lower() == "true"
ENTITY_FUZZY_MATCH_THRESHOLD = int(os.getenv("ENTITY_FUZZY_MATCH_THRESHOLD", "2"))  # Edit distance

# Provenance Tracking Settings
PROVENANCE_CONTEXT_SENTENCES = int(os.getenv("PROVENANCE_CONTEXT_SENTENCES", "2"))  # ±N sentences
PROVENANCE_VALIDATION_STRICT = os.getenv("PROVENANCE_VALIDATION_STRICT", "true").lower() == "true"

# Evidence Aggregation Settings
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.5"))
REIFICATION_ENABLED = os.getenv("REIFICATION_ENABLED", "true").lower() == "true"

# Legacy System (DEPRECATED - kept for rollback only)
# DO NOT USE FOR NEW DATA
NEO4J_LEGACY_URI      = os.getenv("NEO4J_LEGACY_URI", "bolt://localhost:7688")
NEO4J_LEGACY_USER     = os.getenv("NEO4J_LEGACY_USER", "neo4j")
NEO4J_LEGACY_PASSWORD = os.getenv("NEO4J_LEGACY_PASSWORD", "password")
NEO4J_LEGACY_DATABASE = os.getenv("NEO4J_LEGACY_DATABASE", "neo4j_legacy")

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
    "openalex":         0.1,   # 10 req/sec (polite pool with email)
    "crossref":         0.02,  # 50 req/sec (polite pool with User-Agent)
    "core":             0.6,   # 100 req/min with API key = ~1 req/sec
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


# ─── Ollama / LLM Backend Configuration ──────────────────────────────────────

class ConfigurationError(Exception):
    """Raised at import time when environment variable configuration is invalid."""
    pass


@dataclass(frozen=True)
class BackendConfig:
    """Typed, immutable configuration for the LLM backend."""
    llm_backend: str                  # "ollama" | "gemini"
    ollama_base_url: str              # e.g. "http://localhost:11434"
    ollama_extraction_model: str      # e.g. "llama3"
    ollama_grounding_model: str       # e.g. "llama3"
    ollama_timeout_seconds: int       # ≥ 1
    ollama_max_retries: int           # ≥ 0
    ollama_retry_backoff_base: float  # ≥ 1.0
    ollama_fallback_to_gemini: bool
    gemini_extraction_model: str      # e.g. "gemini-2.0-flash"
    gemini_grounding_model: str       # e.g. "gemini-2.5-flash"


def _load_backend_config() -> BackendConfig:
    """
    Reads env vars, validates types and accepted values, raises ConfigurationError
    on any violation. Called once at module import; result stored as BACKEND_CONFIG.
    """
    # ── LLM_BACKEND ──────────────────────────────────────────────────────────
    llm_backend = os.getenv("LLM_BACKEND", "ollama")
    accepted_backends = {"ollama", "gemini"}
    if llm_backend not in accepted_backends:
        raise ConfigurationError(
            f"LLM_BACKEND={llm_backend!r} is not valid. "
            f"Accepted values: {sorted(accepted_backends)}"
        )

    # ── OLLAMA_TIMEOUT_SECONDS ────────────────────────────────────────────────
    _timeout_raw = os.getenv("OLLAMA_TIMEOUT_SECONDS", "30")
    try:
        ollama_timeout_seconds = int(_timeout_raw)
        if ollama_timeout_seconds < 1:
            raise ValueError("must be ≥ 1")
    except (ValueError, TypeError):
        raise ConfigurationError(
            f"OLLAMA_TIMEOUT_SECONDS={_timeout_raw!r} is not a valid integer ≥ 1"
        )

    # ── OLLAMA_MAX_RETRIES ────────────────────────────────────────────────────
    _retries_raw = os.getenv("OLLAMA_MAX_RETRIES", "3")
    try:
        ollama_max_retries = int(_retries_raw)
        if ollama_max_retries < 0:
            raise ValueError("must be ≥ 0")
    except (ValueError, TypeError):
        raise ConfigurationError(
            f"OLLAMA_MAX_RETRIES={_retries_raw!r} is not a valid integer ≥ 0"
        )

    # ── OLLAMA_RETRY_BACKOFF_BASE ─────────────────────────────────────────────
    _backoff_raw = os.getenv("OLLAMA_RETRY_BACKOFF_BASE", "2.0")
    try:
        import math as _math
        ollama_retry_backoff_base = float(_backoff_raw)
        if not _math.isfinite(ollama_retry_backoff_base) or ollama_retry_backoff_base < 1.0:
            raise ValueError("must be a finite float ≥ 1.0")
    except (ValueError, TypeError):
        raise ConfigurationError(
            f"OLLAMA_RETRY_BACKOFF_BASE={_backoff_raw!r} is not a valid float ≥ 1.0"
        )

    # ── OLLAMA_FALLBACK_TO_GEMINI ─────────────────────────────────────────────
    _fallback_raw = os.getenv("OLLAMA_FALLBACK_TO_GEMINI", "false")
    ollama_fallback_to_gemini = _fallback_raw.lower() == "true"

    # ── GEMINI_API_KEY presence checks ────────────────────────────────────────
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    if llm_backend == "gemini" and not gemini_api_key:
        raise ConfigurationError(
            "LLM_BACKEND is set to 'gemini' but GEMINI_API_KEY is not set in the environment"
        )

    if ollama_fallback_to_gemini and not gemini_api_key:
        raise ConfigurationError(
            "OLLAMA_FALLBACK_TO_GEMINI is 'true' but GEMINI_API_KEY is not set in the environment"
        )

    return BackendConfig(
        llm_backend=llm_backend,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_extraction_model=os.getenv("OLLAMA_EXTRACTION_MODEL", "llama3"),
        ollama_grounding_model=os.getenv("OLLAMA_GROUNDING_MODEL", "llama3"),
        ollama_timeout_seconds=ollama_timeout_seconds,
        ollama_max_retries=ollama_max_retries,
        ollama_retry_backoff_base=ollama_retry_backoff_base,
        ollama_fallback_to_gemini=ollama_fallback_to_gemini,
        gemini_extraction_model=os.getenv("GEMINI_EXTRACTION_MODEL", "gemini-2.0-flash"),
        gemini_grounding_model=os.getenv("GEMINI_GROUNDING_MODEL", "gemini-2.5-flash"),
    )


BACKEND_CONFIG: BackendConfig = _load_backend_config()
