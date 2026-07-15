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

# ── Embedding Store Configuration ─────────────────────────────────────────────
EMBEDDING_STORE_DIR = Path(os.getenv("EMBEDDING_STORE_DIR", str(DATA_DIR / "embeddings")))

# Create directories if they don't exist yet (safe to call repeatedly)
for d in [RAW_DIR, PROC_DIR, LOG_DIR, EMBEDDING_STORE_DIR]:
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
#
# Structure: three tiers of specificity
#   Tier 1 — Core microbiome concepts (highest precision, highest recall for this project)
#   Tier 2 — Sequencing & analysis methods (catches papers with data the metagenomics gate wants)
#   Tier 3 — Disease + host contexts (human specificity without duplicating the Humans[MeSH] filter)
#
# Removed: "Bacteria" — far too broad; catches veterinary, food-science, and environmental
#   bacteriology with no human microbiome relevance. Precision penalty outweighs recall gain.
PUBMED_MESH_TERMS = [
    # ── Tier 1: Core microbiome concepts ──────────────────────────────────
    "Microbiota",                           # umbrella: all microbial communities
    "Gastrointestinal Microbiome",          # gut-specific (catches "gut flora" etc.)
    "Dysbiosis",                            # compositional imbalance — core disease link
    "Fecal Microbiota Transplantation",     # major therapeutic modality
    "Probiotics",                           # intervention-focused papers
    "Prebiotics",                           # dietary intervention papers
    "Bacteria",                             # broad but needed for coverage; Stage 1 filter trims noise

    # ── Tier 2: Sequencing & bioinformatics methods ───────────────────────
    "RNA, Ribosomal, 16S",                  # 16S amplicon — most common sequencing target
    "Metagenomics",                         # shotgun metagenomic studies
    "Metatranscriptomics",                  # RNA-seq of microbial communities
    "Sequence Analysis, DNA",               # general sequencing analysis
    "High-Throughput Nucleotide Sequencing",# next-gen sequencing (Illumina, Nanopore, etc.)
    "Phylogeny",                            # taxonomic classification studies

    # ── Tier 3: Disease & host-context terms ──────────────────────────────
    "Inflammatory Bowel Diseases",          # IBD — Crohn's + UC
    "Irritable Bowel Syndrome",             # IBS — high microbiome literature volume
    "Colorectal Neoplasms",                 # colorectal cancer microbiome link
    "Obesity",                              # metabolic disease — strong microbiome signal
    "Diabetes Mellitus, Type 2",            # T2D — gut-brain-metabolic axis
    "Liver Diseases",                       # NAFLD / NASH — gut-liver axis papers

    # ── Tier 4: Anatomical & metabolite context ───────────────────────────
    "Colon",                                # anatomical site — colonic microbiome
    "Intestines",                           # broader GI tract coverage
    "Fatty Acids, Volatile",                # SCFAs (butyrate, propionate, acetate)
]

# Maximum papers to fetch per source per run.
# Override via MAX_PER_SOURCE in .env (e.g. 100 for dev, 500 for production).
MAX_RESULTS_PER_SOURCE = int(os.getenv("MAX_PER_SOURCE", "500"))


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


# ── Embedding Model Configuration ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "allenai/specter2")
EMBEDDING_FALLBACK_MODEL = os.getenv("EMBEDDING_FALLBACK_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ── Stage 3.5 Thresholds ──────────────────────────────────────────────────────
EMBEDDING_POS_KEEP_THRESHOLD = float(os.getenv("EMBEDDING_POS_KEEP_THRESHOLD", "0.85"))
EMBEDDING_NEG_REJECT_THRESHOLD = float(os.getenv("EMBEDDING_NEG_REJECT_THRESHOLD", "0.85"))
EMBEDDING_CROSS_CEILING = float(os.getenv("EMBEDDING_CROSS_CEILING", "0.60"))
EMBEDDING_MIN_PARTITION_SIZE = int(os.getenv("EMBEDDING_MIN_PARTITION_SIZE", "50"))

# ── Semantic Cache ─────────────────────────────────────────────────────────────
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.97"))

# ── Batched Verifier ──────────────────────────────────────────────────────────
BATCH_LLM_SIZE = int(os.getenv("BATCH_LLM_SIZE", "16"))

# ── Hybrid Classifier ─────────────────────────────────────────────────────────
HYBRID_MIN_STORE_SIZE = int(os.getenv("HYBRID_MIN_STORE_SIZE", "2000"))
HYBRID_MIN_TRAIN_SAMPLES = int(os.getenv("HYBRID_MIN_TRAIN_SAMPLES", "200"))
HYBRID_MIN_RETRAIN_NEW = int(os.getenv("HYBRID_MIN_RETRAIN_NEW", "100"))

# ── Disagreement Router ────────────────────────────────────────────────────────
BLENDED_CONFIDENCE_LOW = float(os.getenv("BLENDED_CONFIDENCE_LOW", "0.40"))
BLENDED_CONFIDENCE_HIGH = float(os.getenv("BLENDED_CONFIDENCE_HIGH", "0.70"))

# ── Embedding Store Growth ─────────────────────────────────────────────────────
GROWTH_KEEP_THRESHOLD = float(os.getenv("GROWTH_KEEP_THRESHOLD", "0.80"))
GROWTH_REJECT_THRESHOLD = float(os.getenv("GROWTH_REJECT_THRESHOLD", "0.20"))

# ── Latency Monitoring ─────────────────────────────────────────────────────────
EMBEDDING_LATENCY_WARN_MS = float(os.getenv("EMBEDDING_LATENCY_WARN_MS", "200.0"))

# ── Metagenomics Gate ──────────────────────────────────────────────────────────
METAGENOMICS_GATE_ENABLED = os.getenv("METAGENOMICS_GATE_ENABLED", "true").lower() == "true"

# ── Full-Text Exhausted Cache TTL ──────────────────────────────────────────────
# Papers where all full-text fetch strategies failed are cached as "exhausted"
# to avoid re-trying on every run. But OA embargoes lapse (commonly 6-12 months
# post-publication) and Unpaywall's index grows over time, so permanently
# skipping them leaves potential full text on the table. After this many days,
# an exhausted paper becomes eligible for one retry.
FULLTEXT_EXHAUSTED_TTL_DAYS = int(os.getenv("FULLTEXT_EXHAUSTED_TTL_DAYS", "90"))

# ── Drift Monitor ─────────────────────────────────────────────────────────────
DRIFT_SAMPLE_RATE = float(os.getenv("DRIFT_SAMPLE_RATE", "0.01"))
DRIFT_MIN_SAMPLE = int(os.getenv("DRIFT_MIN_SAMPLE", "10"))

# ── Embedding Store Backend ────────────────────────────────────────────────────
EMBEDDING_STORE_BACKEND = os.getenv("EMBEDDING_STORE_BACKEND", "numpy")  # "numpy" | "faiss"


# ─── Ollama / LLM Backend Configuration ──────────────────────────────────────

class ConfigurationError(Exception):
    """Raised at import time when environment variable configuration is invalid."""
    pass


@dataclass(frozen=True)
class BackendConfig:
    """Typed, immutable configuration for the Ollama LLM backend."""
    llm_backend: str                  # always "ollama"
    ollama_base_url: str              # e.g. "http://localhost:11434"
    ollama_extraction_model: str      # e.g. "llama3"
    ollama_grounding_model: str       # e.g. "llama3"
    ollama_timeout_seconds: int       # ≥ 1
    ollama_max_retries: int           # ≥ 0
    ollama_retry_backoff_base: float  # ≥ 1.0


def _load_backend_config() -> BackendConfig:
    """
    Reads env vars, validates types and accepted values, raises ConfigurationError
    on any violation. Called once at module import; result stored as BACKEND_CONFIG.
    """
    # ── LLM_BACKEND ──────────────────────────────────────────────────────────
    llm_backend = os.getenv("LLM_BACKEND", "ollama")
    if llm_backend != "ollama":
        raise ConfigurationError(
            f"LLM_BACKEND={llm_backend!r} is not valid. "
            f"Accepted value: 'ollama'"
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

    return BackendConfig(
        llm_backend=llm_backend,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_extraction_model=os.getenv("OLLAMA_EXTRACTION_MODEL", "llama3"),
        ollama_grounding_model=os.getenv("OLLAMA_GROUNDING_MODEL", "llama3"),
        ollama_timeout_seconds=ollama_timeout_seconds,
        ollama_max_retries=ollama_max_retries,
        ollama_retry_backoff_base=ollama_retry_backoff_base,
    )


BACKEND_CONFIG: BackendConfig = _load_backend_config()
