"""
nlp/fulltext/domain_throttle.py
---------------------------------
Per-domain rate limiter for full-text fetching.

WHY THIS EXISTS:
  Phase 1 runs 64 parallel threads fetching full text. When many papers
  share the same publisher (e.g., 50 PLOS papers, 30 Elsevier papers),
  those threads all hit the same domain simultaneously. Publishers detect
  this burst and respond with HTTP 429 (Too Many Requests), causing the
  paper's fetch to fail permanently (marked "exhausted" in cache).

  This module enforces a minimum delay between consecutive requests to
  the SAME domain, regardless of which thread or fetcher is making the
  request. Different domains are throttled independently — a request to
  journals.plos.org doesn't block a concurrent request to europepmc.org.

DESIGN:
  - Thread-safe (Phase 1 uses ThreadPoolExecutor, not ProcessPoolExecutor)
  - Per-domain tracking via a dict of {domain: last_request_timestamp}
  - Configurable default delay (1.5s) with per-domain overrides for APIs
    that have published rate limits
  - Blocking wait: if a thread needs to request a domain that was hit
    too recently, it sleeps the remaining time rather than failing

USAGE:
  from nlp.fulltext.domain_throttle import throttle
  throttle(url)  # blocks if needed, then returns immediately
  response = requests.get(url, ...)
"""

import os
import time
import threading
from urllib.parse import urlparse
from loguru import logger

# Default minimum seconds between requests to the same domain.
# 1.5s is conservative enough for most publishers while still allowing
# decent throughput across many different domains in parallel.
_DEFAULT_DELAY = float(os.getenv("FULLTEXT_DOMAIN_DELAY", "1.5"))

# Per-domain overrides for APIs with known/published rate limits.
# These are the actual full-text content servers, not metadata APIs
# (those have their own rate limiting in their respective collectors).
_DOMAIN_DELAYS: dict[str, float] = {
    # EuropePMC is generous — used for XML full text, rarely rate-limits
    "www.ebi.ac.uk": 0.5,
    "europepmc.org": 0.5,
    # NCBI APIs (with API key = 10 req/sec, without = 3 req/sec)
    "eutils.ncbi.nlm.nih.gov": 0.12,
    "pmc.ncbi.nlm.nih.gov": 0.35,
    # Unpaywall metadata lookup (not content download)
    "api.unpaywall.org": 0.5,
    # OpenAIRE search
    "api.openaire.eu": 0.5,
    # Known aggressive rate-limiters (1 req/2s to be safe)
    "www.nature.com": 2.0,
    "link.springer.com": 2.0,
    "www.sciencedirect.com": 2.0,
    "onlinelibrary.wiley.com": 2.0,
    "academic.oup.com": 2.0,
    "www.tandfonline.com": 2.0,
    "www.cell.com": 2.0,
    # Open-access publishers (more generous)
    "journals.plos.org": 1.0,
    "www.frontiersin.org": 1.0,
    "www.mdpi.com": 1.0,
    "bmcmicrobiol.biomedcentral.com": 1.0,
}

_lock = threading.Lock()
_last_request: dict[str, float] = {}  # domain → timestamp of last request


def _get_domain(url: str) -> str:
    """Extracts the domain from a URL for throttling purposes."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"


def _get_delay(domain: str) -> float:
    """Returns the minimum delay for a domain (exact match or default)."""
    if domain in _DOMAIN_DELAYS:
        return _DOMAIN_DELAYS[domain]
    # Check if it's a subdomain of a known domain
    for known, delay in _DOMAIN_DELAYS.items():
        if domain.endswith("." + known) or domain == known:
            return delay
    return _DEFAULT_DELAY


def throttle(url: str) -> None:
    """
    Blocks the calling thread until it's safe to make a request to this
    URL's domain without violating the per-domain rate limit.

    Call this BEFORE making any HTTP request in a fetcher. It's thread-safe
    and handles concurrent calls from 64+ threads correctly — only one
    thread per domain proceeds at a time, others sleep their remaining wait.

    Does NOT raise exceptions — always returns (possibly after sleeping).
    """
    domain = _get_domain(url)
    delay = _get_delay(domain)

    with _lock:
        now = time.time()
        last = _last_request.get(domain, 0.0)
        elapsed = now - last
        wait_needed = delay - elapsed

        if wait_needed > 0:
            # Update timestamp NOW (before sleeping) so other threads
            # checking the same domain see this slot as "taken" and
            # calculate their own wait from this point, not from the
            # old timestamp. This prevents thundering-herd where multiple
            # threads all see the same "last" and all decide to sleep
            # the same short duration, then fire simultaneously.
            _last_request[domain] = now + wait_needed
        else:
            _last_request[domain] = now

    # Sleep OUTSIDE the lock so other domains aren't blocked
    if wait_needed > 0:
        time.sleep(wait_needed)
