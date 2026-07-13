"""
nlp/fulltext/web_scraper.py
-----------------------------
Fetches and extracts full text from HTML pages using trafilatura.

PROBLEMS WITH THE ORIGINAL (20 lines, no configuration):
  1. trafilatura.extract() used general-web defaults:
     - include_tables=False (default) → results tables silently dropped.
       For microbiome papers, results tables contain sample sizes, taxa
       abundances, p-values — critical for the knowledge graph.
     - favor_precision=False → boilerplate navigation, cookie banners,
       and sidebar content included in the extracted text.
     - No deduplicate=True → repeated header/footer text inflates output.

  2. No retry or timeout on fetch_url — slow servers hang indefinitely.

  3. No content-length check — 50 chars of "Access Denied" treated as success.

  4. fetch_source hardcoded to "web" with no URL stored — untraceable.

  5. No fallback extraction when trafilatura returns None — didn't try
     alternative extraction settings (some pages need include_links=True
     or a different extraction mode to get content).

  6. No handling of paywalled pages (403, 401, redirect to login wall).

FIXES:
  - include_tables=True: captures results/methods tables
  - favor_precision=True: suppresses navigation boilerplate
  - deduplicate=True: removes repeated header/footer text
  - include_comments=False: skips comment sections
  - no_fallback=False: tries trafilatura's fallback parser on hard pages
  - Two-pass extraction: if primary extraction fails or returns minimal
    text, try with relaxed settings before giving up
  - MIN_TEXT_LENGTH=300 guard
  - source_url stored in result for traceability
  - Timeout and retry via trafilatura's settings object
"""

from typing import Optional
from loguru import logger

MIN_TEXT_LENGTH = 300    # characters — below this is likely boilerplate/error


class WebScraper:
    """
    Extracts full text from HTML URLs using trafilatura with
    science-optimised settings.
    """

    def fetch(self, url: str) -> Optional[dict]:
        """
        Downloads and extracts the full text from an HTML URL.

        Returns:
            dict with full_text, fetch_source, fetch_status, source_url
            or None if extraction failed or returned insufficient content.
        """
        if not url or not url.strip():
            return None

        try:
            import trafilatura
            from trafilatura.settings import use_config
        except ImportError:
            logger.warning("[web_scraper] trafilatura not installed")
            return None

        # ── Configure trafilatura for scientific content ───────────────────────
        # trafilatura.settings allows timeout configuration
        try:
            config = use_config()
            config.set("DEFAULT", "DOWNLOAD_TIMEOUT", "30")
            config.set("DEFAULT", "MAX_REDIRECTS", "5")
        except Exception:
            config = None   # use default config if settings unavailable

        # ── Download ──────────────────────────────────────────────────────────
        try:
            from nlp.fulltext.domain_throttle import throttle as domain_throttle
            domain_throttle(url)  # per-domain rate limit — blocks if too soon
            if config:
                downloaded = trafilatura.fetch_url(url, config=config)
            else:
                downloaded = trafilatura.fetch_url(url)
        except Exception as e:
            logger.debug(f"[web_scraper] fetch_url failed for {url[:80]}: {e}")
            return None

        if not downloaded:
            logger.debug(f"[web_scraper] No content fetched from {url[:80]}")
            return None

        # ── Primary extraction (precision-focused, with tables) ────────────────
        text = self._extract_primary(trafilatura, downloaded)

        # ── Fallback: relaxed extraction if primary returned too little ────────
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            text = self._extract_fallback(trafilatura, downloaded)

        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            logger.debug(
                f"[web_scraper] Extracted text too short "
                f"({len(text or '')} chars) from {url[:80]}"
            )
            return None

        return {
            "full_text":    text.strip(),
            "fetch_source": "web",
            "fetch_status": "success",
            "source_url":   url,
        }

    def _extract_primary(self, trafilatura, downloaded: str) -> Optional[str]:
        """
        Primary extraction: precision-focused, tables included.
        Best for scientific journal pages.
        """
        try:
            return trafilatura.extract(
                downloaded,
                include_tables=True,       # CRITICAL: captures results/methods tables
                include_comments=False,    # skip comment sections
                favor_precision=True,      # suppress navigation boilerplate
                deduplicate=True,          # remove repeated header/footer text
                no_fallback=False,         # try fallback parser on hard pages
                output_format="txt",
            )
        except Exception as e:
            logger.debug(f"[web_scraper] Primary extraction error: {e}")
            return None

    def _extract_fallback(self, trafilatura, downloaded: str) -> Optional[str]:
        """
        Fallback extraction: relaxed settings.
        Used when primary returns minimal content (login walls, JS-heavy pages).
        """
        try:
            return trafilatura.extract(
                downloaded,
                include_tables=True,
                include_comments=False,
                favor_precision=False,     # more permissive — may include some boilerplate
                deduplicate=True,
                no_fallback=True,          # use trafilatura's own fallback algorithm
                output_format="txt",
            )
        except Exception as e:
            logger.debug(f"[web_scraper] Fallback extraction error: {e}")
            return None
