"""
collectors/pmc_enricher.py
----------------------------
Enriches existing PaperRecord objects by fetching full-text XML from
PubMed Central (PMC) for any paper that has a PMCID.

WHY ENRICHMENT, NOT A COLLECTOR?
  This is NOT a primary collector — it doesn't find new papers.
  It upgrades papers already collected by adding structured full text:
    - Methods section
    - Results section
    - Discussion section
    - Data Availability statement
    - Funding information
    - Conflict of interest statements

  Running it after all collectors ensures we extract maximum value from
  papers we've already decided are relevant.

WHY PMC FULL TEXT MATTERS FOR LAYER 2:
  Abstract-only NLP misses:
    - Specific bacterial strains mentioned only in Methods
    - Statistical results and effect sizes (only in Results)
    - Dataset accession numbers (only in Data Availability)
    - Sequencing methods and protocols (only in Methods)
  Full text gives Layer 2 NLP ~10x more extractable content per paper.

API:
  PMC OAI-PMH endpoint (no key required, NCBI polite pool):
  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
    ?db=pmc&id=PMC{id}&rettype=xml&retmode=xml

  OR the PMC OAI endpoint:
  https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi
    ?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{id}&metadataPrefix=pmc

  We use efetch — simpler and returns structured article XML.

RATE LIMIT:
  Uses NCBI_API_KEY if available (10 req/sec), otherwise 3 req/sec.
  We respect rate limits via the standard _wait_for_rate_limit() pattern.
"""

import time
import xml.etree.ElementTree as ET
from typing import List, Optional
from loguru import logger
from tqdm import tqdm

from config import NCBI_API_KEY, NCBI_EMAIL, RATE_LIMITS
from models import PaperRecord

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PMCEnricher:
    """
    Fetches and attaches full-text content from PMC for papers with a PMCID.

    Usage:
        enricher = PMCEnricher()
        enriched_papers = enricher.enrich(papers)
    """

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"MicrobiomeMiner/1.0 (Academic research; mailto:{NCBI_EMAIL})"
        })
        self._last_request_time = 0.0

        # Rate limit: 10 req/sec with API key, 3 req/sec without
        self._rate_limit_seconds = 0.1 if NCBI_API_KEY else 0.35

        self._base_params = {
            "email": NCBI_EMAIL,
            "tool":  "MicrobiomeMiner",
        }
        if NCBI_API_KEY:
            self._base_params["api_key"] = NCBI_API_KEY

        key_status = "with API key" if NCBI_API_KEY else "without API key (3 req/sec)"
        logger.info(f"[pmc_enricher] Initialized {key_status}")

    def enrich(
        self,
        papers: List[PaperRecord],
        max_enrichments: int = 50,
    ) -> List[PaperRecord]:
        """
        For each paper with a PMCID and no existing full_text, fetch and
        attach the full text from PMC.

        Args:
            papers:           List of PaperRecord objects to enrich in-place
            max_enrichments:  Cap on how many papers to enrich per run
                              (avoids slow runs when many PMCIDs exist)

        Returns:
            The same list with full_text populated where available.
        """
        # Find papers that have a PMCID but no full text yet
        candidates = [
            p for p in papers
            if p.pmcid and not p.full_text
        ]

        if not candidates:
            logger.info("[pmc_enricher] No papers with PMCID found — skipping enrichment")
            return papers

        # Cap to avoid slow runs
        to_enrich = candidates[:max_enrichments]
        skipped   = len(candidates) - len(to_enrich)

        logger.info(
            f"[pmc_enricher] Enriching {len(to_enrich)} papers with PMC full text "
            f"({skipped} skipped due to cap of {max_enrichments})"
        )

        enriched_count = 0
        failed_count   = 0

        for paper in tqdm(to_enrich, desc="PMC full-text enrichment"):
            full_text = self._fetch_full_text(paper.pmcid)
            if full_text:
                paper.full_text = full_text
                # Also set full_text_url if not already set
                if not paper.full_text_url:
                    pmcid_num = paper.pmcid.replace("PMC", "")
                    paper.full_text_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{paper.pmcid}/"
                enriched_count += 1
            else:
                failed_count += 1

        logger.info(
            f"[pmc_enricher] Enrichment complete: "
            f"{enriched_count} papers got full text, "
            f"{failed_count} failed/not available"
        )

        return papers

    def _fetch_full_text(self, pmcid: str) -> Optional[str]:
        """
        Fetches the full-text XML for a PMC article and extracts plain text.

        PMC XML structure (JATS format):
          <article>
            <front>
              <article-meta>
                <abstract>...</abstract>
              </article-meta>
            </front>
            <body>
              <sec sec-type="intro"><title>Introduction</title><p>...</p></sec>
              <sec sec-type="methods"><title>Methods</title><p>...</p></sec>
              <sec sec-type="results"><title>Results</title><p>...</p></sec>
              <sec sec-type="discussion"><title>Discussion</title><p>...</p></sec>
            </body>
            <back>
              <sec sec-type="data-availability">...</sec>
              <ack>Acknowledgements</ack>
              <fn-group>
                <fn fn-type="financial-disclosure">Funding</fn>
              </fn-group>
            </back>
          </article>

        We extract body + back sections, preserving section structure.
        """
        # Normalize PMCID — strip "PMC" prefix for the API call
        pmcid_num = pmcid.replace("PMC", "").strip()

        self._wait_for_rate_limit()

        params = {
            **self._base_params,
            "db":      "pmc",
            "id":      pmcid_num,
            "rettype": "xml",
            "retmode": "xml",
        }

        try:
            response = self.session.get(
                f"{EUTILS_BASE}/efetch.fcgi",
                params=params,
                timeout=30,
            )

            if response.status_code == 429:
                logger.warning(f"[pmc_enricher] Rate limited on {pmcid}, waiting 60s")
                time.sleep(60)
                return None

            if response.status_code == 404:
                logger.debug(f"[pmc_enricher] {pmcid} not found in PMC (may be restricted)")
                return None

            response.raise_for_status()

            return self._parse_full_text_xml(response.content, pmcid)

        except Exception as e:
            logger.debug(f"[pmc_enricher] Failed to fetch {pmcid}: {e}")
            return None

    def _parse_full_text_xml(self, xml_content: bytes, pmcid: str) -> Optional[str]:
        """
        Extracts plain text from PMC JATS XML, preserving section structure.

        Sections extracted (in order):
          1. Body sections (intro, methods, results, discussion, etc.)
          2. Data availability statement
          3. Acknowledgements
          4. Funding information

        Each section is prefixed with its title so NLP can identify sections.
        """
        try:
            root = ET.fromstring(xml_content)

            # PMC XML can be wrapped in <pmc-articleset> or start directly at <article>
            article = root.find(".//article")
            if article is None:
                article = root  # root IS the article

            sections = []

            # ── Body sections ─────────────────────────────────────────────────
            body = article.find(".//body")
            if body is not None:
                for sec in body.iter("sec"):
                    title_elem = sec.find("title")
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Section"

                    # Collect all paragraph text in this section
                    paragraphs = []
                    for p in sec.findall("p"):
                        text = "".join(p.itertext()).strip()
                        if text:
                            paragraphs.append(text)

                    if paragraphs:
                        sections.append(f"## {title}\n" + "\n".join(paragraphs))

            # ── Back matter (data availability, funding, ack) ─────────────────
            back = article.find(".//back")
            if back is not None:
                # Data availability
                for sec in back.iter("sec"):
                    sec_type = sec.get("sec-type", "").lower()
                    if "data" in sec_type or "availability" in sec_type:
                        text = "".join(sec.itertext()).strip()
                        if text:
                            sections.append(f"## Data Availability\n{text}")

                # Acknowledgements
                ack = back.find(".//ack")
                if ack is not None:
                    text = "".join(ack.itertext()).strip()
                    if text:
                        sections.append(f"## Acknowledgements\n{text}")

                # Funding / financial disclosure
                for fn in back.iter("fn"):
                    fn_type = fn.get("fn-type", "").lower()
                    if "financial" in fn_type or "fund" in fn_type:
                        text = "".join(fn.itertext()).strip()
                        if text:
                            sections.append(f"## Funding\n{text}")

            if not sections:
                logger.debug(f"[pmc_enricher] {pmcid}: XML parsed but no body text found")
                return None

            full_text = "\n\n".join(sections)
            logger.debug(
                f"[pmc_enricher] {pmcid}: extracted {len(full_text)} chars "
                f"across {len(sections)} sections"
            )
            return full_text

        except ET.ParseError as e:
            logger.debug(f"[pmc_enricher] {pmcid}: XML parse error — {e}")
            return None
        except Exception as e:
            logger.debug(f"[pmc_enricher] {pmcid}: unexpected error — {e}")
            return None

    def _wait_for_rate_limit(self):
        """Enforce minimum gap between requests."""
        elapsed = time.time() - self._last_request_time
        wait    = self._rate_limit_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()
