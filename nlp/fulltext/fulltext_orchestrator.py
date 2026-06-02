"""
nlp/fulltext/fulltext_orchestrator.py
---------------------------------------
Orchestrates all full-text fetching strategies in priority order.

Strategy priority:
  1. EuropePMC XML        (fast REST API — EuropePMC OA corpus)
  2. NCBI PMC full text   (Entrez efetch — larger PMC OA subset, catches what EuropePMC misses)
  3. PDF via pdf_url      (direct PDF download + parser)
  4. HTML scraping        (trafilatura scrape of full_text_url)
  5. Unpaywall            (finds legal OA versions of paywalled papers via DOI)
  6. NCBI abstract        (complete PubMed abstract for any paper with a PMID)

Strategies 2, 5, 6 are additions over the original three-strategy design.
Together they significantly increase full-text and structured-section coverage.
"""

from nlp.fulltext.europepmc_fulltext import EuropePMCFullText
from nlp.fulltext.ncbi_pmc_fetcher import NCBIPMCFetcher
from nlp.fulltext.pdf_parser import PDFParser
from nlp.fulltext.web_scraper import WebScraper
from nlp.fulltext.unpaywall_fetcher import UnpaywallFetcher
from nlp.fulltext.ncbi_abstract_fetcher import NCBIAbstractFetcher


class FullTextOrchestrator:

    def __init__(self):
        self.europepmc  = EuropePMCFullText()
        self.ncbi_pmc   = NCBIPMCFetcher()
        self.pdf        = PDFParser()
        self.web        = WebScraper()
        self.unpaywall  = UnpaywallFetcher()
        self.ncbi_abs   = NCBIAbstractFetcher()

    def fetch(self, paper) -> dict | None:
        """
        Try all full-text strategies in priority order.
        Returns the first successful result, or None if all fail.

        The returned dict always contains fetch_source and fetch_status.
        Full text may be in 'full_text' (unstructured) or in section keys
        ('abstract', 'methods', 'results', 'discussion').
        """

        # ── Strategy 1: EuropePMC XML ─────────────────────────────────────────
        if getattr(paper, "pmcid", None):
            record = self.europepmc.fetch(paper.pmcid)
            if record:
                return record

        # ── Strategy 2: NCBI PMC full text (larger OA subset than EuropePMC) ──
        if getattr(paper, "pmcid", None):
            record = self.ncbi_pmc.fetch(paper.pmcid)
            if record:
                return record

        # ── Strategy 3: Direct PDF URL ────────────────────────────────────────
        if getattr(paper, "pdf_url", None):
            record = self.pdf.fetch(paper.pdf_url)
            if record:
                return record

        # ── Strategy 4: HTML scraping via full_text_url ───────────────────────
        if getattr(paper, "full_text_url", None):
            record = self.web.fetch(paper.full_text_url)
            if record:
                return record

        # ── Strategy 5: Unpaywall (legal OA versions of paywalled papers) ─────
        if getattr(paper, "doi", None):
            record = self.unpaywall.fetch(paper.doi)
            if record:
                return record

        # ── Strategy 6: NCBI PubMed abstract (complete abstract for any PMID) ─
        if getattr(paper, "pmid", None):
            record = self.ncbi_abs.fetch(paper.pmid)
            if record:
                return record

        return None
