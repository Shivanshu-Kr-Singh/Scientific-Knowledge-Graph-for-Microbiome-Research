"""
collectors/pubmed_collector.py
-------------------------------
Fetches research papers from PubMed using NCBI's E-utilities API.

HOW E-UTILITIES WORKS (2-step process):
  Step 1: esearch → Give it a query, get back a WebEnv + query_key
  Step 2: efetch  → Use WebEnv + query_key to fetch full records in pages

  We use WebHistory (usehistory=y) which avoids PubMed's hard limit of
  retstart ≤ 9,999 on direct PMID-based pagination. WebHistory stores
  the result set server-side and lets us page through any number of results.

API DOCS: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Optional, List
from loguru import logger

from config import NCBI_EMAIL, NCBI_API_KEY, PUBMED_MESH_TERMS
from models import PaperRecord
from collectors.base_collector import BaseCollector


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# PubMed hard-limits retstart to 9,999 for direct PMID pagination.
# WebHistory has no such limit — always use it.
PUBMED_MAX_RETSTART = 9999


class PubMedCollector(BaseCollector):
    """Collects papers from PubMed via NCBI E-utilities."""

    source_name = "pubmed"

    def __init__(self):
        super().__init__()
        # Standard params attached to every E-utilities request
        self._base_params = {
            "email": NCBI_EMAIL,
            "tool":  "MicrobiomeMiner",
        }
        if NCBI_API_KEY:
            # With an API key you get 10 req/sec instead of 3.
            # Get a free one at: https://www.ncbi.nlm.nih.gov/account/
            self._base_params["api_key"] = NCBI_API_KEY
            logger.info("[pubmed] Using NCBI API key — 10 req/sec limit")
        else:
            logger.warning("[pubmed] No NCBI API key — limited to 3 req/sec")

    # ─── Step 1: Build Query ───────────────────────────────────────────────────

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Builds a PubMed search query string using E-utilities syntax.

        PUBMED QUERY SYNTAX:
          term[field]          → e.g. "microbiome"[MeSH Terms]
          term1 AND term2      → both must match
          term1 OR term2       → either can match
          date_from:date_to[PDAT]  → publication date range

        WHY USE MESH TERMS?
          MeSH (Medical Subject Headings) is a controlled vocabulary. A paper
          about "gut flora" will be tagged with the MeSH term "Gastrointestinal
          Microbiome" even if those exact words don't appear in the title.
          This gives us much better recall than keyword search alone.

        EXAMPLE QUERY BUILT:
          ("Microbiota"[MeSH] OR "Gastrointestinal Microbiome"[MeSH] OR
           "human microbiome"[TIAB]) AND
          "2024/01/01"[PDAT]:"2026/12/31"[PDAT] AND "Humans"[MeSH]
        """
        # Build MeSH term clause
        mesh_clause = " OR ".join([f'"{term}"[MeSH Terms]' for term in PUBMED_MESH_TERMS])

        # TIAB = Title + Abstract keyword fallback (catches newer terms not yet in MeSH)
        keyword_clause = f'"{query}"[Title/Abstract]'

        # Limit to human studies only — we don't want mouse microbiome papers
        human_filter = '"Humans"[MeSH Terms]'

        # Date range filter (PDAT = Publication Date)
        date_clause = f'"{date_from}"[PDAT]:"{date_to}"[PDAT]'

        full_query = f"({mesh_clause} OR {keyword_clause}) AND {date_clause} AND {human_filter}"

        logger.debug(f"[pubmed] Query: {full_query}")

        return {
            "query":      full_query,
            "date_from":  date_from,
            "date_to":    date_to,
        }

    # ─── Step 1a: esearch — register query in WebHistory ─────────────────────

    def _esearch_webhistory(self, query: str) -> tuple:
        """
        Runs esearch with usehistory=y — stores results server-side.

        Returns (WebEnv, query_key, total_count).
        WebEnv + query_key are then passed to efetch for any retstart.

        WHY WEBHISTORY:
          Direct PMID pagination (retstart) is hard-capped at 9,999 by PubMed.
          WebHistory has no such limit — you can page through all 70,000+
          results by incrementing retstart in steps of page_size.
        """
        params = {
            **self._base_params,
            "db":         "pubmed",
            "term":       query,
            "retmax":     0,        # We only want the WebEnv, not PMIDs yet
            "retmode":    "json",
            "usehistory": "y",
        }
        response = self._get(f"{EUTILS_BASE}/esearch.fcgi", params=params)

        # Clean response — esearch JSON can contain \n inside string values
        raw  = response.content.decode("utf-8", errors="replace")
        data = json.loads(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", raw))

        result    = data.get("esearchresult", {})
        web_env   = result.get("webenv", "")
        query_key = result.get("querykey", "1")
        total     = int(result.get("count", 0))

        if not web_env:
            raise RuntimeError(f"[pubmed] esearch returned no WebEnv: {result}")

        logger.info(f"[pubmed] Total results in PubMed: {total}")
        return web_env, query_key, total

    # ─── Step 1b: efetch — fetch a page using WebHistory ─────────────────────

    def _efetch_page(self, web_env: str, query_key: str,
                     retstart: int, retmax: int) -> ET.Element:
        """
        Fetches one page of full records using WebHistory.
        No retstart limit — can page through all results.
        Strips illegal XML control characters before parsing.
        """
        params = {
            **self._base_params,
            "db":        "pubmed",
            "WebEnv":    web_env,
            "query_key": query_key,
            "retstart":  retstart,
            "retmax":    retmax,
            "retmode":   "xml",
            "rettype":   "abstract",
        }
        response = self._get(f"{EUTILS_BASE}/efetch.fcgi", params=params)

        # Strip XML-illegal control chars (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F)
        raw   = response.content.decode("utf-8", errors="replace")
        clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
        return ET.fromstring(clean.encode("utf-8"))

    # ─── Main collection loop ─────────────────────────────────────────────────

    def collect(
        self,
        query: str,
        date_from: str,
        date_to: str,
        max_results: int = 500,
        page_size: int = 200,
        start_offset: int = 0,
    ) -> List[PaperRecord]:
        """
        Collects papers from PubMed using WebHistory pagination.

        PUBMED PAGINATION HARD LIMIT:
          PubMed's efetch retstart is capped at 9,999. A single query session
          can return at most ~10,000 papers. This is an NCBI API constraint.

        STRATEGY FOR 60,000+ PAPERS:
          We split the date range into monthly sub-ranges. Each sub-range
          gets its own WebHistory session with up to 10,000 results.
          start_offset tracks which month we're on across runs.

          For 2024-01-01 to 2026-12-31 = 36 months.
          Each month can have up to ~3,000 microbiome papers = 100,000+ total.

          start_offset encodes: (month_index * 10000) + retstart_within_month
          e.g. offset=15000 → month_index=1, retstart=5000
               offset=0     → month_index=0, retstart=0
        """
        import datetime as dt
        from dateutil.relativedelta import relativedelta
        from datetime import date

        logger.info(
            f"[{self.source_name}] Starting collection | "
            f"query='{query}' | {date_from} → {date_to}"
        )

        # Parse date range into list of monthly sub-ranges
        try:
            d_from = dt.datetime.strptime(date_from[:10].replace("/", "-"), "%Y-%m-%d").date()
            d_to   = dt.datetime.strptime(date_to[:10].replace("/", "-"),   "%Y-%m-%d").date()
        except Exception:
            d_from = date(2024, 1, 1)
            d_to   = date(2026, 12, 31)

        # Build monthly sub-ranges
        months = []
        cur = d_from.replace(day=1)
        while cur <= d_to:
            month_end = (cur + relativedelta(months=1)) - relativedelta(days=1)
            month_end = min(month_end, d_to)
            months.append((
                cur.strftime("%Y/%m/%d"),
                month_end.strftime("%Y/%m/%d"),
            ))
            cur = cur + relativedelta(months=1)

        MONTH_WINDOW = PUBMED_MAX_RETSTART  # max retstart per month session

        # Decode start_offset into (month_index, retstart_within_month)
        month_index    = start_offset // (MONTH_WINDOW + 1)
        retstart_start = start_offset %  (MONTH_WINDOW + 1)

        if start_offset > 0:
            logger.info(
                f"[pubmed] Resuming from month {month_index + 1}/{len(months)} "
                f"at retstart {retstart_start}"
            )

        papers: List[PaperRecord] = []
        self._last_retstart = start_offset   # default — updated as we go

        for m_idx in range(month_index, len(months)):
            if len(papers) >= max_results:
                break

            m_from, m_to = months[m_idx]
            query_params = self.build_query(query, m_from, m_to)

            # Register this month's query in WebHistory
            try:
                web_env, query_key, total = self._esearch_webhistory(
                    query_params["query"]
                )
            except Exception as e:
                logger.error(f"[pubmed] esearch failed for {m_from}–{m_to}: {e}")
                continue

            if total == 0:
                logger.info(f"[pubmed] No results for {m_from}–{m_to}")
                # Advance to next month
                self._last_retstart = (m_idx + 1) * (MONTH_WINDOW + 1)
                continue

            logger.info(
                f"[pubmed] Month {m_idx + 1}/{len(months)}: "
                f"{m_from}→{m_to} | {total} total results"
            )

            # retstart within this month — 0 for new months, resumed for current
            retstart = retstart_start if m_idx == month_index else 0

            while len(papers) < max_results:
                if retstart > MONTH_WINDOW:
                    logger.info(
                        f"[pubmed] Reached retstart limit for {m_from}–{m_to}"
                    )
                    break

                batch = min(page_size, max_results - len(papers),
                            MONTH_WINDOW - retstart)

                try:
                    xml_root = self._efetch_page(
                        web_env, query_key, retstart, batch
                    )
                except Exception as e:
                    logger.error(
                        f"[pubmed] efetch failed at {m_from}–{m_to} "
                        f"offset {retstart}: {e}"
                    )
                    break

                articles = xml_root.findall(".//PubmedArticle")
                if not articles:
                    logger.info(
                        f"[pubmed] No more articles at {m_from}–{m_to} "
                        f"offset {retstart}"
                    )
                    break

                batch_count = 0
                for article_elem in articles:
                    if len(papers) >= max_results:
                        break
                    paper = self.parse_record({"_xml_element": article_elem})
                    if paper:
                        paper.source       = self.source_name
                        paper.content_hash = self._compute_hash(paper)
                        paper.fetched_at   = dt.datetime.utcnow().isoformat()
                        papers.append(paper)
                        batch_count += 1

                # Track real position for cursor saving:
                # encoded as month_index * (MONTH_WINDOW+1) + retstart
                self._last_retstart = (
                    m_idx * (MONTH_WINDOW + 1) + retstart
                )

                logger.info(
                    f"[pubmed] {m_from}–{m_to} offset {retstart}: "
                    f"{batch_count} records | Total: {len(papers)}"
                )

                if len(articles) < batch:
                    # End of this month's results — move to next month
                    logger.info(
                        f"[pubmed] Exhausted {m_from}–{m_to} "
                        f"({retstart + len(articles)} of {total})"
                    )
                    self._last_retstart = (m_idx + 1) * (MONTH_WINDOW + 1)
                    break

                retstart += len(articles)

        logger.success(f"[pubmed] Collection complete: {len(papers)} papers")
        return papers

    # ─── Legacy interface methods (kept for base class compatibility) ─────────

    def _esearch(self, query: str, retstart: int, retmax: int) -> List[str]:
        """Legacy method — not used by the overridden collect(). Kept for compatibility."""
        return []

    def _efetch_xml(self, pmids: List[str]) -> ET.Element:
        """Legacy method — not used by the overridden collect(). Kept for compatibility."""
        return ET.Element("PubmedArticleSet")

    def _efetch_chunk(self, pmids: List[str]) -> ET.Element:
        """Legacy method — not used by the overridden collect(). Kept for compatibility."""
        return ET.Element("PubmedArticleSet")

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """Not used — overridden by collect() above. Required by BaseCollector."""
        return {"records": []}

    def _extract_items(self, raw_page: dict) -> list:
        return raw_page.get("records", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one PubmedArticle XML element into a PaperRecord.

        XML STRUCTURE OF A PUBMED ARTICLE:
          PubmedArticle
            MedlineCitation
              PMID
              Article
                ArticleTitle
                Abstract / AbstractText
                AuthorList / Author (LastName, ForeName, Affiliation)
                Journal
                  Title, ISOAbbreviation, ISSN
                  JournalIssue (Volume, Issue, PubDate)
                PublicationTypeList / PublicationType
                ELocationID (DOI)
              MeshHeadingList / MeshHeading / DescriptorName
            PubmedData
              ArticleIdList / ArticleId
        """
        elem = raw.get("_xml_element")
        if elem is None:
            return None

        try:
            mc = elem.find("MedlineCitation")
            if mc is None:
                return None

            art = mc.find("Article")
            if art is None:
                return None

            # ── PMID ──────────────────────────────────────────────────────────
            pmid_elem = mc.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None

            # ── DOI ───────────────────────────────────────────────────────────
            # DOIs can appear in two places in PubMed XML
            doi = None
            for eid in elem.findall(".//ELocationID[@EIdType='doi']"):
                doi = eid.text
                break
            if not doi:
                for aid in elem.findall(".//ArticleId[@IdType='doi']"):
                    doi = aid.text
                    break

            # ── Title ─────────────────────────────────────────────────────────
            title_elem = art.find("ArticleTitle")
            title = "".join(title_elem.itertext()) if title_elem is not None else ""
            if not title.strip():
                return None   # Skip records with no title

            # ── Abstract ──────────────────────────────────────────────────────
            # Structured abstracts have multiple AbstractText elements with labels
            # (Background, Methods, Results, Conclusions). We join them all.
            abstract_parts = []
            for atext in art.findall(".//AbstractText"):
                label = atext.get("Label", "")
                text  = "".join(atext.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = "\n".join(abstract_parts) if abstract_parts else None

            # ── Authors ───────────────────────────────────────────────────────
            authors = []
            for author in art.findall(".//Author"):
                last  = author.findtext("LastName", "")
                fore  = author.findtext("ForeName", "")
                cname = author.findtext("CollectiveName", "")
                if cname:
                    authors.append(cname)
                elif last:
                    authors.append(f"{last}, {fore}".strip(", "))

            # ── Journal ───────────────────────────────────────────────────────
            journal_elem = art.find("Journal")
            journal      = journal_elem.findtext("Title")        if journal_elem else None
            j_abbrev     = journal_elem.findtext("ISOAbbreviation") if journal_elem else None
            issn_elem    = journal_elem.find("ISSN")             if journal_elem else None
            issn         = issn_elem.text                        if issn_elem is not None else None

            # ── Date ──────────────────────────────────────────────────────────
            pub_year = None
            pub_date_str = None
            ji = art.find(".//JournalIssue/PubDate")
            if ji is not None:
                year  = ji.findtext("Year")
                month = ji.findtext("Month", "01")
                day   = ji.findtext("Day",   "01")
                if year:
                    pub_year = int(year)
                    # Month might be abbreviated ("Jan") — normalize to "01"
                    month_map = {
                        "Jan":"01","Feb":"02","Mar":"03","Apr":"04",
                        "May":"05","Jun":"06","Jul":"07","Aug":"08",
                        "Sep":"09","Oct":"10","Nov":"11","Dec":"12"
                    }
                    month = month_map.get(month, month).zfill(2)
                    pub_date_str = f"{year}-{month}-{day.zfill(2)}"

            volume = art.findtext(".//JournalIssue/Volume")
            issue  = art.findtext(".//JournalIssue/Issue")
            pages  = art.findtext(".//Pagination/MedlinePgn")

            # ── Article Types ─────────────────────────────────────────────────
            # PubMed explicitly tags papers as "Review", "Clinical Trial", etc.
            # We keep the raw tags; Layer 3 will normalize them.
            article_types = [
                pt.text for pt in art.findall(".//PublicationType")
                if pt.text
            ]

            # ── MeSH Terms ────────────────────────────────────────────────────
            mesh_terms = [
                dn.text for dn in mc.findall(".//MeshHeading/DescriptorName")
                if dn.text
            ]

            # ── Keywords ──────────────────────────────────────────────────────
            keywords = [
                kw.text for kw in mc.findall(".//KeywordList/Keyword")
                if kw.text
            ]

            # ── Open Access / Full Text ────────────────────────────────────────
            pmcid = None
            for aid in elem.findall(".//ArticleId[@IdType='pmc']"):
                pmcid = aid.text
                break
            is_oa = pmcid is not None
            full_text_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None

            return PaperRecord(
                pmid=pmid,
                pmcid=pmcid,
                doi=doi,
                title=title.strip(),
                abstract=abstract,
                authors=authors,
                keywords=keywords,
                journal=journal,
                journal_abbrev=j_abbrev,
                issn=issn,
                publication_date=pub_date_str,
                publication_year=pub_year,
                volume=volume,
                issue=issue,
                pages=pages,
                article_types=article_types,
                mesh_terms=mesh_terms,
                is_open_access=is_oa,
                full_text_url=full_text_url,
                is_preprint=False,
            )

        except Exception as e:
            logger.warning(f"[pubmed] Failed to parse record: {e}")
            return None   # Skip bad records rather than crash the whole job
