"""
collectors/pubmed_collector.py
-------------------------------
Fetches research papers from PubMed using NCBI's E-utilities API.

WHY PUBMED IS THE PRIMARY SOURCE:
  PubMed indexes 35+ million citations from MEDLINE and life science journals.
  For human microbiome research specifically, it's the most comprehensive
  database. The E-utilities API is free, well-documented, and returns
  structured XML/JSON with MeSH terms — which are invaluable for NLP.

HOW E-UTILITIES WORKS (2-step process):
  Step 1: esearch → Give it a query, get back a LIST of PMIDs
  Step 2: efetch  → Give it those PMIDs, get back FULL records

  We never skip step 1 and go straight to efetch because we need the
  PMIDs to know which records to fetch.

API DOCS: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

import xml.etree.ElementTree as ET
from typing import Optional, List
from loguru import logger

from config import NCBI_EMAIL, NCBI_API_KEY, PUBMED_MESH_TERMS
from models import PaperRecord
from collectors.base_collector import BaseCollector


# E-utilities base URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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

    # ─── Step 1a: esearch — get PMIDs ─────────────────────────────────────────

    def _esearch(self, query: str, retstart: int, retmax: int) -> List[str]:
        """
        Searches PubMed and returns a list of PMIDs.

        retstart: offset (for pagination)
        retmax:   how many PMIDs to return

        RETURNS: list of PMID strings like ["38765432", "38712345", ...]
        """
        params = {
            **self._base_params,
            "db":       "pubmed",
            "term":     query,
            "retstart": retstart,
            "retmax":   retmax,
            "retmode":  "json",
            "usehistory": "n",  # We'll use the PMID list directly
        }

        response = self._get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
        data = response.json()

        pmids = data.get("esearchresult", {}).get("idlist", [])
        total = int(data.get("esearchresult", {}).get("count", 0))

        # Always log total on the first call of each run (retstart may be non-zero on resumed runs)
        if not getattr(self, "_total_logged", False):
            logger.info(f"[pubmed] Total results in PubMed: {total}")
            self._total_logged = True

        return pmids

    # ─── Step 1b: efetch — get full records from PMIDs ────────────────────────

    def _efetch_xml(self, pmids: List[str]) -> ET.Element:
        """
        Fetches full PubMed records for a list of PMIDs.
        Returns the root XML element (PubmedArticleSet).

        WHY XML NOT JSON?
          PubMed's JSON mode doesn't include all fields (notably MeSH terms
          and detailed author affiliations). XML has everything.
          We parse it manually using Python's built-in xml.etree module.
        """
        params = {
            **self._base_params,
            "db":      "pubmed",
            "id":      ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        response = self._get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        root = ET.fromstring(response.content)
        return root

    # ─── Implement BaseCollector interface ────────────────────────────────────

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Combines esearch (get PMIDs) + efetch (get records) for one page.
        Returns a dict that _extract_items will process.
        """
        retstart = page * page_size

        # Reset the total-logged flag at the start of each collect() run
        # so the total is always printed once per run regardless of start page
        if not hasattr(self, "_total_logged") or page == 0:
            self._total_logged = False

        pmids = self._esearch(query_params["query"], retstart=retstart, retmax=page_size)

        if not pmids:
            return {"records": []}

        # Save raw PMID list
        self._save_raw(f"pmids_page{page}", {"pmids": pmids, "query": query_params})

        # Fetch full XML records
        xml_root = self._efetch_xml(pmids)

        # Convert XML articles to raw dicts for parse_record to process
        articles = []
        for article_elem in xml_root.findall(".//PubmedArticle"):
            articles.append({"_xml_element": article_elem})

        return {"records": articles}

    def _extract_items(self, raw_page: dict) -> list:
        """Override base — our records are already a list of dicts."""
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
