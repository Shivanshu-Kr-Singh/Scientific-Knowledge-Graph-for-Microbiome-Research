"""
collectors/europepmc_collector.py
----------------------------------
Fetches papers from Europe PubMed Central (Europe PMC).

WHY EUROPE PMC IN ADDITION TO PUBMED?
  1. Europe PMC provides FULL TEXT for open-access papers — not just abstracts.
     This means we can extract Methods and Data Availability sections.
  2. It covers preprints from bioRxiv/medRxiv that haven't yet been indexed
     by NCBI PubMed.
  3. It has a clean JSON REST API (no XML parsing needed).

API DOCS: https://europepmc.org/RestfulWebService
Base URL:  https://www.ebi.ac.uk/europepmc/webservices/rest/search
"""

from typing import Optional
from loguru import logger

from models import PaperRecord
from collectors.base_collector import BaseCollector


EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMCCollector(BaseCollector):
    """Collects papers from Europe PubMed Central."""

    source_name = "europepmc"

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Europe PMC uses a Lucene-style query syntax.

        KEY FIELDS:
          TITLE       → paper title
          ABSTRACT    → abstract text
          MH          → MeSH heading
          PUB_YEAR    → publication year
          ORGANISM    → species filter
          HAS_FT      → has full text (y/n)
          OPEN_ACCESS → is open access (y/n)

        We query for microbiome terms AND restrict to human studies.
        The (ORGANISM:"Homo sapiens") filter maps to the same concept as
        [Humans] in PubMed.
        """
        year_from = date_from[:4]   # "2024/01/01" → "2024"
        year_to   = date_to[:4]     # "2026/12/31" → "2026"

        q = (
            f'(TITLE:"microbiome" OR TITLE:"metagenomics" OR TITLE:"microbiota" '
            f'OR ABSTRACT:"human microbiome" OR MH:"Microbiota") '
            f'AND (PUB_YEAR:[{year_from} TO {year_to}]) '
            f'AND (ORGANISM:"Homo sapiens" OR SRC:MED OR SRC:PMC)'
        )

        logger.debug(f"[europepmc] Query: {q}")
        return {"q": q, "year_from": year_from, "year_to": year_to}

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Fetches one page from Europe PMC's search endpoint.

        Europe PMC uses cursor-based pagination (cursorMark) for deep paging,
        but for our use case, simple offset (page * size) works fine for
        up to a few thousand results.
        """
        params = {
            "query":      query_params["q"],
            "format":     "json",
            "pageSize":   page_size,
            "page":       page + 1,             # Europe PMC is 1-indexed
            "resultType": "core",               # "core" includes full metadata
            "synonym":    "true",               # Expand synonyms in query
        }

        response = self._get(f"{EPMC_BASE}/search", params=params)
        data = response.json()

        # Cache the raw response
        self._save_raw(f"page_{page}", data)

        return data

    def _extract_items(self, raw_page: dict) -> list:
        """Europe PMC nests results under resultList.result"""
        return raw_page.get("resultList", {}).get("result", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one Europe PMC JSON record.

        EUROPE PMC JSON STRUCTURE (key fields):
          {
            "id": "38765432",               ← PMID or Europe PMC ID
            "pmid": "38765432",
            "pmcid": "PMC11234567",
            "doi": "10.1038/...",
            "title": "...",
            "abstractText": "...",
            "authorList": {"author": [{"fullName": "Smith J", ...}]},
            "journalInfo": {
              "journal": {"title": "...", "issn": "..."},
              "volume": "15", "issue": "3",
              "dateOfPublication": "2024 Mar"
            },
            "pubTypeList": {"pubType": ["research-article", ...]},
            "keywordList": {"keyword": [...]},
            "meshHeadingList": {"meshHeading": [{"descriptorName": "..."}]},
            "isOpenAccess": "Y",
            "inEPMC": "Y",              ← Has full text in Europe PMC
            "citedByCount": 12
          }
        """
        try:
            title = raw.get("title", "").strip()
            if not title:
                return None

            # ── Authors ───────────────────────────────────────────────────────
            authors = []
            author_list = raw.get("authorList", {}).get("author", [])
            for a in author_list:
                # Europe PMC provides fullName directly — much easier than PubMed XML
                name = a.get("fullName") or f"{a.get('lastName', '')} {a.get('initials', '')}".strip()
                if name:
                    authors.append(name)

            # ── Journal & Date ────────────────────────────────────────────────
            ji = raw.get("journalInfo", {})
            journal_data = ji.get("journal", {})
            journal = journal_data.get("title")
            issn    = journal_data.get("issn") or journal_data.get("essn")

            # Europe PMC date comes as "2024 Mar" or "2024" — normalize it
            date_str = ji.get("dateOfPublication", "")
            pub_year = None
            pub_date = None
            if date_str:
                parts = date_str.strip().split()
                if parts[0].isdigit():
                    pub_year = int(parts[0])
                    pub_date = parts[0]   # At minimum we have the year

            # ── Article Types ─────────────────────────────────────────────────
            pub_types = raw.get("pubTypeList", {}).get("pubType", [])
            if isinstance(pub_types, str):
                pub_types = [pub_types]

            # ── MeSH Terms ────────────────────────────────────────────────────
            mesh_terms = [
                mh.get("descriptorName", "")
                for mh in raw.get("meshHeadingList", {}).get("meshHeading", [])
                if mh.get("descriptorName")
            ]

            # ── Keywords ──────────────────────────────────────────────────────
            keywords = raw.get("keywordList", {}).get("keyword", [])
            if isinstance(keywords, str):
                keywords = [keywords]

            # ── Open Access + Full Text ───────────────────────────────────────
            pmcid = raw.get("pmcid")
            is_oa = raw.get("isOpenAccess", "N") == "Y"
            in_epmc = raw.get("inEPMC", "N") == "Y"

            full_text_url = None
            if pmcid:
                full_text_url = f"https://europepmc.org/article/MED/{raw.get('pmid', pmcid)}"

            return PaperRecord(
                pmid=raw.get("pmid"),
                pmcid=pmcid,
                doi=raw.get("doi"),
                title=title,
                abstract=raw.get("abstractText"),
                authors=authors,
                keywords=keywords,
                journal=journal,
                issn=issn,
                publication_date=pub_date,
                publication_year=pub_year,
                volume=ji.get("volume"),
                issue=ji.get("issue"),
                article_types=pub_types,
                mesh_terms=mesh_terms,
                is_open_access=is_oa,
                full_text_url=full_text_url,
                citation_count=raw.get("citedByCount"),
                is_preprint=False,
            )

        except Exception as e:
            logger.warning(f"[europepmc] Failed to parse record: {e}")
            return None
