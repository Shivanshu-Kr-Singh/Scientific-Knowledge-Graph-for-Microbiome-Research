"""
collectors/europepmc_collector.py
----------------------------------
Fetches papers from Europe PubMed Central (Europe PMC).

WHY EUROPE PMC IN ADDITION TO PUBMED?
  1. Europe PMC provides FULL TEXT for open-access papers — not just abstracts.
     This means we can extract Methods and Data Availability sections.
  2. It covers preprints from medRxiv and other sources that haven't yet been indexed
     by NCBI PubMed.
  3. It has a clean JSON REST API (no XML parsing needed).

API DOCS: https://europepmc.org/RestfulWebService
Base URL:  https://www.ebi.ac.uk/europepmc/webservices/rest/search
"""

from typing import Optional, List
from loguru import logger

from models import PaperRecord
from collectors.base_collector import BaseCollector


EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMCCollector(BaseCollector):
    """Collects papers from Europe PubMed Central."""

    source_name = "europepmc"

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        TWO-LAYER FILTER STRATEGY:

        Layer A — Positive: require human context explicitly.
          Require "human" or human-specific terms in title/abstract,
          OR use human-specific MeSH terms.
          Also catches papers that study humans without saying "human"
          in the title (e.g. "Gut microbiota diversity in IBD patients").

        Layer B — Negative: exclude known non-human study types.
          Zebrafish, mouse, rat, soil, plant, marine, food fermentation
          all use "microbiome" but are never about humans. NOT filters
          remove these cleanly.

        WHY BOTH LAYERS:
          Positive-only: misses papers that don't say "human" explicitly.
          Negative-only: misses papers where exclusion terms aren't in title.
          Together: high precision AND recall for human microbiome papers.
        """
        year_from = date_from[:4]
        year_to   = date_to[:4]

        positive = (
            f'(TITLE:"human microbiome" OR TITLE:"human microbiota" '
            f'OR TITLE:"human gut" OR TITLE:"gut microbiome" '
            f'OR TITLE:"gut microbiota" OR TITLE:"intestinal microbiome" '
            f'OR TITLE:"intestinal microbiota" OR TITLE:"oral microbiome" '
            f'OR TITLE:"skin microbiome" OR TITLE:"vaginal microbiome" '
            f'OR TITLE:"lung microbiome" OR TITLE:"metagenomics" '
            f'OR MH:"Gastrointestinal Microbiome" OR MH:"Microbiota" '
            f'OR (ABSTRACT:"human" AND ABSTRACT:"microbiome") '
            f'OR (ABSTRACT:"patients" AND ABSTRACT:"microbiota") '
            f'OR (ABSTRACT:"participants" AND ABSTRACT:"microbiome"))' 
        )

        negative = (
            f'NOT TITLE:"zebrafish" NOT TITLE:"murine" '
            f'NOT TITLE:"mouse model" NOT TITLE:"rat model" '
            f'NOT TITLE:"porcine" NOT TITLE:"bovine" NOT TITLE:"poultry" '
            f'NOT TITLE:"soil microbiome" NOT TITLE:"soil microbiota" '
            f'NOT TITLE:"plant microbiome" NOT TITLE:"rhizosphere" '
            f'NOT TITLE:"marine microbiome" NOT TITLE:"aquatic microbiome" '
            f'NOT TITLE:"fermented food" NOT TITLE:"fermented beverage" '
            f'NOT TITLE:"kombucha" NOT TITLE:"tepache" NOT TITLE:"kefir" '
            f'NOT TITLE:"tortoise" NOT TITLE:"frog" NOT TITLE:"zebrafish" '
            f'NOT TITLE:"insect" NOT TITLE:"honey bee" NOT TITLE:"coral"'
        )

        q = f'{positive} AND (PUB_YEAR:[{year_from} TO {year_to}]) {negative}'
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

        # Log total hit count once, on the first page — mirrors PubMed behaviour
        if page == 0:
            total = data.get("hitCount", 0)
            logger.info(f"[europepmc] Total results in Europe PMC: {total}")

        # Cache the raw response
        self._save_raw(f"page_{page}", data)

        return data

    def _extract_items(self, raw_page: dict) -> list:
        """Europe PMC nests results under resultList.result"""
        return raw_page.get("resultList", {}).get("result", [])

    def collect(
        self,
        query: str,
        date_from: str,
        date_to: str,
        max_results: int = 500,
        page_size: int = 1000,    # Europe PMC max per page is 1000
        start_offset: int = 0,
    ) -> List[PaperRecord]:
        """
        Override base collect() to paginate Europe PMC correctly.

        The base collect() uses max_results as page_size. Europe PMC caps
        at 1000 per page, so asking for 5000 returns 0 results (page too
        large). This override uses a fixed 1000-per-page loop.

        Europe PMC is 1-indexed — page 1 = first page.
        start_offset maps to: start_page = (start_offset // page_size) + 1
        """
        import datetime as dt

        logger.info(
            f"[{self.source_name}] Starting collection | "
            f"query='{query}' | {date_from} → {date_to} | max={max_results}"
        )

        query_params = self.build_query(query, date_from, date_to)
        papers: List[PaperRecord] = []

        # Convert numeric offset to 1-indexed page number
        rows      = min(page_size, 1000)
        start_page = (start_offset // rows) + 1 if start_offset > 0 else 1

        if start_page > 1:
            logger.info(f"[{self.source_name}] Resuming from page {start_page}")

        page = start_page

        while len(papers) < max_results:
            try:
                raw_page = self.fetch_page(query_params, page=page - 1, page_size=rows)
            except Exception as e:
                logger.error(f"[europepmc] Failed at page {page}: {e}")
                break

            items = self._extract_items(raw_page)
            if not items:
                logger.info(f"[europepmc] No more results at page {page}")
                break

            batch_count = 0
            for raw in items:
                if len(papers) >= max_results:
                    break
                paper = self.parse_record(raw)
                if paper:
                    paper.source       = self.source_name
                    paper.content_hash = self._compute_hash(paper)
                    paper.fetched_at   = dt.datetime.utcnow().isoformat()
                    papers.append(paper)
                    batch_count += 1

            logger.info(
                f"[europepmc] Page {page}: {batch_count} records | "
                f"Total so far: {len(papers)}"
            )

            # Fewer than requested → last page
            if len(items) < rows:
                logger.info(f"[europepmc] Reached end of results at page {page}")
                break

            page += 1

        logger.success(f"[europepmc] Collection complete: {len(papers)} papers")
        return papers

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
            keywords = [
                str(k)
                for k in raw.get("keywordList",{}).get("keyword",[])
                if k is not None]
            
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
