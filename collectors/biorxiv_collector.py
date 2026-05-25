"""
Fetches preprints from bioRxiv and medRxiv.
WHY PREPRINTS?
In fast-moving fields like microbiome research, important work appears
on bioRxiv months or years before journal publication. Tracking preprints lets you:-
- Catch cutting-edge work early
- Track which preprints eventually get published (and link them)
- See how conclusions change between preprint and final version

Returns JSON.
"""

from typing import Optional 
from loguru import logger
from models import PaperRecord
from collectors.base_collector import BaseCollector

BIORXIV_BASE = "https://api.biorxiv.org"

class BioRxivCollector(BaseCollector):
    """Fetches preprints from bioRxiv and medRxiv."""

    source_name = "biorxiv"

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        bioRxiv API filters by date range and category, not full-text search.
        We filter for microbiome-relevant categories:
          - microbiology: obvious
          - genomics: covers metagenomics papers
          - bioinformatics: covers analysis methods
          - physiology: some microbiome-host interaction papers

        We'll do keyword filtering AFTER fetching (client-side) because the
        API doesn't support full-text search.

        Date format for bioRxiv API: YYYY-MM-DD
        """
        date_from_clean = date_from.replace("/", "-")[:10]
        date_to_clean   = date_to.replace("/", "-")[:10]

        return {
            "date_from":  date_from_clean,
            "date_to":    date_to_clean,
            "servers":    ["biorxiv", "medrxiv"],
        }

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        bioRxiv API URL format:
          /details/{server}/{date_from}/{date_to}/{cursor}/{format}

        cursor = page number (0-indexed)
        Each page returns up to 100 results.

        We fetch from both bioRxiv and medRxiv.
        """
        all_results = []

        for server in query_params["servers"]:
            url = (
                f"{BIORXIV_BASE}/details/{server}/"
                f"{query_params['date_from']}/"
                f"{query_params['date_to']}/"
                f"{page * 100}/json"
            )

            try:
                response = self._get(url)
                data = response.json()
                collection = data.get("collection", [])

                # Client-side keyword filtering:
                # bioRxiv API returns ALL papers in a date range regardless of topic.
                # We only keep papers mentioning microbiome-related terms.
                MICROBIOME_KEYWORDS = {"microbiome", "microbiota", "metagenom", "16s", "microbial",
    "dysbiosis", "probiotics", "gut bacteria", "microorganism",
    "bacteriome", "virome", "mycobiome"}

                filtered = [
                    item for item in collection
                    if self._is_microbiome_related(item, MICROBIOME_KEYWORDS)
                ]

                # Tag each item with which server it came from
                for item in filtered:
                    item["_server"] = server

                all_results.extend(filtered)
                logger.info(f"[biorxiv] {server} page {page}: {len(filtered)}/{len(collection)} papers kept after filtering")

            except Exception as e:
                logger.error(f"[biorxiv] Failed to fetch from {server}: {e}")

        self._save_raw(f"page_{page}", {"results": all_results})
        return {"records": all_results}

    def _is_microbiome_related(self, item: dict, keywords: set) -> bool:
        """
        Returns True if the paper title or abstract contains any microbiome keyword.
        Case-insensitive substring match.
        """
        text = (
            (item.get("title") or "") + " " +
            (item.get("abstract") or "")
        ).lower()
        return any(kw in text for kw in keywords)

    def _extract_items(self, raw_page: dict) -> list:
        return raw_page.get("records", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one bioRxiv/medRxiv JSON record.

        BIORXIV JSON STRUCTURE:
          {
            "doi": "10.1101/2024.01.15.575821",
            "title": "...",
            "authors": "Smith J; Jones K; Lee M",  ← semicolon-separated string!
            "author_corresponding": "Smith J",
            "author_corresponding_institution": "MIT",
            "date": "2024-01-15",
            "version": "1",                          ← preprint version number
            "type": "new results",
            "category": "microbiology",
            "jatsxml": "https://...",               ← full XML if available
            "abstract": "...",
            "published": "10.1038/...",              ← DOI of final published version, if known
            "server": "biorxiv"
          }
        """
        try:
            title = (raw.get("title") or "").strip()
            if not title:
                return None

            doi = raw.get("doi")

            # bioRxiv authors come as a semicolon-separated string, not a list
            authors_raw = raw.get("authors", "")
            if isinstance(authors_raw, str):
                authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
            else:
                authors = authors_raw

            # If there's a published DOI, the preprint became a paper
            published_doi = raw.get("published")
            is_published  = bool(published_doi and published_doi != "NA")

            return PaperRecord(
                doi=doi,
                title=title,
                abstract=raw.get("abstract"),
                authors=authors,
                publication_date=raw.get("date"),
                publication_year=int(raw.get("date", "2024")[:4]) if raw.get("date") else None,
                journal=f"{raw.get('_server', 'biorxiv').capitalize()} (preprint)",
                article_types=[raw.get("type", "preprint"), raw.get("category", "")],
                is_open_access=True,   # All bioRxiv preprints are open access
                is_preprint=not is_published,
            )

        except Exception as e:
            logger.warning(f"[biorxiv] Failed to parse record: {e}")
            return None
