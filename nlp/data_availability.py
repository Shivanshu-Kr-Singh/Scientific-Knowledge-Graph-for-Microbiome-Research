"""
nlp/data_availability.py
--------------------------
Extracts structured data availability information from papers.

WHY THIS IS A DEDICATED MODULE:
  Data availability is a core requirement of this project. Knowing whether
  a paper's sequencing data is publicly accessible determines whether you
  can actually reproduce or build on the study. This module:
    - Finds the data availability section
    - Extracts accession numbers (SRA, GEO, ArrayExpress, Zenodo, etc.)
    - Classifies availability status: open / restricted / not_stated
    - Extracts repository names and direct URLs

ACCESSION NUMBER FORMATS BY REPOSITORY:
  NCBI SRA:        SRP/SRR/SRX/SRS + digits   e.g. SRP123456, SRR7654321
  NCBI GEO:        GSE/GSM + digits            e.g. GSE123456
  NCBI BioProject: PRJNA + digits              e.g. PRJNA123456
  NCBI BioSample:  SAMN + digits               e.g. SAMN12345678
  ENA (Europe):    ERP/ERR/ERX/ERS + digits    e.g. ERP123456
                   PRJEB + digits              e.g. PRJEB12345
  ArrayExpress:    E-MTAB- + digits            e.g. E-MTAB-1234
  DDBJ:            DRP/DRR + digits            e.g. DRP001234
  Zenodo:          zenodo.org/record/XXXXXXX
  Figshare:        figshare.com/articles/XXXXXXX
  GitHub:          github.com/user/repo
  OSF:             osf.io/XXXXX
  Dryad:           datadryad.org/stash/dataset/doi:10.5061/...
"""

import re
from typing import Optional, List
from loguru import logger

from nlp.enriched_record import DataAvailabilityInfo, ParsedSection


# ── Accession number regex patterns ───────────────────────────────────────────

ACCESSION_PATTERNS = {
    "SRA": [
        r"\b(SRP\d{6,9})\b",       # SRA Study
        r"\b(SRR\d{6,9})\b",       # SRA Run
        r"\b(SRX\d{6,9})\b",       # SRA Experiment
        r"\b(SRS\d{6,9})\b",       # SRA Sample
    ],
    "NCBI BioProject": [
        r"\b(PRJNA\d{6,9})\b",
        r"\b(PRJEB\d{5,8})\b",
    ],
    "NCBI BioSample": [
        r"\b(SAMN\d{8,11})\b",
        r"\b(SAME\d{7,10})\b",
    ],
    "GEO": [
        r"\b(GSE\d{4,8})\b",       # GEO Series
        r"\b(GSM\d{5,9})\b",       # GEO Sample
    ],
    "ENA": [
        r"\b(ERP\d{6,9})\b",
        r"\b(ERR\d{6,9})\b",
        r"\b(ERX\d{6,9})\b",
        r"\b(ERS\d{6,9})\b",
    ],
    "ArrayExpress": [
        r"\b(E-MTAB-\d{3,6})\b",
        r"\b(E-GEOD-\d{4,7})\b",
        r"\b(E-TABM-\d{3,6})\b",
    ],
    "DDBJ": [
        r"\b(DRP\d{6,9})\b",
        r"\b(DRR\d{6,9})\b",
    ],
    "MGnify": [
        r"\b(MGP\d{4,7})\b",
        r"\bMGYG[A-Z0-9]{8,12}\b",
    ],
}

# ── URL patterns ──────────────────────────────────────────────────────────────

URL_PATTERNS = {
    "Zenodo":   r"https?://zenodo\.org/(?:record|deposit)/\d+",
    "Figshare": r"https?://(?:figshare\.com|doi\.org/10\.6084)/\S+",
    "GitHub":   r"https?://github\.com/[\w\-]+/[\w\-]+(?:/[\w\-\.]+)*",
    "OSF":      r"https?://osf\.io/[a-zA-Z0-9]{5,}",
    "Dryad":    r"https?://datadryad\.org/\S+",
    "Mendeley": r"https?://data\.mendeley\.com/datasets/\S+",
    "Harvard Dataverse": r"https?://dataverse\.harvard\.edu/\S+",
    "NCBI SRA": r"https?://www\.ncbi\.nlm\.nih\.gov/(?:sra|bioproject|biosample)/\S+",
    "GEO":      r"https?://www\.ncbi\.nlm\.nih\.gov/geo/query/acc\.cgi\?acc=\S+",
    "ENA":      r"https?://(?:www\.)?ebi\.ac\.uk/ena/\S+",
}

# ── Availability status signals ───────────────────────────────────────────────

OPEN_SIGNALS = [
    r"data (are|is) (publicly |freely )?(available|accessible)",
    r"publicly available",
    r"freely available",
    r"open access",
    r"deposited (at|in|to)",
    r"available (at|from|in|on)",
    r"can be (accessed|downloaded|found) (at|from)",
    r"uploaded to",
    r"submitted to",
]

RESTRICTED_SIGNALS = [
    r"available (upon|on) (reasonable )?request",
    r"upon reasonable request",
    r"available from the (corresponding |)author",
    r"request to the author",
    r"may be made available",
    r"due to privacy",
    r"data cannot be shared",
    r"ethical restrictions",
    r"institutional data sharing agreement",
    r"data will be made available",
    r"available to (qualified |bona fide )?researchers",
    r"access can be requested",
    r"data access committee",
    r"controlled access",
]

NOT_APPLICABLE_SIGNALS = [
    r"no (new )?data (were|was) (generated|created|collected)",
    r"all data (are|is) (contained|included) (within|in) (the|this)",
    r"data sharing not applicable",
    r"data sharing is not applicable",
    r"not applicable",
]


class DataAvailabilityExtractor:
    """
    Extracts structured data availability information from paper sections.
    """

    def extract(
        self,
        sections: List[ParsedSection],
        abstract: Optional[str] = None,
        full_text: Optional[str] = None,
    ) -> DataAvailabilityInfo:
        """
        Main extraction method.

        SEARCH ORDER:
          1. Look for a dedicated data availability section first
             (most accurate — authors wrote this specifically)
          2. Search the full text if no dedicated section
          3. Search the abstract as last resort
        """
        # Find dedicated data availability section
        da_section = next(
            (s for s in sections if s.section_type == "data_availability"),
            None
        )

        if da_section:
            logger.debug("[data_avail] Found dedicated data availability section")
            return self._parse_text(da_section.content, source="dedicated_section")

        # Search full text
        if full_text:
            da_text = self._find_da_in_full_text(full_text)
            if da_text:
                logger.debug("[data_avail] Found data availability in full text")
                return self._parse_text(da_text, source="full_text")

        # Search abstract
        if abstract:
            da_text = self._find_da_in_full_text(abstract)
            if da_text:
                return self._parse_text(da_text, source="abstract")

        # Nothing found
        return DataAvailabilityInfo(status="not_stated")

    def _find_da_in_full_text(self, text: str) -> Optional[str]:
        """
        Searches for data availability language in unstructured text.
        Returns the surrounding context (up to 500 chars) if found.
        """
        # Look for the phrase "data availability" and grab surrounding context
        match = re.search(
            r"(data\s+(availability|sharing|access)|availability\s+of\s+data)",
            text, re.IGNORECASE
        )
        if match:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 500)
            return text[start:end]

        # Also look for accession numbers in the text
        for repo, patterns in ACCESSION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Return the context around the accession number
                    m = re.search(pattern, text, re.IGNORECASE)
                    if m:
                        start = max(0, m.start() - 100)
                        end = min(len(text), m.end() + 200)
                        return text[start:end]

        return None

    def _parse_text(self, text: str, source: str = "") -> DataAvailabilityInfo:
        """
        Parses the data availability text into structured fields.
        Extracts: accession numbers, repository names, URLs, status.
        """
        accessions = []
        repositories = []
        urls = []

        # ── Extract accession numbers ──────────────────────────────────────────
        for repo_name, patterns in ACCESSION_PATTERNS.items():
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                if found:
                    accessions.extend(found)
                    if repo_name not in repositories:
                        repositories.append(repo_name)

        # ── Extract URLs ───────────────────────────────────────────────────────
        for repo_name, url_pattern in URL_PATTERNS.items():
            found = re.findall(url_pattern, text, re.IGNORECASE)
            if found:
                urls.extend(found)
                if repo_name not in repositories:
                    repositories.append(repo_name)

        # ── Determine availability status ──────────────────────────────────────
        text_lower = text.lower()

        # Not applicable (no data generated)
        for signal in NOT_APPLICABLE_SIGNALS:
            if re.search(signal, text_lower):
                return DataAvailabilityInfo(
                    status="not_stated",
                    raw_text=text,
                    notes="No primary data generated",
                )

        # Open access with accession
        if accessions:
            return DataAvailabilityInfo(
                status="accession_linked",
                raw_text=text,
                accession_numbers=list(dict.fromkeys(accessions)),   # deduplicated
                repositories=repositories,
                urls=list(dict.fromkeys(urls)),
            )

        # Open access with URL
        if urls:
            return DataAvailabilityInfo(
                status="open",
                raw_text=text,
                repositories=repositories,
                urls=list(dict.fromkeys(urls)),
            )

        # Check for open signals
        for signal in OPEN_SIGNALS:
            if re.search(signal, text_lower):
                return DataAvailabilityInfo(
                    status="open",
                    raw_text=text,
                    repositories=repositories,
                )

        # Check for restricted signals
        for signal in RESTRICTED_SIGNALS:
            if re.search(signal, text_lower):
                # Extract any notes about how to request
                notes = self._extract_request_info(text)
                return DataAvailabilityInfo(
                    status="restricted",
                    raw_text=text,
                    notes=notes,
                )

        # Has a data availability section but unclear status
        return DataAvailabilityInfo(
            status="not_stated",
            raw_text=text,
        )

    def _extract_request_info(self, text: str) -> Optional[str]:
        """
        Extracts contact or request information from restricted data statements.
        e.g. "available upon reasonable request to the corresponding author (j.smith@uni.ac.uk)"
        """
        # Look for email addresses
        email = re.search(r"[\w\.-]+@[\w\.-]+\.\w{2,}", text)
        if email:
            return f"Contact: {email.group(0)}"

        # Look for "corresponding author" language
        if re.search(r"corresponding author", text, re.IGNORECASE):
            return "Available upon request to corresponding author"

        return None
