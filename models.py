"""
models.py
---------
The unified data model for a research paper.

WHY A SHARED MODEL?
  PubMed, Europe PMC, Semantic Scholar, and bioRxiv all return data in
  different formats (different field names, different nesting, different
  date formats). Every collector converts its raw response into THIS
  single PaperRecord. That way Layer 2 (NLP) and Layer 4 (storage) never
  need to know where a paper came from.

  Pydantic handles validation automatically — if a required field is missing
  or has the wrong type, you get a clear error immediately rather than a
  mysterious crash later.
"""

from __future__ import annotations
from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    """
    One research paper from any source, in a normalized format.
    All fields marked Optional may not be present in every source.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    # These are the unique identifiers. We use DOI as the primary dedup key
    # because it works across all sources. PMID is PubMed-specific.

    doi:   Optional[str] = None        # e.g. "10.1038/s41586-024-07999-z"
    pmid:  Optional[str] = None        # e.g. "38765432"
    pmcid: Optional[str] = None        # e.g. "PMC11234567"
    arxiv_id: Optional[str] = None     # e.g. "2024.12345"
    source: str = ""                   # "pubmed" | "europepmc" | "semantic_scholar" | "biorxiv"

    # ── Core Metadata ─────────────────────────────────────────────────────────
    title:    str = ""
    abstract: Optional[str] = None
    authors:  List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    # ── Publication Info ───────────────────────────────────────────────────────
    journal:          Optional[str] = None   # Full journal name
    journal_abbrev:   Optional[str] = None   # Abbreviated journal name
    issn:             Optional[str] = None
    publication_date: Optional[str] = None   # ISO format: "2024-03-15"
    publication_year: Optional[int] = None
    volume:           Optional[str] = None
    issue:            Optional[str] = None
    pages:            Optional[str] = None

    # ── Article Type ──────────────────────────────────────────────────────────
    # Raw type string from the source. Layer 3 will standardize this into
    # a controlled vocabulary: original_research | review | meta_analysis | etc.
    article_types: List[str] = Field(default_factory=list)

    # ── Access & Availability ─────────────────────────────────────────────────
    is_open_access:     bool = False
    full_text_url:      Optional[str] = None   # URL to full text if available
    pdf_url:            Optional[str] = None
    full_text:          Optional[str] = None   # Parsed full text (from PMC XML enrichment)

    # ── Citations ─────────────────────────────────────────────────────────────
    citation_count:     Optional[int] = None
    reference_count:    Optional[int] = None

    # ── MeSH / Subject Terms ─────────────────────────────────────────────────
    # PubMed-specific controlled vocabulary terms. Very useful for NLP later.
    mesh_terms: List[str] = Field(default_factory=list)

    # ── Ingestion Metadata ────────────────────────────────────────────────────
    # These fields are used by the scheduler to detect updates.
    content_hash:  Optional[str] = None   # MD5 of title+abstract — changes if paper is corrected
    fetched_at:    Optional[str] = None   # ISO timestamp of when we fetched this
    is_preprint:   bool = False

    class Config:
        # Allow extra fields from sources we haven't mapped yet,
        # instead of crashing with a validation error.
        extra = "allow"

    def get_dedup_key(self) -> str:
        """
        Returns the best available unique key for deduplication.
        Priority: DOI > PMID > title (fallback — not perfect but better than nothing).
        """
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        return f"title:{self.title.lower()[:80]}"
