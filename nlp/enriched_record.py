"""
nlp/enriched_record.py
-----------------------
The output schema for Layer 2.

After the NLP pipeline processes a PaperRecord, it produces an
EnrichedPaperRecord — the same base fields PLUS all the structured
information extracted by the 5 NLP modules.

This is what Layer 3 (knowledge graph) reads and stores.
"""

from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from models import PaperRecord
from typing import Optional
from typing import Any
from pydantic import Field
# ── Article type vocabulary ────────────────────────────────────────────────────
# These are the standardized labels the article classifier outputs.
# No matter what the raw source says ("research-article", "Journal Article",
# "Original Investigation"), it gets normalized to one of these.

ARTICLE_TYPES = [
    "original_research",   # Has primary data, methods, results
    "systematic_review",   # Structured review with PRISMA-style inclusion criteria
    "meta_analysis",       # Pools statistics across studies (has I², pooled estimates)
    "narrative_review",    # Broad overview without systematic search protocol
    "case_report",         # Single patient or small case series
    "letter",              # Short correspondence or rapid communication
    "commentary",          # Opinion or editorial
    "protocol",            # Study protocol (pre-registered methods)
    "dataset",             # Data paper (describes a released dataset)
    "unknown",             # Could not be classified confidently
]


# ── Journal quartiles ──────────────────────────────────────────────────────────
QUARTILES = ["Q1", "Q2", "Q3", "Q4", "unknown"]


# ── Data availability statuses ────────────────────────────────────────────────
DATA_AVAILABILITY_STATUSES = [
    "open",              # Data freely available (link or accession provided)
    "restricted",        # Data available on request or after approval
    "not_stated",        # No data availability section found
    "accession_linked",  # Specific accession number found (SRA, GEO, etc.)
]


class NamedEntity(BaseModel):
    """One named entity extracted by the NER module."""
    text:        str            # The actual text span, e.g. "Bacteroides fragilis"
    label:       str            # Entity type: taxon | disease | method | body_site | treatment | dataset
    start:       Optional[int] = None   # Character offset in source text
    end:         Optional[int] = None
    confidence:  Optional[float] = None  # Model confidence score 0-1

    # ── Grounding fields (populated by NLPPipeline after extraction) ──────────
    # These make the enriched record self-contained — no need to re-normalize in Layer 3
    canonical_name:       Optional[str]   = None  # e.g. "Helicobacter pylori"
    ontology_id:          Optional[str]   = None  # e.g. "ncbi:210", "mesh:D015212", "chebi:17968"
    ontology_name:        Optional[str]   = None  # e.g. "NCBI Taxonomy", "MeSH", "ChEBI"
    grounded:             bool            = False  # True if authoritative ontology ID found
    grounding_confidence: Optional[float] = None  # 1.0=authoritative, 0.8=fuzzy, 0.6=LLM, 0.0=none
    grounding_source:     Optional[str]   = None  # "ncbi" | "ols" | "uniprot" | "llm" | "none"


class ParsedSection(BaseModel):
    """One logical section of the paper's full text."""
    section_type:  str    # abstract | introduction | methods | results | discussion | data_availability | other
    header:        Optional[str] = None   # Original header text as it appears in the paper
    content:       str    # Full text of the section


class DataAvailabilityInfo(BaseModel):
    """Structured information about where the paper's data can be found."""
    status:          str = "not_stated"   # From DATA_AVAILABILITY_STATUSES
    raw_text:        Optional[str] = None  # The full data availability section text
    accession_numbers: List[str] = Field(default_factory=list)
    # Accession IDs found: e.g. ["PRJNA123456", "GSE98765"]

    repositories:    List[str] = Field(default_factory=list)
    # Repository names found: e.g. ["NCBI SRA", "GEO", "Zenodo"]

    urls:            List[str] = Field(default_factory=list)
    # Direct URLs to data: e.g. ["https://zenodo.org/record/1234567"]

    notes:           Optional[str] = None
    # Any extra details, e.g. "available upon reasonable request to corresponding author"


class JournalInfo(BaseModel):
    """Enriched journal metadata from external classification."""
    name:            Optional[str] = None
    issn:            Optional[str] = None
    impact_factor:   Optional[float] = None
    quartile:        str = "unknown"     # Q1 | Q2 | Q3 | Q4 | unknown
    field:           Optional[str] = None   # e.g. "Microbiology", "Gastroenterology"
    is_predatory:    bool = False           # Flagged by Beall's list or similar
    is_open_access:  bool = False


class EnrichedPaperRecord(PaperRecord):
    """
    A PaperRecord with all Layer 2 NLP annotations attached.
    Inherits every field from PaperRecord (doi, title, abstract, authors, etc.)
    and adds the 5 NLP module outputs below.
    """
    # ── Module 1: Article classifier output ───────────────────────────────────
    article_type_normalized: str = "unknown"
    # One value from ARTICLE_TYPES

    article_type_confidence: Optional[float] = None
    # How confident the classifier is (0.0 – 1.0)

    # ── Module 2: Journal classifier output ───────────────────────────────────
    journal_info: Optional[JournalInfo] = None

    # ── Module 3: NER output ──────────────────────────────────────────────────
    entities: List[NamedEntity] = Field(default_factory=list)
    # All entities found across title + abstract (and full text if available)
    # Convenience grouped views (populated by pipeline from entities list)
    taxa:       List[str] = Field(default_factory=list)   # e.g. ["Lactobacillus", "E. coli"]
    diseases:   List[str] = Field(default_factory=list)   # e.g. ["IBD", "Crohn's disease"]
    methods:    List[str] = Field(default_factory=list)   # e.g. ["16S rRNA", "shotgun metagenomics"]
    body_sites: List[str] = Field(default_factory=list)   # e.g. ["gut", "oral cavity"]
    treatments: List[str] = Field(default_factory=list)   # e.g. ["probiotics", "FMT"]
    datasets: list = Field(default_factory=list)
    # ── New entity group fields (12 additional categories) ────────────────────
    metabolites:           List[str] = Field(default_factory=list)   # SCFAs, bile acids, LPS, etc.
    genes:                 List[str] = Field(default_factory=list)   # TLR4, NOD2, IL-6, etc.
    proteins:              List[str] = Field(default_factory=list)   # zonulin, calprotectin, etc.
    biomarkers:            List[str] = Field(default_factory=list)   # CRP, Shannon index, etc.
    pathways:              List[str] = Field(default_factory=list)   # NF-κB, butyrate metabolism, etc.
    populations:           List[str] = Field(default_factory=list)   # healthy adults, IBD patients, etc.
    dietary_components:    List[str] = Field(default_factory=list)   # dietary fiber, inulin, etc.
    immune_cells:          List[str] = Field(default_factory=list)   # Treg, Th17, macrophages, etc.
    clinical_outcomes:     List[str] = Field(default_factory=list)   # remission, dysbiosis, etc.
    environmental_factors: List[str] = Field(default_factory=list)   # antibiotic exposure, birth mode, etc.
    sequencing_platforms:  List[str] = Field(default_factory=list)   # Illumina MiSeq, PacBio, etc.
    omics_features:        List[str] = Field(default_factory=list)   # OTU, ASV, MAG, KEGG, etc.

    # ── Open-world entity store ───────────────────────────────────────────────
    # Entities discovered by BioBERT or LLM that don't fit the 18 known categories
    # are stored here instead of being silently dropped.
    # Key = entity type string (e.g. "biological_process", "receptor", "therapeutic")
    # Value = list of entity name strings
    # This enables open-world entity discovery without schema changes.
    other_entities: Dict[str, List[str]] = Field(default_factory=dict)
    # ── Module 4: Section parser output ───────────────────────────────────────
    sections: List[ParsedSection] = Field(default_factory=list)

    # ── Module 5: Data availability output ────────────────────────────────────
    data_availability: Optional[DataAvailabilityInfo] = None
    full_text: str | None = None
    fetch_source: str | None = None
    fetch_status: str | None = None

    study_design: Optional[dict] = None

    evidence_score: float = 0
    quality_score: float = 0

    filter_stage: str | None = None
    filter_reason: str | None = None

    llm_verified: bool = False
    hard_example: bool = False
    # ── Pipeline metadata ─────────────────────────────────────────────────────
    nlp_processed_at: Optional[str] = None   # ISO timestamp
    nlp_version:      str = "1.0"            # Bump when pipeline logic changes

