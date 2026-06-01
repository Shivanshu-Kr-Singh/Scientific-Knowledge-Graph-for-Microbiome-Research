"""
Shared Pydantic models for the Deterministic Entity Resolution Pipeline.

All models are defined here and imported by other components.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """Supported entity types in the resolution pipeline."""

    TAXON = "taxon"
    DISEASE = "disease"
    METHOD = "method"


class SynonymProvenance(str, Enum):
    """Source of a synonym / surface-form registration."""

    ONTOLOGY = "ontology"      # Sourced from NCBI Taxonomy / MeSH
    PAPER_TEXT = "paper_text"  # Extracted from paper text
    CURATOR = "curator"        # Added manually by a curator


# ---------------------------------------------------------------------------
# Core resolution models
# ---------------------------------------------------------------------------


class NormalizationResult(BaseModel):
    """
    Drop-in replacement for the Spec 1 NormalizationResult interface.

    Requirements: 14.1
    """

    canonical_id: Optional[str] = None  # None if unresolved
    grounded: bool


class CandidateScore(BaseModel):
    """
    A single candidate produced by one resolution strategy, with scores.

    Requirements: 4.1, 4.4
    """

    canonical_id: str
    strategy: str
    grounding_confidence: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)


class ResolutionResult(BaseModel):
    """
    Full resolution output including audit fields.

    Requirements: 1.1, 2.1, 7.2
    """

    surface_form: str
    entity_type: str  # taxon | disease | method
    canonical_id: Optional[str] = None
    grounded: bool
    winning_strategy: str  # manual_override | exact | normalized | abbreviation | synonym | fuzzy | ontology | none
    grounding_confidence: float = Field(ge=0.0, le=1.0)
    conflict_set: List[CandidateScore] = Field(default_factory=list)
    paper_id: str
    timestamp: datetime
    high_conflict: bool = False       # True when 3+ strategies produced candidates
    hierarchy_match: bool = False     # True when ontology traversal was the winner
    hierarchy_level: Optional[int] = None  # 1, 2, or 3 when hierarchy_match=True


# ---------------------------------------------------------------------------
# Registry models
# ---------------------------------------------------------------------------


class SynonymRecord(BaseModel):
    """
    A single synonym / surface-form entry for a canonical entity.

    Requirements: 5.1, 5.3
    """

    surface_form: str = Field(max_length=500)  # NFC-normalized, max 500 chars
    provenance: SynonymProvenance
    added_by: Optional[str] = None  # curator_id if provenance=CURATOR
    added_at: datetime


class CanonicalEntityRecord(BaseModel):
    """
    Authoritative record for a canonical entity in the registry.

    Requirements: 3.1
    """

    canonical_id: str          # NCBI Taxonomy int (as str), MeSH "D######", or "METHOD-xxx"
    primary_name: str
    entity_type: EntityType
    ontology_source: str       # "ncbi_taxonomy" | "mesh" | "internal"
    synonyms: List[SynonymRecord] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Unresolved / conflict models
# ---------------------------------------------------------------------------


class RegistrationError(BaseModel):
    """
    Error returned when a canonical entity registration fails validation.

    Requirements: 3.2, 3.3, 3.4
    """

    field: str
    message: str


class UnresolvedEntity(BaseModel):
    """
    A surface form that failed all resolution strategies.

    Requirements: 1.3, 14.4
    """

    surface_form: str
    entity_type: str
    paper_id: str
    timestamp: datetime
    local_id: Optional[str] = None  # Temporary local ID assigned pending curator review


class ShadowModeDiscrepancy(BaseModel):
    """
    Logged when Spec 1 and Spec 2 normalizers produce different results.

    Requirements: 14.5, 14.6
    """

    surface_form: str
    entity_type: str
    paper_id: str
    spec1_canonical_id: Optional[str] = None
    spec1_grounded: bool
    spec2_canonical_id: Optional[str] = None
    spec2_grounded: bool
    timestamp: datetime


class SynonymConflictRecord(BaseModel):
    """
    Logged when a surface form is registered for two different canonical entities.

    Requirements: 3.7, 5.4
    """

    duplicate_surface_form: str
    entity_a_id: str
    entity_b_id: str
    timestamp: datetime
    provenance_source: Optional[str] = None


class SynonymIndexEntry(BaseModel):
    """
    A single entry returned by SynonymIndex.prefix_lookup().

    Requirements: 5.5
    """

    surface_form_normalized: str  # NFC + lowercase
    canonical_id: str
    entity_type: EntityType


# ---------------------------------------------------------------------------
# Override models
# ---------------------------------------------------------------------------


class ManualOverride(BaseModel):
    """
    A curator-defined mapping pinning a surface form to a canonical ID.

    Requirements: 9.3
    """

    surface_form: str
    canonical_id: str
    entity_type: str
    curator_id: str
    timestamp: datetime
    justification: Optional[str] = Field(default=None, max_length=500)


class BulkImportResult(BaseModel):
    """
    Result of a bulk CSV import of manual overrides.

    Requirements: 9.7, 9.8
    """

    total_rows: int
    imported_count: int
    skipped_count: int
    skipped_rows: List[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Audit models
# ---------------------------------------------------------------------------


class ResolutionRecord(BaseModel):
    """
    Audit log entry for a single resolution attempt.

    Requirements: 7.1, 7.2, 7.3
    """

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # UUID
    surface_form: str
    entity_type: str
    timestamp: datetime  # UTC ISO-8601
    winning_strategy: str  # strategy name or "none" if unresolved
    canonical_id: Optional[str] = None  # None if unresolved
    grounding_confidence: float = Field(ge=0.0, le=1.0)
    conflict_set: List[CandidateScore] = Field(default_factory=list)
    paper_id: str
    high_conflict: bool
    hierarchy_match: bool
    hierarchy_level: Optional[int] = None
    curator_override: Optional[str] = None  # curator_id if winning_strategy="manual_override"


class AuditQuery(BaseModel):
    """
    Query parameters for filtering resolution audit records.

    Requirements: 7.6
    """

    surface_form: Optional[str] = None
    canonical_id: Optional[str] = None
    winning_strategy: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    paper_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Cache models
# ---------------------------------------------------------------------------


class CacheEntry(BaseModel):
    """
    A single entry in the resolution cache.

    Requirements: 8.4
    """

    surface_form: str
    resolution_result: ResolutionResult
    cache_timestamp: datetime
    registry_version: int  # Version of CanonicalRegistry at time of caching


# ---------------------------------------------------------------------------
# Metrics models
# ---------------------------------------------------------------------------


class EntityTypeMetrics(BaseModel):
    """
    Per-entity-type resolution metrics for a single pipeline run.

    Requirements: 10.2
    """

    entity_type: str
    resolution_rate: float
    avg_grounding_confidence: float
    unresolved_count: int


class RunMetricsSnapshot(BaseModel):
    """
    Aggregated metrics snapshot for a completed pipeline run.

    Requirements: 10.1, 10.3
    """

    run_id: str
    timestamp: datetime
    paper_ids: List[str] = Field(default_factory=list)
    total_forms: int
    resolved_count: int
    unresolved_count: int
    resolution_rate: float  # resolved / total; 0.0 when total == 0
    per_strategy_counts: dict = Field(default_factory=dict)
    entity_type_metrics: List[EntityTypeMetrics] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Merger models
# ---------------------------------------------------------------------------


class MergeLogEntry(BaseModel):
    """
    Audit record for a successful entity merge operation.

    Requirements: 6.4
    """

    source_node_ids: List[str]
    target_canonical_id: str
    triggering_resolution: str  # surface_form that triggered the merge
    timestamp: datetime
    relationships_transferred: int
    relationships_deduplicated: int


class MergeRollbackEntry(BaseModel):
    """
    Audit record for a failed (rolled-back) entity merge operation.

    Requirements: 6.7
    """

    source_node_ids: List[str]
    target_canonical_id: str
    failed_step: str  # "relationship_transfer" | "node_deletion" | "audit_log_write"
    error_message: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Abbreviation expansion models
# ---------------------------------------------------------------------------


class ExpansionCandidate(BaseModel):
    """
    A single candidate expansion produced by the AbbreviationExpander.

    Confidence rules:
    - Unambiguous (1 candidate): confidence = 1.0
    - Ambiguous (N candidates):  confidence = 1.0 / N for each candidate

    Requirements: 11.2, 11.4
    """

    expanded_form: str
    confidence: float = Field(ge=0.0, le=1.0)  # 1.0/N for N candidates
