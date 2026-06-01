"""
entity_resolution — Deterministic Entity Resolution Pipeline (Spec 2).

This package provides a robust, reproducible, multi-strategy entity resolution
system for the Scientific Knowledge Graph microbiome research system.

Public surface:
    models   — All shared Pydantic models
    utils    — validate_canonical_id, normalize_surface_form
"""

from entity_resolution import models, utils

__all__ = ["models", "utils"]
