"""
collectors/metadata_filter.py
-------------------------------
Stage 1 — PubMed MeSH metadata filter.

Standalone module. Import and use independently from the main pipeline
or let relevance_filter.py call it automatically.

WHAT IT DOES:
  PubMed attaches human-curated MeSH (Medical Subject Headings) to every
  indexed paper. These are assigned by NCBI librarians — not extracted by
  algorithms. They are the single most reliable signal we have.

  This stage makes confident KEEP/REJECT decisions only when MeSH is available.
  For papers without MeSH (EuropePMC, Semantic Scholar, bioRxiv) it returns
  UNKNOWN so Stage 2 rules handle them.

DECISIONS:
  human MeSH + microbiome MeSH        → KEEP  (score 0.90)
  human MeSH + microbiome + animal    → KEEP  (score 0.70, human wins)
  animal MeSH only, no human          → REJECT (score 0.05)
  microbiome MeSH, no human/animal    → UNKNOWN (score 0.45, pass to Stage 2)
  no microbiome MeSH at all           → UNKNOWN (score 0.10, pass to Stage 2)
  no MeSH terms                       → UNKNOWN (score 0.0, pass to Stage 2)
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import yaml
from loguru import logger

CONFIG_PATH = Path(__file__).parent.parent / "config" / "organisms.yaml"


@dataclass
class MetadataVerdict:
    """Result from Stage 1."""
    decision:  str      # "KEEP" | "REJECT" | "UNKNOWN"
    score:     float    # 0.0–1.0 confidence
    reason:    str
    stage:     str = "stage1_metadata"


class MetadataFilter:
    """
    Stage 1 filter. Uses PubMed MeSH terms for immediate high-confidence decisions.
    Returns UNKNOWN for papers without MeSH — passes to Stage 2.
    """

    def __init__(self):
        try:
            cfg = yaml.safe_load(open(CONFIG_PATH))
            self.mesh_keep   = [m.lower() for m in cfg.get("mesh_keep", [])]
            self.mesh_human  = [m.lower() for m in cfg.get("mesh_human_signal", [])]
            self.mesh_animal = [m.lower() for m in cfg.get("mesh_animal_only", [])]
        except Exception as e:
            logger.warning(f"[metadata_filter] Could not load config: {e} — using defaults")
            self.mesh_keep   = ["gastrointestinal microbiome", "microbiota", "metagenomics",
                                "fecal microbiota transplantation", "dysbiosis"]
            self.mesh_human  = ["humans", "adult", "aged", "child", "infant", "adolescent"]
            self.mesh_animal = ["animals", "mice", "rats", "zebrafish", "swine", "cattle"]

        logger.debug(f"[metadata_filter] {len(self.mesh_keep)} microbiome terms, "
                     f"{len(self.mesh_human)} human, {len(self.mesh_animal)} animal")

    def evaluate(self, paper) -> MetadataVerdict:
        """
        Evaluates one paper using its MeSH terms.
        paper: any object with .mesh_terms (list) and .source (str) attributes.
        """
        mesh = [m.lower() for m in (getattr(paper, "mesh_terms", None) or [])]

        # No MeSH — can't decide here
        if not mesh:
            return MetadataVerdict("UNKNOWN", 0.0, "no_mesh_available")

        has_microbiome = any(m in mesh for m in self.mesh_keep)
        has_human      = any(m in mesh for m in self.mesh_human)
        has_animal     = any(m in mesh for m in self.mesh_animal)

        # Animal only (no human tag) — reject confidently
        if has_animal and not has_human:
            matched_animal = next((m for m in mesh if m in self.mesh_animal), "")
            return MetadataVerdict("REJECT", 0.05, f"animal_only_mesh:{matched_animal}")

        # No microbiome MeSH — not a microbiome paper, but not confident enough to reject
        # Could be a relevant paper with no microbiome-specific MeSH tag
        if not has_microbiome:
            return MetadataVerdict("UNKNOWN", 0.10, "no_microbiome_mesh_pass_to_rules")

        # Human + microbiome — confident keep
        if has_human and not has_animal:
            matched_human = next((m for m in mesh if m in self.mesh_human), "")
            matched_micro = next((m for m in mesh if m in self.mesh_keep), "")
            return MetadataVerdict(
                "KEEP", 0.90,
                f"human+microbiome_mesh:{matched_human},{matched_micro}"
            )

        # Human + microbiome + animal (e.g. comparative study with human cohort)
        if has_human and has_animal and has_microbiome:
            return MetadataVerdict("KEEP", 0.70,
                                   "human+microbiome+animal_mesh:human_wins")

        # Microbiome MeSH but no human/animal signal — borderline
        matched_micro = next((m for m in mesh if m in self.mesh_keep), "")
        return MetadataVerdict("UNKNOWN", 0.45,
                               f"microbiome_mesh_no_human_signal:{matched_micro}")

    def _first_match(self, mesh: List[str], targets: List[str]) -> Optional[str]:
        for m in mesh:
            if m in targets:
                return m
        return None
