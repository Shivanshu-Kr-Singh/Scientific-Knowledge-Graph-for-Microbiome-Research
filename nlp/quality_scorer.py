"""
nlp/quality_scorer.py
-----------------------
Scores paper quality on a 0.0–1.0 scale using evidence signals.

SCORING COMPONENTS (max 1.0 total):
  Journal quartile      0.00–0.30  (Q1=0.30, Q2=0.20, Q3=0.10, Q4=0.05)
  Study design          0.00–0.30  (RCT=0.30, cohort=0.20, cross-sec=0.10)
  Sample size           0.00–0.20  (≥1000=0.20, ≥500=0.15, ≥100=0.10, ≥10=0.05)
  Data availability     0.00–0.15  (accession_linked=0.15, open=0.10, restricted=0.05)
  Sequencing method     0.00–0.05  (shotgun/WGS=0.05, 16S=0.03, other=0.01)

WHY THESE WEIGHTS:
  Data availability was previously just a truthy check.
  Accession-linked papers (actual SRA/GEO IDs) are reproducible —
  they deserve a higher score than "data available on request".

  Study design now uses study_confidence from evidence_extractor
  rather than a hardcoded type string, so it works across all sources.

USAGE:
  score(paper_dict) → float in [0.0, 1.0]
  paper_dict keys: journal_info, article_type, data_availability,
                   study_design, sample_size, sequencing_methods
"""


def score(paper: dict) -> float:
    s = 0.0

    # ── Journal quartile (0–0.30) ──────────────────────────────────────────────
    journal  = paper.get("journal_info")
    quartile = None
    if journal:
        quartile = getattr(journal, "quartile", None)

    if quartile == "Q1":
        s += 0.30
    elif quartile == "Q2":
        s += 0.20
    elif quartile == "Q3":
        s += 0.10
    elif quartile == "Q4":
        s += 0.05

    # ── Study design (0–0.30) ──────────────────────────────────────────────────
    # study_design.py now returns calibrated confidence (0.30–0.95).
    # Scale the design score proportionally so a double-blind RCT (0.95)
    # gets 0.30 and a weak pilot signal (0.30) gets ~0.09.
    study      = paper.get("study_design") or {}
    study_type = study.get("type", "")
    study_conf = study.get("confidence", 0.0)
    article_type = (paper.get("article_type") or "").lower()

    # Article-type overrides — use fixed scores for well-defined types
    if "meta_analysis" in article_type:
        # Meta-analyses pool evidence but don't generate primary data
        s += 0.22
    elif "systematic_review" in article_type:
        s += 0.20
    elif "protocol" in article_type or study_type == "protocol":
        # Protocol = planned study, no results yet — cap at 0.05
        s += 0.05
    elif "case_report" in article_type:
        s += 0.05
    elif "letter" in article_type or "commentary" in article_type:
        s += 0.0
    elif study_conf > 0:
        # Scale calibrated confidence linearly: 0.95 → 0.285, 0.30 → 0.09
        design_score = round(study_conf * 0.30, 3)
        # Cap meta-analysis/systematic_review types at 0.25
        if study_type in ("meta_analysis", "systematic_review"):
            design_score = min(design_score, 0.25)
        s += design_score
    elif "original_research" in article_type:
        s += 0.10

    # ── Sample size (0–0.20) ──────────────────────────────────────────────────
    sample = paper.get("sample_size", 0) or 0
    if sample >= 1000:
        s += 0.20
    elif sample >= 500:
        s += 0.15
    elif sample >= 100:
        s += 0.10
    elif sample >= 10:
        s += 0.05

    # ── Data availability (0–0.15) ────────────────────────────────────────────
    # Granular scoring: accession ID > open URL > restricted > not_stated
    da     = paper.get("data_availability")
    status = None
    if da:
        status = getattr(da, "status", None) or da.get("status") if isinstance(da, dict) else getattr(da, "status", None)

    if status == "accession_linked":
        s += 0.15   # reproducible — has SRA/GEO/ENA accession number
    elif status == "open":
        s += 0.10   # data available but no structured accession
    elif status == "restricted":
        s += 0.05   # data exists but behind access control

    # ── Sequencing method depth (0–0.05) ──────────────────────────────────────
    # Shotgun/WGS gives more information than 16S alone
    methods = paper.get("sequencing_methods", []) or []
    if isinstance(methods, list):
        methods_lower = [m.lower() for m in methods]
        if any(k in " ".join(methods_lower) for k in
               ("shotgun", "whole metagenome", "whole genome", "wgs", "metatranscriptom")):
            s += 0.05
        elif any("16s" in m or "amplicon" in m for m in methods_lower):
            s += 0.03
        elif methods:
            s += 0.01

    return round(min(s, 1.0), 3)
