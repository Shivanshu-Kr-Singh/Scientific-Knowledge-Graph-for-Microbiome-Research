"""
nlp/evidence_extractor.py
--------------------------
Extracts quantitative evidence signals from paper text.

FIX: "Largest number" strategy was wrong for meta-analyses.

OLD BEHAVIOUR:
  best_size = max of all pattern matches across entire text.
  A systematic review: "we identified 12,450 records, screened 3,200
  abstracts, and included 45 studies with 8,920 participants"
  → returned 12,450 (database records, not participants).

NEW BEHAVIOUR — priority hierarchy:
  Priority 1: n=X notation (most reliable, author-stated)
  Priority 2: "recruited/enrolled/randomized N" (verb patterns)
  Priority 3: "N patients/participants/subjects" (noun patterns)
  Priority 4: "total of N", "cohort of N" (preamble patterns)
  Priority 5: "N healthy/obese/..." (adjective patterns)
  Priority 6: "samples from N" (proxy patterns — lowest priority)

  Search exclusion: numbers preceded by database-search language
  ("identified N records", "screened N", "searched N databases")
  are explicitly excluded — these are search counts, not participants.

  For meta-analyses (detected by forest_plot/I² patterns), the
  "participants" noun pattern is preferred over the largest number.
"""

import re
from typing import Optional, Tuple


# ── Exclusion patterns — numbers that are NOT participant counts ──────────────
# These appear in systematic reviews describing the search process.
_EXCLUDE_PATTERNS = re.compile(
    r"\b(?:identified|screened|retrieved|searched|found|yielded|"
    r"abstracts?|titles?|records?|articles?|citations?|studies\s+identified|"
    r"databases?|references?|included\s+studies|eligible\s+studies|"
    r"selected\s+studies)\s+(?:\w+\s+){0,3}(\d{2,6})\b|"
    r"\b(\d{2,6})\s+(?:records?|abstracts?|titles?|citations?|"
    r"studies|trials|articles?\s+identified|"
    r"studies\s+were\s+(?:identified|included|eligible)|"
    r"references?)\b",
    re.IGNORECASE,
)

# ── Priority-ordered sample size patterns ─────────────────────────────────────
# Each entry: (priority, pattern) — lower priority = more reliable
# Priority 1 is tried first; if it matches, higher priorities are skipped.

_PRIORITY_PATTERNS: list = [
    # Priority 1: n=X notation (most reliable — author explicitly states N)
    (1, re.compile(r"[\(\[]\s*[nN]\s*=\s*(\d{2,6})\s*[\)\]]", re.IGNORECASE)),
    (1, re.compile(r"\bn\s*=\s*(\d{2,6})\b", re.IGNORECASE)),
    (1, re.compile(r"\bN\s*=\s*(\d{2,6})\b")),

    # Priority 2: verb + number (author action verb implies recruitment)
    # Note: "included" is excluded here to avoid "included 45 studies"
    # clashing with the exclusion set. Use "enrolled/recruited/randomized" instead.
    (2, re.compile(
        r"\b(?:recruited|enrolled|randomized|randomised|"
        r"assigned|registered|screened\s+and\s+enrolled)\s+(\d{2,6})\b",
        re.IGNORECASE,
    )),

    # Priority 3: number + subject noun (direct cohort description)
    (3, re.compile(
        r"\b(\d{2,6})\s+(?:patients?|participants?|subjects?|volunteers?|"
        r"individuals?|adults?|children|infants?|neonates?|women|men|"
        r"cases?|controls?|donors?)\b",
        re.IGNORECASE,
    )),

    # Priority 4: preamble phrases
    (4, re.compile(
        r"\b(?:a total of|total of|comprising|consisting of|including)\s+(\d{2,6})\s+"
        r"(?:patients?|participants?|subjects?|volunteers?|individuals?|adults?)\b",
        re.IGNORECASE,
    )),
    (4, re.compile(
        r"\b(?:cohort|study population|sample)\s+of\s+(\d{2,6})\b",
        re.IGNORECASE,
    )),

    # Priority 5: adjective noun combos
    (5, re.compile(
        r"\b(\d{2,6})\s+(?:healthy|obese|overweight|diabetic|"
        r"ibd|uc|cd|treatment[-\s]naive|antibiotic[-\s]naive|"
        r"age[- ]matched|sex[- ]matched)\b",
        re.IGNORECASE,
    )),

    # Priority 6: non-English (lower priority — may be less precise)
    (6, re.compile(
        r"\b(\d{2,6})\s+(?:patienten|probanden|teilnehmer|personen|"
        r"freiwilligen|sujets?|volontaires?|pacientes?|participantes?|"
        r"sujetos?)\b",
        re.IGNORECASE,
    )),
    (6, re.compile(r"\b(\d{2,6})\s*例\b", re.IGNORECASE)),
    (6, re.compile(r"\bau total\s+(\d{2,6})\b", re.IGNORECASE)),
    (6, re.compile(r"\binsgesamt\s+(\d{2,6})\b", re.IGNORECASE)),

    # Priority 7: proxy patterns (sample count — least reliable)
    (7, re.compile(
        r"\b(?:stool|fecal|faecal|feces|faeces|biopsy|serum|blood)\s+"
        r"samples?\s+(?:from|of)\s+(\d{2,6})\b",
        re.IGNORECASE,
    )),
    (7, re.compile(r"\b(\d{2,6})\s+(?:persons?|individuals?|people)\b",
                   re.IGNORECASE)),
]

# Compile the combined sample regex for the old-style "all matches" scan
_ALL_SAMPLE_REGEX = re.compile(
    "|".join(f"(?:{p.pattern})" for _, p in _PRIORITY_PATTERNS),
    re.IGNORECASE,
)

# ── Sequencing method patterns ────────────────────────────────────────────────
_SEQUENCING_METHODS = {
    "16S rRNA sequencing":     re.compile(r"\b16s\s*r?rna\b",              re.I),
    "shotgun metagenomics":    re.compile(r"\bshotgun\s+metagenomic",       re.I),
    "whole metagenome seq":    re.compile(r"\bwhole[\s-]metagenome\s+seq",  re.I),
    "metatranscriptomics":     re.compile(r"\bmetatranscriptom",            re.I),
    "whole genome sequencing": re.compile(r"\bwhole[\s-]genome\s+seq|\bwgs\b", re.I),
    "amplicon sequencing":     re.compile(r"\bamplicon\s+sequencing\b",     re.I),
    "metaproteomics":          re.compile(r"\bmetaproteom",                 re.I),
    "metabolomics":            re.compile(r"\bmetabolomic",                 re.I),
    "RNA-seq":                 re.compile(r"\brna[\s-]?seq\b",              re.I),
    "nanopore":                re.compile(r"\bnanopore\b|\boxford\s+nanopore\b", re.I),
    "PacBio":                  re.compile(r"\bpacbio\b|\bpacific\s+biosciences\b", re.I),
    "ITS sequencing":          re.compile(r"\bits\s*[12]?\s+sequencing\b",  re.I),
}

# ── Dataset / accession patterns ──────────────────────────────────────────────
_DATASET_PATTERN = re.compile(
    r"\b(?:PRJNA|PRJEB|PRJDB|SRP|SRR|SRX|SRS|ERR|ERX|ERS|ERP|"
    r"DRP|DRR|GSE|GSM|E-MTAB-|E-GEOD-|SAMN|SAME|MGP)\d+\b",
    re.IGNORECASE,
)

# ── Study design confidence signals ──────────────────────────────────────────
_HIGH_CONFIDENCE = re.compile(
    r"\b(?:randomized\s+controlled|randomised\s+controlled|"
    r"double[- ]blind|placebo[- ]controlled|"
    r"prospective\s+(?:longitudinal\s+)?cohort)\b",
    re.IGNORECASE,
)
_MED_CONFIDENCE = re.compile(
    r"\b(?:cohort\s+study|cross[- ]sectional|case[- ]control|"
    r"observational|retrospective\s+cohort|"
    r"prospective\s+study)\b",
    re.IGNORECASE,
)

# Meta-analysis detection (for choosing the right sample size strategy)
_IS_META = re.compile(
    r"\bforest\s+plot\b|\bI[²2]\s*(?:heterogeneity|statistic)\b|"
    r"\bpooled\s+(?:odds\s+ratio|risk\s+ratio|effect\s+size)\b|"
    r"\brandom[- ]effects\s+model\b",
    re.IGNORECASE,
)


def extract(text: str) -> dict:
    """
    Extracts evidence signals from paper text.

    Returns:
        {
          "sample_size":        int
          "sample_size_raw":    str   — matched text snippet for audit
          "sample_size_priority": int — 1=most reliable, 7=least reliable
          "sequencing_methods": list
          "datasets":           list
          "study_confidence":   float
          "n_datasets":         int
          "is_meta_analysis":   bool
        }
    """
    text = text or ""
    out: dict = {
        "sample_size":          0,
        "sample_size_raw":      "",
        "sample_size_priority": 99,
        "sequencing_methods":   [],
        "datasets":             [],
        "study_confidence":     0.0,
        "n_datasets":           0,
        "is_meta_analysis":     False,
    }

    # ── Meta-analysis detection ───────────────────────────────────────────────
    is_meta = bool(_IS_META.search(text))
    out["is_meta_analysis"] = is_meta

    # Build exclusion set — positions of numbers that are search counts
    excluded_positions: set = set()
    for m in _EXCLUDE_PATTERNS.finditer(text):
        for grp in m.groups():
            if grp is not None:
                # Mark the position in text to exclude this number
                start = text.find(grp, max(0, m.start() - 5))
                if start >= 0:
                    excluded_positions.add(start)

    # ── Sample size — priority-based ──────────────────────────────────────────
    best_size     = 0
    best_match    = ""
    best_priority = 99

    for priority, pattern in _PRIORITY_PATTERNS:
        # For meta-analyses, skip priority-7 proxy patterns entirely
        if is_meta and priority >= 7:
            continue
        # Already have a high-priority match — don't downgrade it
        if best_priority < priority and best_size > 0:
            continue

        for m in pattern.finditer(text):
            # Find the first non-None capturing group
            for grp in m.groups():
                if grp is None:
                    continue
                try:
                    n = int(grp.replace(",", ""))
                except ValueError:
                    continue

                if n < 2 or n > 500_000:
                    continue

                # Skip if this position was in the exclusion set
                pos = text.find(grp, max(0, m.start() - 5))
                if pos in excluded_positions:
                    continue

                # Take this match if it's higher priority or same priority + larger
                if priority < best_priority or (priority == best_priority and n > best_size):
                    best_size     = n
                    best_match    = m.group(0)
                    best_priority = priority
                break

    out["sample_size"]          = best_size
    out["sample_size_raw"]      = best_match
    out["sample_size_priority"] = best_priority

    # ── Sequencing methods ─────────────────────────────────────────────────────
    out["sequencing_methods"] = [
        name for name, rx in _SEQUENCING_METHODS.items()
        if rx.search(text)
    ]

    # ── Datasets ───────────────────────────────────────────────────────────────
    datasets = list(dict.fromkeys(
        m.group(0).upper() for m in _DATASET_PATTERN.finditer(text)
    ))
    out["datasets"]   = datasets
    out["n_datasets"] = len(datasets)

    # ── Study design confidence ────────────────────────────────────────────────
    # Require the signal to appear in first 2000 chars (Methods section context)
    # to avoid narrative reviews triggering on descriptions of other studies
    context = text[:2000]
    if _HIGH_CONFIDENCE.search(context):
        out["study_confidence"] = 0.9
    elif _MED_CONFIDENCE.search(context):
        out["study_confidence"] = 0.5

    return out
