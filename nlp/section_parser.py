"""
nlp/section_parser.py
----------------------
Parses full-text paper content into logical sections.

WHY SECTION PARSING MATTERS:
  "Methods" text is completely different from "Discussion" text.
  If you run NER on the whole paper as one blob:
    - Methods sections mention methods used → correct
    - Discussion sections mention methods other papers used → false positives
  Parsing sections first lets Layer 3 store and query each section
  independently: "show me papers where 16S rRNA appears in Methods,
  not just in the abstract or discussion."

HOW IT WORKS:
  1. For structured abstracts: PubMed provides labeled sections
     (Background, Methods, Results, Conclusions). We split on these.
  2. For full-text (Europe PMC OA papers): We detect section headers
     by regex — lines that are short, bold-suggestive, and match
     known section names.
  3. For unstructured abstracts: We treat the whole abstract as one
     "abstract" section.

SECTION VOCABULARY (standardized):
  abstract | introduction | background | methods | results |
  discussion | conclusion | data_availability | funding |
  acknowledgements | supplementary | other
"""

import re
from typing import List, Optional, Tuple
from loguru import logger

from nlp.enriched_record import ParsedSection


# ── Section header patterns ───────────────────────────────────────────────────
# Each entry: (regex pattern, normalized section type)
# Patterns are checked against the start of each line (case-insensitive).

SECTION_PATTERNS: List[Tuple[str, str]] = [
    # Abstract sub-sections (structured abstract labels)
    (r"^background[:\s]*$",             "background"),
    (r"^introduction[:\s]*$",           "introduction"),
    (r"^objective[s]?[:\s]*$",          "background"),
    (r"^purpose[:\s]*$",                "background"),

    # Methods
    (r"^method[s]?[:\s]*$",             "methods"),
    (r"^materials?\s+and\s+method[s]?", "methods"),
    (r"^study design[:\s]*$",           "methods"),
    (r"^experimental\s+design",         "methods"),
    (r"^subjects?\s+and\s+method",      "methods"),
    (r"^patients?\s+and\s+method",      "methods"),

    # Results
    (r"^results?[:\s]*$",               "results"),
    (r"^findings[:\s]*$",               "results"),
    (r"^outcomes?[:\s]*$",              "results"),

    # Discussion
    (r"^discussion[:\s]*$",             "discussion"),
    (r"^interpretation[:\s]*$",         "discussion"),

    # Conclusion
    (r"^conclusion[s]?[:\s]*$",         "conclusion"),
    (r"^summary[:\s]*$",                "conclusion"),
    (r"^concluding\s+remark",           "conclusion"),

    # Data availability (critical for our project)
    (r"^data\s+avail",                  "data_availability"),
    (r"^availability\s+of\s+data",      "data_availability"),
    (r"^data\s+and\s+code\s+avail",     "data_availability"),
    (r"^data\s+access",                 "data_availability"),
    (r"^code\s+avail",                  "data_availability"),
    (r"^data\s+sharing",                "data_availability"),
    (r"^supplementary\s+data",          "data_availability"),

    # Funding / acknowledgements
    (r"^funding[:\s]*$",                "funding"),
    (r"^financial\s+support",           "funding"),
    (r"^acknowledg",                    "acknowledgements"),

    # Supplementary
    (r"^supplementary\s+method",        "supplementary"),
    (r"^supplemental\s+information",    "supplementary"),
    (r"^supporting\s+information",      "supplementary"),
]

# Structured abstract label patterns (found inside abstract text itself)
STRUCTURED_ABSTRACT_LABELS = {
    "background":    "background",
    "introduction":  "introduction",
    "objective":     "background",
    "purpose":       "background",
    "methods":       "methods",
    "results":       "results",
    "findings":      "results",
    "conclusions":   "conclusion",
    "conclusion":    "conclusion",
    "significance":  "conclusion",
    "importance":    "conclusion",
}


class SectionParser:
    """
    Splits paper text (abstract or full text) into labeled sections.
    """

    def parse_abstract(self, abstract: Optional[str]) -> List[ParsedSection]:
        """
        Parses an abstract into sections.

        STRUCTURED ABSTRACTS look like:
          "Background: Gut microbiome composition...
           Methods: We recruited 200 participants...
           Results: We found significant differences...
           Conclusions: These findings suggest..."

        UNSTRUCTURED ABSTRACTS are just a single paragraph.
        We return them as one "abstract" section.
        """
        if not abstract or not abstract.strip():
            return []

        # Check if it's a structured abstract by looking for label: patterns
        structured_match = re.search(
            r"(Background|Methods|Results|Conclusions|Objective|Purpose|Findings)[:\s]",
            abstract, re.IGNORECASE
        )

        if structured_match:
            return self._parse_structured_abstract(abstract)
        else:
            return [ParsedSection(
                section_type="abstract",
                header="Abstract",
                content=abstract.strip(),
            )]

    def _parse_structured_abstract(self, abstract: str) -> List[ParsedSection]:
        """
        Splits a structured abstract on its label boundaries.

        Example input:
          "Background: Gut microbiome... Methods: We collected... Results: ..."

        The regex finds all "Label:" patterns and uses them as split points.
        """
        # Pattern: one of our known labels followed by colon (and possibly newline)
        label_pattern = r"(" + "|".join(STRUCTURED_ABSTRACT_LABELS.keys()) + r")[:\s]+"
        flags = re.IGNORECASE

        sections = []
        parts = re.split(f"({label_pattern})", abstract, flags=flags)

        # re.split with a capturing group returns: [before, label, text, label, text, ...]
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue

            # Check if this part is a known label
            label_match = re.match(label_pattern, part, flags=flags)
            if label_match and i + 1 < len(parts):
                # The content follows the label
                label_text = parts[i].strip().rstrip(":")
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                section_type = STRUCTURED_ABSTRACT_LABELS.get(
                    label_text.lower(), "other"
                )
                if content:
                    sections.append(ParsedSection(
                        section_type=section_type,
                        header=label_text,
                        content=content,
                    ))
                i += 2
            else:
                # Unmatched content before first label
                if len(part) > 20:   # Skip very short fragments
                    sections.append(ParsedSection(
                        section_type="abstract",
                        content=part,
                    ))
                i += 1

        return sections if sections else [ParsedSection(
            section_type="abstract", content=abstract.strip()
        )]

    def parse_full_text(self, full_text: Optional[str]) -> List[ParsedSection]:
        """
        Parses a full paper text into sections.
        Used when full text is available (open-access papers from Europe PMC).

        APPROACH:
          Split text into lines. For each line, check if it looks like
          a section header (short line matching our patterns). Everything
          between two headers becomes one section's content.
        """
        if not full_text or not full_text.strip():
            return []

        lines = full_text.split("\n")
        sections = []
        current_type = "other"
        current_header = None
        current_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Check if this line is a section header
            section_type = self._detect_header(line_stripped)

            if section_type:
                # Save the previous section before starting a new one
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append(ParsedSection(
                            section_type=current_type,
                            header=current_header,
                            content=content,
                        ))
                current_type = section_type
                current_header = line_stripped
                current_lines = []
            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(ParsedSection(
                    section_type=current_type,
                    header=current_header,
                    content=content,
                ))

        return sections

    def _detect_header(self, line: str) -> Optional[str]:
        """
        Returns the section type if the line looks like a section header,
        or None if it's regular content.

        A line is a header if:
          1. It's short (< 80 chars) — long lines are paragraph text
          2. It matches one of our section patterns
        """
        if not line or len(line) > 80:
            return None

        for pattern, section_type in SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return section_type

        return None

    def get_section(
        self,
        sections: List[ParsedSection],
        section_type: str,
    ) -> Optional[ParsedSection]:
        """Helper: returns first section of the given type, or None."""
        for s in sections:
            if s.section_type == section_type:
                return s
        return None

    def get_section_text(
        self,
        sections: List[ParsedSection],
        section_type: str,
    ) -> Optional[str]:
        """Helper: returns text of first section of given type, or None."""
        s = self.get_section(sections, section_type)
        return s.content if s else None
