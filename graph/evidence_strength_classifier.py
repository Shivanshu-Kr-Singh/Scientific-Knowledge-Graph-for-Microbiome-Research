"""
graph/evidence_strength_classifier.py
--------------------------------------
Classifies evidence strength for LLM-extracted triples based on
extraction confidence, source section type, and article type.

Classification rules:
- strong:   results/discussion section + original_research/meta_analysis + confidence >= 0.85
- moderate: results/discussion section + confidence >= 0.7 and < 0.85
- weak:     abstract/introduction section OR confidence < 0.7

Claim-level rules:
- If >= 3 papers with "moderate" or "strong" → claim is "strong"
- Otherwise, claim strength = max of individual strengths (strong > moderate > weak)

Requirements: 6.1, 6.2, 6.3, 6.5, 6.6
"""

from typing import List


# Strength ordering for comparison (higher index = stronger)
_STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}

# Valid section types for strong/moderate classification
_STRONG_SECTIONS = {"results", "discussion"}

# Valid article types that enable "strong" classification
_STRONG_ARTICLE_TYPES = {"original_research", "meta_analysis"}

# Section types that always produce "weak" classification
_WEAK_SECTIONS = {"abstract", "introduction"}


class EvidenceStrengthClassifier:
    """
    Assigns evidence strength labels to LLM-extracted triples based on:
    - Extraction confidence score
    - Source section type (results/discussion vs abstract/introduction)
    - Article type (original_research/meta_analysis vs review)

    Classification rules:
    - strong:   results/discussion section + original_research/meta_analysis + confidence >= 0.85
    - moderate: results/discussion section + confidence >= 0.7 and < 0.85
    - weak:     abstract/introduction section OR confidence < 0.7
    """

    def classify_single(
        self,
        confidence: float,
        section_type: str,
        article_type: str,
    ) -> str:
        """
        Classify evidence strength for a single triple.

        Args:
            confidence: Extraction confidence score [0.5, 1.0]
            section_type: Section from which triple was extracted
                (e.g., "results", "discussion", "abstract", "introduction")
            article_type: Normalized article type of the source paper
                (e.g., "original_research", "meta_analysis", "review")

        Returns:
            "strong", "moderate", or "weak"
        """
        section_lower = section_type.lower().strip()
        article_lower = article_type.lower().strip()

        # Rule 3 (weak): abstract/introduction OR confidence < 0.7
        # Check confidence first — if below 0.7, always weak regardless of section
        if confidence < 0.7:
            return "weak"

        # Check if section is a weak section (abstract/introduction)
        if section_lower in _WEAK_SECTIONS:
            return "weak"

        # At this point: confidence >= 0.7 and section is NOT abstract/introduction
        # Check if section is results/discussion for strong/moderate
        if section_lower in _STRONG_SECTIONS:
            # Rule 1 (strong): results/discussion + original_research/meta_analysis + confidence >= 0.85
            if article_lower in _STRONG_ARTICLE_TYPES and confidence >= 0.85:
                return "strong"

            # Rule 2 (moderate): results/discussion + confidence >= 0.7 and < 0.85
            # Also covers: results/discussion + confidence >= 0.85 but article_type
            # is NOT original_research/meta_analysis (still >= 0.7, so moderate)
            return "moderate"

        # Section is not in strong_sections or weak_sections (e.g., "methods", "other")
        # Since it's not abstract/introduction and confidence >= 0.7, treat as moderate
        # based on the rule: confidence >= 0.7 and not weak-section → at least moderate
        # But per the strict rules, "moderate" requires results/discussion section.
        # Any section not explicitly categorized and confidence >= 0.7 → weak
        # because it doesn't match any strong/moderate rule's section requirement.
        return "weak"

    def classify_claim(
        self,
        individual_strengths: List[str],
        paper_count: int,
    ) -> str:
        """
        Classify evidence strength for an aggregated Open_World_Claim.

        Rules:
        - If >= 3 papers with individual strength "moderate" or "strong" → "strong"
        - Otherwise, return the strongest individual classification

        Args:
            individual_strengths: Evidence strength of each supporting triple
            paper_count: Number of distinct papers (informational, not used in logic
                since individual_strengths already captures per-paper evidence)

        Returns:
            "strong", "moderate", or "weak"
        """
        if not individual_strengths:
            return "weak"

        # Count papers with moderate or strong evidence
        moderate_or_strong_count = sum(
            1 for s in individual_strengths if s in ("moderate", "strong")
        )

        # Promotion rule: >= 3 papers with moderate+ evidence → "strong"
        if moderate_or_strong_count >= 3:
            return "strong"

        # Otherwise, return the maximum strength among all individual items
        max_strength_value = max(
            _STRENGTH_ORDER.get(s, 0) for s in individual_strengths
        )

        # Convert back from numeric to string
        for strength_name, strength_value in _STRENGTH_ORDER.items():
            if strength_value == max_strength_value:
                return strength_name

        return "weak"
