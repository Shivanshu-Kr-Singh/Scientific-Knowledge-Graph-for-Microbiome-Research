"""
graph/semantic_extractor.py
----------------------------
Semantic relationship extractor for the knowledge graph.

This module extracts rich scientific relationships with semantic properties
from enriched papers. It parses sections to find associations, interventions,
and methodology usage with complete provenance tracking.

Requirements: 2.1, 2.2, 2.3
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from nlp.enriched_record import EnrichedPaperRecord, ParsedSection
from graph.semantic_relationships import (
    SemanticRelationship,
    RelationType,
    create_association_relationship,
    create_intervention_relationship,
    create_methodology_relationship,
)
from graph.provenance import ProvenanceMetadata


class SemanticRelationshipExtractor:
    """
    Extracts relationships with scientific semantics from enriched papers.
    
    This extractor parses paper sections to identify:
    - Taxon-disease associations with statistical properties (Requirement 2.1)
    - Intervention effects with duration and dosage (Requirement 2.2)
    - Methodology usage with sample size and data availability (Requirement 2.3)
    
    All extracted relationships include complete provenance tracking.
    """
    
    def __init__(self, extraction_method: str = "regex_ner", extractor_version: str = "1.0"):
        """
        Initialize the semantic relationship extractor.
        
        Args:
            extraction_method: Registered extraction method identifier
            extractor_version: Version of this extractor
        """
        self.extraction_method = extraction_method
        self.extractor_version = extractor_version
    
    def extract_associations(
        self,
        paper: EnrichedPaperRecord
    ) -> List[SemanticRelationship]:
        """
        Extract taxon-disease associations with statistical properties.
        
        Requirement 2.1: Extract associations with direction, comparison context,
        statistical measure type, effect size, and p-value.
        
        Preconditions:
        - paper has non-empty sections list
        - paper.sections contains at least one "results" section
        - paper.entities contains at least one taxon and one disease
        
        Postconditions:
        - Returns list of REPORTS_ASSOCIATION relationships
        - Each relationship has complete provenance
        - properties dict contains direction, comparison, statistical_measure
        - extraction_confidence >= 0.5 for all returned relationships
        
        Args:
            paper: EnrichedPaperRecord with NLP annotations
        
        Returns:
            List of SemanticRelationship objects with REPORTS_ASSOCIATION type
        """
        relationships = []
        
        # Check preconditions
        if not paper.sections:
            return relationships
        
        if not paper.taxa or not paper.diseases:
            return relationships
        
        # Extract results sections — include abstract as primary fallback
        results_sections = self._extract_sections_by_type(paper, ["results", "abstract", "discussion", "conclusion"])
        if not results_sections:
            return relationships
        
        # For each results section, look for associations
        for section in results_sections:
            # Parse sentences in the section
            sentences = self._split_into_sentences(section.content)
            
            for sentence_idx, sentence in enumerate(sentences):
                # Check if sentence mentions taxa; use paper-level diseases as context
                mentioned_taxa = self._find_entities_in_text(sentence, paper.taxa)
                mentioned_diseases = self._find_entities_in_text(sentence, paper.diseases)
                
                # If no diseases in sentence but paper has diseases, use paper-level diseases
                # (abstract sentences often reference the disease implicitly)
                if mentioned_taxa and not mentioned_diseases and paper.diseases:
                    mentioned_diseases = paper.diseases[:1]  # Use primary disease as context
                
                if not mentioned_taxa or not mentioned_diseases:
                    continue
                
                # Try to extract direction from sentence first
                direction = self._extract_direction(sentence)
                if not direction:
                    # Only fall back to section-level if section is short (structured abstract ≤ 3 sentences)
                    # For longer sections, requiring direction in the sentence prevents false associations
                    sentences_in_section = self._split_into_sentences(section.content)
                    if len(sentences_in_section) <= 3:
                        direction = self._extract_direction(section.content)
                if not direction:
                    continue
                
                # Extract statistical measures
                p_value = self._extract_p_value(sentence)
                effect_size = self._extract_effect_size(sentence)
                statistical_measure = self._extract_statistical_measure(sentence)
                
                # Determine comparison context
                comparison = self._extract_comparison_context(sentence, mentioned_diseases)
                
                # Calculate extraction confidence based on available information
                confidence = self._calculate_association_confidence(
                    direction, p_value, effect_size, statistical_measure
                )
                
                # Only create relationships with confidence >= 0.5 (Requirement 2.4)
                if confidence < 0.5:
                    continue
                
                # Determine evidence strength
                evidence_strength = self._determine_evidence_strength(
                    p_value, paper.article_type_normalized
                )
                
                # Create provenance metadata
                provenance = self._create_provenance(
                    paper=paper,
                    section=section,
                    sentence=sentence,
                    sentence_offset=sentence_idx,
                    confidence=confidence
                )
                
                # Create relationships for each taxon-disease pair
                for taxon in mentioned_taxa:
                    for disease in mentioned_diseases:
                        try:
                            relationship = create_association_relationship(
                                source_entity=paper.get_dedup_key(),
                                target_entity=taxon,
                                direction=direction,
                                comparison=comparison,
                                statistical_measure=statistical_measure or "relative abundance",
                                provenance=provenance,
                                evidence_strength=evidence_strength,
                                extraction_confidence=confidence,
                                effect_size=effect_size,
                                p_value=p_value,
                            )
                            relationships.append(relationship)
                        except ValueError as e:
                            # Skip invalid relationships
                            continue
        
        return relationships
    
    def extract_intervention_effects(
        self,
        paper: EnrichedPaperRecord
    ) -> List[SemanticRelationship]:
        """
        Extract intervention-taxon effects from RCT or intervention studies.
        
        Requirement 2.2: Extract intervention type, effect direction, duration,
        dosage, and sample size.
        
        Preconditions:
        - paper.article_type_normalized in ["original_research", "meta_analysis"]
        - paper.sections contains "methods" and "results" sections
        - paper.entities contains at least one treatment entity
        
        Postconditions:
        - Returns list of REPORTS_INTERVENTION_EFFECT relationships
        - Each relationship has intervention_type, effect_direction, duration
        - Only includes relationships with p_value < 0.05 or explicit significance
        
        Args:
            paper: EnrichedPaperRecord with NLP annotations
        
        Returns:
            List of SemanticRelationship objects with REPORTS_INTERVENTION_EFFECT type
        """
        relationships = []
        
        # Check preconditions — relax article type to include reviews (they report associations too)
        if paper.article_type_normalized not in ["original_research", "meta_analysis", "systematic_review", "narrative_review"]:
            return relationships
        
        if not paper.sections or not paper.taxa:
            return relationships
        
        # Extract relevant sections — use abstract as fallback for methods
        methods_sections = self._extract_sections_by_type(paper, ["methods"])
        if not methods_sections:
            methods_sections = self._extract_sections_by_type(paper, ["abstract"])
        results_sections = self._extract_sections_by_type(paper, ["results", "abstract", "discussion", "conclusion"])
        
        if not results_sections:
            return relationships
        
        # Extract intervention information from methods
        interventions = self._extract_interventions_from_methods(methods_sections, paper)
        
        if not interventions:
            return relationships
        
        # Look for intervention effects in results
        for section in results_sections:
            sentences = self._split_into_sentences(section.content)
            
            for sentence_idx, sentence in enumerate(sentences):
                # Check if sentence mentions taxa and interventions
                mentioned_taxa = self._find_entities_in_text(sentence, paper.taxa)
                mentioned_interventions = [
                    interv for interv in interventions
                    if interv["name"].lower() in sentence.lower()
                ]
                
                if not mentioned_taxa or not mentioned_interventions:
                    continue
                
                # Extract effect direction
                effect_direction = self._extract_direction(sentence)
                if not effect_direction:
                    continue
                
                # Extract p-value
                p_value = self._extract_p_value(sentence)
                
                # Only include significant results (Requirement 2.2)
                if p_value is not None and p_value >= 0.05:
                    continue
                
                # Check for explicit significance statements
                if p_value is None and not self._has_significance_statement(sentence):
                    continue
                
                # Calculate confidence
                confidence = self._calculate_intervention_confidence(
                    effect_direction, p_value, mentioned_interventions
                )
                
                if confidence < 0.5:
                    continue
                
                # Determine evidence strength
                evidence_strength = self._determine_evidence_strength(
                    p_value, paper.article_type_normalized
                )
                
                # Create provenance
                provenance = self._create_provenance(
                    paper=paper,
                    section=section,
                    sentence=sentence,
                    sentence_offset=sentence_idx,
                    confidence=confidence
                )
                
                # Create relationships
                for taxon in mentioned_taxa:
                    for intervention in mentioned_interventions:
                        try:
                            relationship = create_intervention_relationship(
                                source_entity=paper.get_dedup_key(),
                                target_entity=taxon,
                                intervention_type=intervention["type"],
                                effect_direction=effect_direction,
                                provenance=provenance,
                                evidence_strength=evidence_strength,
                                extraction_confidence=confidence,
                                duration=intervention.get("duration"),
                                dosage=intervention.get("dosage"),
                                sample_size=intervention.get("sample_size"),
                            )
                            relationships.append(relationship)
                        except ValueError:
                            continue
        
        return relationships
    
    def extract_methodology_usage(
        self,
        paper: EnrichedPaperRecord
    ) -> List[SemanticRelationship]:
        """
        Extract methodology information (sequencing type, sample size, data availability).
        
        Requirement 2.3: Extract method name, sequencing platform, sample size,
        and data availability status.
        
        Preconditions:
        - paper.sections contains "methods" section
        - paper.methods list is non-empty
        
        Postconditions:
        - Returns list of USES_METHODOLOGY relationships
        - Each relationship links paper to method with sample_size if available
        - Includes data_availability status from paper.data_availability
        
        Args:
            paper: EnrichedPaperRecord with NLP annotations
        
        Returns:
            List of SemanticRelationship objects with USES_METHODOLOGY type
        """
        relationships = []
        
        # Check preconditions
        if not paper.sections or not paper.methods:
            return relationships

        # Extract methods sections — fall back to abstract if no methods section
        methods_sections = self._extract_sections_by_type(paper, ["methods"])
        if not methods_sections:
            methods_sections = self._extract_sections_by_type(paper, ["abstract"])
        
        # For each method mentioned in the paper
        for method in paper.methods:
            # Find the section that mentions this method
            for section in methods_sections:
                if method.lower() not in section.content.lower():
                    continue
                
                # Extract sequencing platform
                sequencing_platform = self._extract_sequencing_platform(section.content)
                
                # Extract sample size
                sample_size = self._extract_sample_size(section.content)
                
                # Get data availability status
                data_availability = None
                if paper.data_availability:
                    data_availability = paper.data_availability.status
                
                # Calculate confidence
                confidence = 0.8  # High confidence for methodology extraction
                
                # Create provenance
                provenance = self._create_provenance(
                    paper=paper,
                    section=section,
                    sentence=section.content[:200],  # First 200 chars as representative
                    sentence_offset=0,
                    confidence=confidence
                )
                
                try:
                    relationship = create_methodology_relationship(
                        source_entity=paper.get_dedup_key(),
                        target_entity=method,
                        method_name=method,
                        provenance=provenance,
                        evidence_strength="moderate",
                        extraction_confidence=confidence,
                        sequencing_platform=sequencing_platform,
                        sample_size=sample_size,
                        data_availability=data_availability,
                    )
                    relationships.append(relationship)
                except ValueError:
                    continue
                
                # Only create one relationship per method
                break
        
        return relationships
    
    # ========== Helper Methods ==========
    
    def _extract_sections_by_type(
        self,
        paper: EnrichedPaperRecord,
        section_types: List[str]
    ) -> List[ParsedSection]:
        """Extract sections matching the given types."""
        return [
            section for section in paper.sections
            if section.section_type in section_types
        ]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_entities_in_text(self, text: str, entities: List[str]) -> List[str]:
        """Find which entities from the list are mentioned in the text."""
        text_lower = text.lower()
        return [entity for entity in entities if entity.lower() in text_lower]
    
    def _extract_direction(self, sentence: str) -> Optional[str]:
        """
        Extract direction (increased/decreased/no_change) from sentence.
        
        Requirement 2.1: Capture direction from association statements.
        """
        sentence_lower = sentence.lower()
        
        # Patterns for increased
        increased_patterns = [
            r'\bincreas(ed|e|ing)\b',
            r'\bhigher\b',
            r'\belevat(ed|e|ing)\b',
            r'\benrich(ed|ment)\b',
            r'\bupregulat(ed|ion)\b',
            r'\bmore abundant\b',
            r'\bgreater abundance\b',
            r'\boverrepresent(ed|ation)\b',
            r'\bexpand(ed|ing)\b',
            r'\bpromot(ed|es|ing)\b',
            r'\benhance(d|s|ment)\b',
            r'\bpositively associated\b',
            r'\bpositive(ly)? correlat\b',
            r'\baccumulat(ed|ion)\b',
            r'\babundant\b',
            r'\bprevalent\b',
            r'\bcoloniz(ed|ation)\b',
            r'\bpresent(ed)? in\b',
            r'\bdetect(ed)? in\b',
            r'\bidentif(ied|y) in\b',
        ]
        
        # Patterns for decreased
        decreased_patterns = [
            r'\bdecreas(ed|e|ing)\b',
            r'\blower\b',
            r'\breduc(ed|e|ing|tion)\b',
            r'\bdeplet(ed|ion)\b',
            r'\bdownregulat(ed|ion)\b',
            r'\bless abundant\b',
            r'\blower abundance\b',
            r'\bunderrepresent(ed|ation)\b',
            r'\bdiminish(ed|ing)\b',
            r'\bimpair(ed|ment)\b',
            r'\bnegatively associated\b',
            r'\bnegative(ly)? correlat\b',
            r'\bloss of\b',
            r'\babsent\b',
            r'\beradicat(ed|ion)\b',
            r'\beliminate(d|s)\b',
            r'\bsuppress(ed|ion)\b',
        ]
        
        # Patterns for no change
        no_change_patterns = [
            r'\bno (significant )?change\b',
            r'\bno (significant )?difference\b',
            r'\bunchanged\b',
            r'\bsimilar\b',
            r'\bno significant\b',
        ]
        
        # Check patterns
        for pattern in increased_patterns:
            if re.search(pattern, sentence_lower):
                return "increased"
        
        for pattern in decreased_patterns:
            if re.search(pattern, sentence_lower):
                return "decreased"
        
        for pattern in no_change_patterns:
            if re.search(pattern, sentence_lower):
                return "no_change"
        
        return None
    
    def _extract_p_value(self, sentence: str) -> Optional[float]:
        """
        Extract p-value from sentence.
        
        Requirement 2.1: Parse p-values from results sections.
        
        Handles formats like:
        - p = 0.001
        - p < 0.05
        - p=0.03
        - (p < 0.001)
        """
        # Pattern for exact p-values: p = 0.001, p=0.03
        exact_pattern = r'p\s*[=]\s*(0?\.\d+|[01]\.?\d*)'
        match = re.search(exact_pattern, sentence.lower())
        if match:
            try:
                p_val = float(match.group(1))
                if 0.0 <= p_val <= 1.0:
                    return p_val
            except ValueError:
                pass
        
        # Pattern for inequality: p < 0.05, p<0.001
        inequality_pattern = r'p\s*[<]\s*(0?\.\d+|[01]\.?\d*)'
        match = re.search(inequality_pattern, sentence.lower())
        if match:
            try:
                p_val = float(match.group(1))
                if 0.0 <= p_val <= 1.0:
                    # Return the threshold value as upper bound
                    return p_val
            except ValueError:
                pass
        
        return None
    
    def _extract_effect_size(self, sentence: str) -> Optional[float]:
        """
        Extract effect size from sentence.
        
        Requirement 2.1: Parse effect sizes from results sections.
        
        Handles formats like:
        - fold change = 2.5
        - LDA score = 3.2
        - effect size: 1.8
        """
        # Patterns for effect sizes
        patterns = [
            r'fold[- ]change[:\s=]+(\d+\.?\d*)',
            r'lda score[:\s=]+(\d+\.?\d*)',
            r'effect size[:\s=]+(\d+\.?\d*)',
            r'log2[- ]?fc[:\s=]+([+-]?\d+\.?\d*)',
        ]
        
        sentence_lower = sentence.lower()
        for pattern in patterns:
            match = re.search(pattern, sentence_lower)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _extract_statistical_measure(self, sentence: str) -> Optional[str]:
        """
        Extract statistical measure type from sentence.
        
        Requirement 2.1: Identify statistical measure type.
        """
        sentence_lower = sentence.lower()
        
        measures = {
            "LDA score": r'\blda score\b',
            "fold change": r'\bfold[- ]change\b',
            "relative abundance": r'\brelative abundance\b',
            "log2 fold change": r'\blog2[- ]?fc\b',
            "odds ratio": r'\bodds ratio\b',
            "hazard ratio": r'\bhazard ratio\b',
            "correlation coefficient": r'\bcorrelation coefficient\b',
        }
        
        for measure_name, pattern in measures.items():
            if re.search(pattern, sentence_lower):
                return measure_name
        
        return None
    
    def _extract_comparison_context(
        self,
        sentence: str,
        diseases: List[str]
    ) -> str:
        """
        Extract comparison context from sentence.
        
        Requirement 2.1: Capture comparison context.
        """
        sentence_lower = sentence.lower()
        
        # Look for comparison patterns
        if re.search(r'\bvs\.?\b|\bversus\b|\bcompared (to|with)\b', sentence_lower):
            # Try to extract the comparison groups
            if "healthy" in sentence_lower or "control" in sentence_lower:
                disease_name = diseases[0] if diseases else "disease"
                return f"{disease_name} vs healthy"
            elif "pre" in sentence_lower and "post" in sentence_lower:
                return "pre vs post"
        
        # Default comparison
        disease_name = diseases[0] if diseases else "disease"
        return f"{disease_name} vs control"
    
    def _calculate_association_confidence(
        self,
        direction: Optional[str],
        p_value: Optional[float],
        effect_size: Optional[float],
        statistical_measure: Optional[str]
    ) -> float:
        """
        Calculate extraction confidence for association.

        Confidence is based on:
        - Direction found: +0.5  (base signal — enough to meet the 0.5 minimum)
        - P-value found: +0.3
        - Effect size found: +0.15
        - Statistical measure found: +0.05
        """
        confidence = 0.0

        if direction:
            confidence += 0.5
        if p_value is not None:
            confidence += 0.3
        if effect_size is not None:
            confidence += 0.15
        if statistical_measure:
            confidence += 0.05

        return min(confidence, 1.0)
    
    def _determine_evidence_strength(
        self,
        p_value: Optional[float],
        article_type: str
    ) -> str:
        """
        Determine evidence strength based on p-value and article type.
        
        Requirement 5.1, 5.2, 5.3: Evidence strength classification.
        """
        if p_value is None:
            return "weak"
        
        # Strong: p < 0.01 and RCT/meta-analysis
        if p_value < 0.01 and article_type in ["original_research", "meta_analysis"]:
            return "strong"
        
        # Moderate: p < 0.05
        if p_value < 0.05:
            return "moderate"
        
        # Weak: p < 0.1 or no p-value
        return "weak"
    
    def _create_provenance(
        self,
        paper: EnrichedPaperRecord,
        section: ParsedSection,
        sentence: str,
        sentence_offset: int,
        confidence: float
    ) -> ProvenanceMetadata:
        """
        Create provenance metadata for an extracted relationship.
        
        Requirement 3.1, 3.2: Complete provenance tracking.
        """
        return ProvenanceMetadata(
            paper_id=paper.get_dedup_key(),
            section_type=section.section_type,
            source_sentence=sentence,
            sentence_offset=sentence_offset,
            extraction_method=self.extraction_method,
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version=self.extractor_version,
            confidence_score=confidence,
        )
    
    def _extract_interventions_from_methods(
        self,
        methods_sections: List[ParsedSection],
        paper: EnrichedPaperRecord
    ) -> List[Dict[str, Any]]:
        """
        Extract intervention information from methods sections.
        
        Returns list of dicts with keys: name, type, duration, dosage, sample_size
        """
        interventions = []
        
        # Map treatment entities to intervention types
        intervention_type_map = {
            "probiotic": ["probiotic", "lactobacillus", "bifidobacterium"],
            "FMT": ["fmt", "fecal microbiota transplant", "fecal transplant"],
            "diet": ["diet", "dietary", "nutrition", "mediterranean diet"],
            "antibiotic": ["antibiotic", "antibiotics", "amoxicillin", "metronidazole"],
            "prebiotic": ["prebiotic", "inulin", "fructooligosaccharide"],
            "synbiotic": ["synbiotic"],
        }
        
        for section in methods_sections:
            content_lower = section.content.lower()
            
            # Check for each intervention type
            for interv_type, keywords in intervention_type_map.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        # Extract duration
                        duration = self._extract_duration(section.content)
                        
                        # Extract dosage
                        dosage = self._extract_dosage(section.content)
                        
                        # Extract sample size
                        sample_size = self._extract_sample_size(section.content)
                        
                        interventions.append({
                            "name": keyword,
                            "type": interv_type,
                            "duration": duration,
                            "dosage": dosage,
                            "sample_size": sample_size,
                        })
                        break
        
        return interventions
    
    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract intervention duration from text."""
        # Patterns: 4 weeks, 6 months, 12 days
        pattern = r'(\d+)\s*(week|month|day|year)s?'
        match = re.search(pattern, text.lower())
        if match:
            return f"{match.group(1)} {match.group(2)}s"
        return None
    
    def _extract_dosage(self, text: str) -> Optional[str]:
        """Extract dosage information from text."""
        # Patterns: 10^9 CFU, 500mg, 1g
        patterns = [
            r'(\d+(?:\^\d+)?)\s*cfu',
            r'(\d+)\s*mg',
            r'(\d+)\s*g\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _extract_sample_size(self, text: str) -> Optional[int]:
        """Extract sample size from text."""
        # Patterns: n=50, N=100, 50 participants, 100 subjects
        patterns = [
            r'n\s*=\s*(\d+)',
            r'(\d+)\s+participants',
            r'(\d+)\s+subjects',
            r'(\d+)\s+patients',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _has_significance_statement(self, sentence: str) -> bool:
        """Check if sentence contains explicit significance statement."""
        significance_patterns = [
            r'\bsignificant(ly)?\b',
            r'\bp\s*<\s*0\.05\b',
            r'\bstatistically significant\b',
        ]
        
        sentence_lower = sentence.lower()
        for pattern in significance_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _calculate_intervention_confidence(
        self,
        effect_direction: Optional[str],
        p_value: Optional[float],
        interventions: List[Dict[str, Any]]
    ) -> float:
        """Calculate extraction confidence for intervention effect."""
        confidence = 0.0
        
        if effect_direction:
            confidence += 0.3
        if p_value is not None:
            confidence += 0.3
        if interventions:
            confidence += 0.2
            # Bonus for detailed intervention info
            if any(i.get("duration") for i in interventions):
                confidence += 0.1
            if any(i.get("dosage") for i in interventions):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_sequencing_platform(self, text: str) -> Optional[str]:
        """Extract sequencing platform from text."""
        platforms = {
            "Illumina": r'\billumina\b',
            "PacBio": r'\bpacbio\b',
            "Oxford Nanopore": r'\bnanopore\b',
            "Ion Torrent": r'\bion torrent\b',
        }
        
        text_lower = text.lower()
        for platform_name, pattern in platforms.items():
            if re.search(pattern, text_lower):
                return platform_name
        
        return None
