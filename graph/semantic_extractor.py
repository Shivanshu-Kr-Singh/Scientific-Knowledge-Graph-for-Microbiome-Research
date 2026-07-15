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
    create_taxon_produces_metabolite,
    create_taxon_modulates_pathway,
    create_taxon_regulates_gene,
    create_taxon_influences_immune_cell,
    create_taxon_affects_clinical_outcome,
    create_metabolite_linked_to_disease,
    create_metabolite_induces_immune_response,
    create_gene_predisposes_to_disease,
    create_diet_shapes_taxon,
    create_environment_shapes_taxon,
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
                    confidence=confidence,
                    sentences=sentences,
                )
                
                # Create relationships for each taxon-disease pair.
                # source_entity = taxon (the scientific subject of the claim)
                # target_entity = disease (the condition it is associated with)
                # The paper DOI is tracked in provenance.paper_id — NOT here.
                # Previously source_entity was set to paper.get_dedup_key() which
                # caused ScientificClaim.subject_entity to become a DOI string
                # instead of a bacterium name.
                for taxon in mentioned_taxa:
                    for disease in mentioned_diseases:
                        try:
                            relationship = create_association_relationship(
                                source_entity=taxon,
                                target_entity=disease,
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
                    confidence=confidence,
                    sentences=sentences,
                )

                # Create relationships for each taxon-intervention pair
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
                    confidence=confidence,
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
    
    # ========== New entity-pair extraction methods ==========
    # Each method follows the same pattern as extract_associations():
    #   1. Check that both entity lists for the pair are non-empty.
    #   2. Find co-occurring entity pairs in results/discussion sentences.
    #   3. Extract direction via _extract_direction_for_type().
    #   4. Build provenance and call the appropriate factory function.

    def _extract_direction_for_type(
        self,
        sentence: str,
        allowed: set,
        fallback_patterns: Dict[str, List[str]],
    ) -> Optional[str]:
        """
        Extract a direction value restricted to the allowed set for a given type.

        fallback_patterns maps direction label → list of regex patterns.
        Returns None when no pattern from any allowed direction matches.
        """
        sentence_lower = sentence.lower()
        for direction, patterns in fallback_patterns.items():
            if direction not in allowed:
                continue
            for pat in patterns:
                if re.search(pat, sentence_lower):
                    return direction
        return None

    # ── Direction pattern libraries for each new type ─────────────────────────
    _PRODUCE_PATTERNS: Dict[str, List[str]] = {
        "produces": [
            r'\bproduc(es?|ed|tion)\b', r'\bgenerat(es?|ed|ion)\b',
            r'\bsynth(esizes?|esized|esis)\b', r'\bsecret(es?|ed|ion)\b',
            r'\bexcret(es?|ed|ion)\b', r'\breleas(es?|ed|ing)\b',
        ],
        "inhibits": [
            r'\binhibit(s|ed|ion|ing)\b', r'\bsuppress(es?|ed|ion)\b',
            r'\bblocks?\b', r'\breduces? (production|synthesis|secretion)\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\bcorrelat(ed|ion)\b',
            r'\blinked\s+to\b', r'\brelated\s+to\b',
        ],
    }
    _PATHWAY_PATTERNS: Dict[str, List[str]] = {
        "activates": [
            r'\bactivat(es?|ed|ion|ing)\b', r'\binduces?\b',
            r'\bupregulat(es?|ed|ion)\b', r'\bpromot(es?|ed|ing)\b',
            r'\benhances?\b', r'\bstimulat(es?|ed|ion)\b',
            r'\btriggers?\b',
        ],
        "inhibits": [
            r'\binhibit(s|ed|ion|ing)\b', r'\bdownregulat(es?|ed|ion)\b',
            r'\bsuppress(es?|ed|ion)\b', r'\battenuates?\b',
            r'\bblocks?\b', r'\bimpairs?\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\bmodulat(es?|ed|ion)\b',
            r'\binfluences?\b', r'\baffects?\b',
        ],
    }
    _GENE_REG_PATTERNS: Dict[str, List[str]] = {
        "upregulates": [
            r'\bupregulat(es?|ed|ion)\b', r'\bincreased expression\b',
            r'\bhigher expression\b', r'\binduc(es?|ed|tion) of\b',
            r'\bactivat(es?|ed) (expression|transcription)\b',
        ],
        "downregulates": [
            r'\bdownregulat(es?|ed|ion)\b', r'\bdecreased expression\b',
            r'\blower expression\b', r'\brepresses?\b',
            r'\bsilenc(es?|ed|ing)\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\bregulat(es?|ed|ion)\b',
            r'\bexpression of\b', r'\baffects? expression\b',
        ],
    }
    _IMMUNE_PATTERNS: Dict[str, List[str]] = {
        "activates": [
            r'\bactivat(es?|ed|ion|ing)\b', r'\bexpands?\b',
            r'\bpolariz(es?|ed|ation)\b', r'\bpromot(es?|ed) (differentiation|activation)\b',
            r'\bstimulat(es?|ed|ion)\b', r'\binduces?\b',
        ],
        "suppresses": [
            r'\bsuppress(es?|ed|ion)\b', r'\binhibit(s|ed)\b',
            r'\bimpairs?\b', r'\breduces? (activation|number|count)\b',
            r'\bregulat(es?|ed) negatively\b',
        ],
        "recruits": [
            r'\brecruit(s|ed|ment)\b', r'\battract(s|ed|ion)\b',
            r'\bmigrat(es?|ed|ion) of\b', r'\baccumulat(es?|ed|ion) of\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\binfluences?\b',
            r'\baffects?\b', r'\bcorrelat(ed|ion)\b',
        ],
    }
    _CLINICAL_PATTERNS: Dict[str, List[str]] = {
        "improves": [
            r'\bimproves?\b', r'\bameliorate(s|d)\b', r'\balleviate(s|d)\b',
            r'\breduces? (symptoms?|severity|score)\b',
            r'\bbetter\b', r'\bremission\b', r'\bhealing\b', r'\bresolution\b',
        ],
        "worsens": [
            r'\bworsens?\b', r'\bexacerbate(s|d)\b', r'\baggravate(s|d)\b',
            r'\bincreases? (symptoms?|severity|score|risk)\b',
            r'\brelapse\b', r'\bprogress(es?|ion)\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\bcorrelat(ed|ion)\b',
            r'\bpredicts?\b', r'\bbiomarker\b', r'\blinked\s+to\b',
        ],
    }
    _META_DISEASE_PATTERNS: Dict[str, List[str]] = {
        "increased": [
            r'\bincreased\b', r'\bhigher levels?\b', r'\belevated\b',
            r'\baccumulat(ed|ion)\b',
        ],
        "decreased": [
            r'\bdecreased\b', r'\blower levels?\b', r'\bdepleted\b',
            r'\bdeficien(t|cy)\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\blinked\s+to\b', r'\bmarker\b',
            r'\bcorrelat(ed|ion)\b',
        ],
    }
    _META_IMMUNE_PATTERNS = _IMMUNE_PATTERNS   # reuse same vocabulary
    _GENE_DISEASE_PATTERNS: Dict[str, List[str]] = {
        "predisposes": [
            r'\bpredispos(es?|ed|ition)\b', r'\brisk factor\b',
            r'\bsusceptib(le|ility)\b', r'\bmutation\b',
            r'\bpolymorphism\b', r'\bvariant\b',
            r'\bincreases? risk\b',
        ],
        "protective": [
            r'\bprotect(s|ed|ive|ion)\b', r'\breduces? risk\b',
            r'\bprotective variant\b', r'\bresistance\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\blinked\s+to\b',
            r'\bcorrelat(ed|ion)\b', r'\bgwas\b',
        ],
    }
    _DIET_TAXON_PATTERNS: Dict[str, List[str]] = {
        "enriches": [
            r'\benrich(es?|ed|ment)\b', r'\bincreases? (abundance|count|level)\b',
            r'\bpromot(es?|ed) growth\b', r'\bfavor(s|ed)\b',
            r'\bexpand(s|ed)\b', r'\bprefer(s|red|ential)\b',
        ],
        "depletes": [
            r'\bdeplet(es?|ed|ion)\b', r'\breduces? (abundance|count|level)\b',
            r'\bsuppress(es?|ed) growth\b', r'\bdecreases? (abundance)\b',
            r'\beradicat(es?|ed)\b',
        ],
        "associated": [
            r'\bassociat(ed|ion)\b', r'\bshapes?\b', r'\bmodulat(es?|ed)\b',
            r'\binfluences?\b', r'\bcorrelat(ed|ion)\b',
        ],
    }
    _ENV_TAXON_PATTERNS = _DIET_TAXON_PATTERNS  # same vocabulary

    # ── Helper: generic extraction for all 10 new pair types ──────────────────

    def _extract_entity_pair_relationships(
        self,
        paper: "EnrichedPaperRecord",
        source_entities: List[str],
        target_entities: List[str],
        relation_type: RelationType,
        direction_patterns: Dict[str, List[str]],
        factory_fn,
        context_extractor=None,
    ) -> List[SemanticRelationship]:
        """
        Generic co-occurrence extractor for entity-pair relationship types.

        Args:
            paper:              Enriched paper record.
            source_entities:    Left-hand entity list (e.g. paper.taxa).
            target_entities:    Right-hand entity list (e.g. paper.metabolites).
            relation_type:      The RelationType being extracted.
            direction_patterns: {direction_label: [regex_pattern, …]} dict.
            factory_fn:         One of the create_* factory functions.
            context_extractor:  Optional callable(sentence, source, target) → dict
                                 that returns extra kwargs for the factory function.

        Returns:
            List of SemanticRelationship objects.
        """
        relationships: List[SemanticRelationship] = []

        if not paper.sections or not source_entities or not target_entities:
            return relationships

        allowed = set(direction_patterns.keys())
        priority_types = ["results", "abstract", "discussion", "conclusion"]
        sections = self._extract_sections_by_type(paper, priority_types)
        if not sections:
            return relationships

        for section in sections:
            sentences = self._split_into_sentences(section.content)
            for sent_idx, sentence in enumerate(sentences):
                mentioned_src = self._find_entities_in_text(sentence, source_entities)
                mentioned_tgt = self._find_entities_in_text(sentence, target_entities)
                if not mentioned_src or not mentioned_tgt:
                    continue

                direction = self._extract_direction_for_type(
                    sentence, allowed, direction_patterns
                )
                if not direction:
                    continue

                p_value = self._extract_p_value(sentence)
                confidence = self._calculate_association_confidence(
                    direction, p_value, None, None
                )
                if confidence < 0.5:
                    continue

                evidence_strength = self._determine_evidence_strength(
                    p_value, paper.article_type_normalized
                )
                provenance = self._create_provenance(
                    paper=paper,
                    section=section,
                    sentence=sentence,
                    sentence_offset=sent_idx,
                    confidence=confidence,
                    sentences=sentences,
                )

                for src in mentioned_src:
                    for tgt in mentioned_tgt:
                        extra_kwargs = {}
                        if context_extractor is not None:
                            extra_kwargs = context_extractor(sentence, src, tgt)
                        try:
                            rel = factory_fn(
                                source_entity=src,
                                target_entity=tgt,
                                direction=direction,
                                provenance=provenance,
                                evidence_strength=evidence_strength,
                                extraction_confidence=confidence,
                                p_value=p_value,
                                **extra_kwargs,
                            )
                            relationships.append(rel)
                        except ValueError:
                            continue

        return relationships

    # ── 10 public extraction methods ─────────────────────────────────────────

    def extract_taxon_metabolite(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract taxon → metabolite production / inhibition relationships.

        Examples: "F. prausnitzii produces butyrate (p < 0.01)"
        """
        def _metabolite_class(sentence: str, src: str, tgt: str):
            for cls, pats in {
                "SCFA": [r'\bscfa\b', r'\bshort[- ]chain fatty acid\b'],
                "bile acid": [r'\bbile acid\b'],
                "indole": [r'\bindole\b'],
                "LPS": [r'\blps\b', r'\blipopolysaccharide\b'],
            }.items():
                for p in pats:
                    if re.search(p, sentence.lower()):
                        return {"metabolite_class": cls}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.taxa, paper.metabolites,
            RelationType.TAXON_PRODUCES_METABOLITE,
            self._PRODUCE_PATTERNS, create_taxon_produces_metabolite,
            context_extractor=_metabolite_class,
        )

    def extract_taxon_pathway(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract taxon → pathway modulation relationships.

        Examples: "Akkermansia inhibits NF-κB signaling"
        """
        def _pathway_cat(sentence: str, src: str, tgt: str):
            for cat, pats in {
                "inflammatory": [r'\binflammator\b', r'\bnf.?kb\b', r'\btlr\b'],
                "metabolic": [r'\bmetabol\b', r'\bbutyrate\b', r'\bscfa\b'],
                "immune": [r'\bimmune\b', r'\bjak.?stat\b'],
            }.items():
                for p in pats:
                    if re.search(p, sentence.lower()):
                        return {"pathway_category": cat}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.taxa, paper.pathways,
            RelationType.TAXON_MODULATES_PATHWAY,
            self._PATHWAY_PATTERNS, create_taxon_modulates_pathway,
            context_extractor=_pathway_cat,
        )

    def extract_taxon_gene(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract taxon → gene regulation relationships.

        Examples: "Lactobacillus upregulates MUC2 expression"
        """
        def _mech(sentence: str, src: str, tgt: str):
            for mech, pats in {
                "epigenetic": [r'\bepigenetic\b', r'\bhistone\b', r'\bmethylation\b'],
                "transcriptional": [r'\btranscription\b', r'\bpromoter\b'],
                "post-translational": [r'\bpost.?translational\b', r'\bphosphorylation\b'],
            }.items():
                for p in pats:
                    if re.search(p, sentence.lower()):
                        return {"regulation_mechanism": mech}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.taxa, paper.genes,
            RelationType.TAXON_REGULATES_GENE,
            self._GENE_REG_PATTERNS, create_taxon_regulates_gene,
            context_extractor=_mech,
        )

    def extract_taxon_immune_cell(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract taxon → immune cell influence relationships.

        Examples: "SCFAs activate Treg cells in the intestinal mucosa"
        """
        def _ctx(sentence: str, src: str, tgt: str):
            for ctx, pats in {
                "intestinal": [r'\bintestin\b', r'\bmucosal\b', r'\bgut\b'],
                "systemic": [r'\bsystemic\b', r'\bperipheral blood\b', r'\bblood\b'],
            }.items():
                for p in pats:
                    if re.search(p, sentence.lower()):
                        return {"immune_context": ctx}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.taxa, paper.immune_cells,
            RelationType.TAXON_INFLUENCES_IMMUNE_CELL,
            self._IMMUNE_PATTERNS, create_taxon_influences_immune_cell,
            context_extractor=_ctx,
        )

    def extract_taxon_clinical_outcome(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract taxon → clinical outcome associations.

        Examples: "Reduced Faecalibacterium was associated with relapse"
        """
        def _otype(sentence: str, src: str, tgt: str):
            for ot, pats in {
                "remission": [r'\bremission\b'],
                "relapse": [r'\brelapse\b', r'\brecurrence\b'],
                "dysbiosis": [r'\bdysbiosis\b'],
                "permeability": [r'\bpermeab\b', r'\bleaky gut\b'],
            }.items():
                for p in pats:
                    if re.search(p, sentence.lower()):
                        return {"outcome_type": ot}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.taxa, paper.clinical_outcomes,
            RelationType.TAXON_AFFECTS_CLINICAL_OUTCOME,
            self._CLINICAL_PATTERNS, create_taxon_affects_clinical_outcome,
            context_extractor=_otype,
        )

    def extract_metabolite_disease(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract metabolite → disease associations.

        Examples: "Butyrate deficiency was linked to IBD severity"
        """
        def _role(sentence: str, src: str, tgt: str):
            sl = sentence.lower()
            if re.search(r'\bprotect(ive|s|ed)\b', sl):
                return {"metabolite_role": "protective"}
            if re.search(r'\bpathogen(ic|esis)?\b|\bdamag(es?|ing)\b', sl):
                return {"metabolite_role": "pathogenic"}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.metabolites, paper.diseases,
            RelationType.METABOLITE_LINKED_TO_DISEASE,
            self._META_DISEASE_PATTERNS, create_metabolite_linked_to_disease,
            context_extractor=_role,
        )

    def extract_metabolite_immune_cell(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract metabolite → immune cell response relationships.

        Examples: "Propionate induces regulatory T cell differentiation"
        """
        def _ctx(sentence: str, src: str, tgt: str):
            sl = sentence.lower()
            if re.search(r'\bintestin\b|\bmucosal\b|\bgut\b', sl):
                return {"immune_context": "intestinal"}
            if re.search(r'\bsystemic\b|\bperipheral\b', sl):
                return {"immune_context": "systemic"}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.metabolites, paper.immune_cells,
            RelationType.METABOLITE_INDUCES_IMMUNE_RESPONSE,
            self._META_IMMUNE_PATTERNS, create_metabolite_induces_immune_response,
            context_extractor=_ctx,
        )

    def extract_gene_disease(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract gene → disease predisposition / protective associations.

        Examples: "NOD2 mutations increase risk of Crohn's disease"
        """
        def _variant(sentence: str, src: str, tgt: str):
            sl = sentence.lower()
            for vt, pats in {
                "SNP": [r'\bsnp\b', r'\bsingle nucleotide\b'],
                "mutation": [r'\bmutation\b'],
                "polymorphism": [r'\bpolymorphism\b'],
            }.items():
                for p in pats:
                    if re.search(p, sl):
                        return {"variant_type": vt}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.genes, paper.diseases,
            RelationType.GENE_PREDISPOSES_TO_DISEASE,
            self._GENE_DISEASE_PATTERNS, create_gene_predisposes_to_disease,
            context_extractor=_variant,
        )

    def extract_diet_taxon(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract dietary component → taxon shaping relationships.

        Examples: "Inulin supplementation enriched Bifidobacterium"
        """
        def _dp(sentence: str, src: str, tgt: str):
            sl = sentence.lower()
            for dp, pats in {
                "Mediterranean": [r'\bmediterranean\b'],
                "high-fiber": [r'\bhigh.?fiber\b', r'\bdietary fiber\b'],
                "ketogenic": [r'\bketogenic\b'],
            }.items():
                for p in pats:
                    if re.search(p, sl):
                        return {"dietary_pattern": dp}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.dietary_components, paper.taxa,
            RelationType.DIET_SHAPES_TAXON,
            self._DIET_TAXON_PATTERNS, create_diet_shapes_taxon,
            context_extractor=_dp,
        )

    def extract_environment_taxon(
        self, paper: "EnrichedPaperRecord"
    ) -> List[SemanticRelationship]:
        """
        Extract environmental factor → taxon shaping relationships.

        Examples: "Antibiotic exposure depleted Bacteroidetes"
        """
        def _exp(sentence: str, src: str, tgt: str):
            sl = sentence.lower()
            for et, pats in {
                "antibiotic": [r'\bantibiotic\b'],
                "birth_mode": [r'\bcesarean\b', r'\bc.?section\b', r'\bvaginal delivery\b'],
                "breastfeeding": [r'\bbreastfeed\b', r'\bhuman milk\b'],
                "geographic": [r'\bgeograph\b', r'\bcountry\b', r'\bregion\b'],
            }.items():
                for p in pats:
                    if re.search(p, sl):
                        return {"exposure_type": et}
            return {}

        return self._extract_entity_pair_relationships(
            paper, paper.environmental_factors, paper.taxa,
            RelationType.ENVIRONMENT_SHAPES_TAXON,
            self._ENV_TAXON_PATTERNS, create_environment_shapes_taxon,
            context_extractor=_exp,
        )

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
        Extract direction (increased/decreased/no_change/associated) from sentence.

        Priority order:
          1. increased (directional patterns)
          2. decreased (directional patterns)
          3. no_change (stability patterns)
          4. associated (non-directional association signals)
          5. None (no pattern matched, relationship is discarded)

        Requirement 2.1: Capture direction from association statements.
        The "associated" direction captures non-directional co-occurrence signals
        (e.g., "implicated in", "biomarker for") that would otherwise be discarded.
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
        
        # Patterns for non-directional association (checked last)
        associated_patterns = [
            r'\bimplicat(ed|es|ing)\b',
            r'\brole\s+(in|of)\b',
            r'\bbiomarker\b',
            r'\bpathobiont\b',
            r'\bdysbiosis\b',
            r'\bdysbiotic\b',
            r'\bprotective\b',
            r'\blinked\s+to\b',
            r'\bcontribut(or|es|ing|ed)\s+(to|in)\b',
            r'\bmarker\s+(of|for)\b',
            r'\bsignature\s+(of|for)\b',
            r'\binvolv(ed|ement)\b',
            r'\bcorrelat(ed|ion)\b',
            r'\brelated\s+to\b',
            r'\bpathogen(ic|esis)?\b',
            r'\bcommensal\b',
            r'\bmutualist(ic)?\b',
            r'\bsymbio(nt|tic|sis)\b',
            r'\bdominant\b',
            r'\bkey\s+(species|taxon|organism|member)\b',
            r'\bdiffered\s+significantly\b',
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
        
        for pattern in associated_patterns:
            if re.search(pattern, sentence_lower):
                return "associated"
        
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
        confidence: float,
        sentences: Optional[List[str]] = None,
    ) -> ProvenanceMetadata:
        """
        Create provenance metadata for an extracted relationship.

        Args:
            sentences: The full list of sentences from the section. When supplied,
                       the ±1 sentences around sentence_offset are joined and stored
                       as surrounding_context so the edge is self-explanatory in the
                       graph browser without having to look up the original paper.

        Requirement 3.1, 3.2: Complete provenance tracking.
        """
        surrounding_context: Optional[str] = None
        if sentences and len(sentences) > 1:
            start = max(0, sentence_offset - 1)
            end   = min(len(sentences), sentence_offset + 2)   # +2 because slice is exclusive
            window = sentences[start:end]
            # Only store context when there are neighbour sentences — not just the sentence itself
            if len(window) > 1:
                surrounding_context = " ".join(window)

        return ProvenanceMetadata(
            paper_id=paper.get_dedup_key(),
            section_type=section.section_type,
            source_sentence=sentence,
            sentence_offset=sentence_offset,
            extraction_method=self.extraction_method,
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version=self.extractor_version,
            confidence_score=confidence,
            surrounding_context=surrounding_context,
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
        
        # Map treatment entities to intervention types.
        # Covers all 7 treatment categories extracted by Layer 2 NER (nlp/ner.py
        # TREATMENT_PATTERNS), so no recognised intervention is silently dropped.
        # Each key becomes the intervention_type stored on the graph edge.
        intervention_type_map = {
            # ── Microbial interventions ───────────────────────────────────────
            "probiotic": [
                "probiotic", "lactobacillus", "bifidobacterium", "saccharomyces",
                "streptococcus thermophilus", "enterococcus faecium",
                "pediococcus", "leuconostoc",
                "live biotherapeutic", "biotherapeutic",
                "paraprobiotic", "psychobiotic", "postbiotic",
                "fermented milk", "yogurt", "kefir",
            ],
            "prebiotic": [
                "prebiotic", "inulin", "fructooligosaccharide", "fos",
                "galactooligosaccharide", "gos", "pectin", "psyllium",
                "betaglucan", "beta-glucan", "resistant starch",
                "dietary fiber", "fiber supplement",
            ],
            "synbiotic": [
                "synbiotic",
            ],
            "postbiotic": [
                "postbiotic", "paraprobiotics", "heat-killed",
                "bacterial lysate", "cell-free supernatant",
            ],
            "FMT": [
                "fmt", "fecal microbiota transplant", "faecal microbiota transplant",
                "fecal transplant", "stool transplant", "microbiota transplant",
                "fecal bacteriotherapy", "bacteriotherapy",
                "capsule fmt", "enema fmt", "colonoscopic fmt",
            ],
            "fermented_food": [
                "fermented food", "fermented beverage", "fermented dairy",
                "sauerkraut", "kimchi", "kombucha", "miso", "tempeh",
                "natto", "kefir", "fermented soy",
            ],
            # ── Antibiotic / antimicrobial interventions ──────────────────────
            "antibiotic": [
                "antibiotic", "antibiotics", "antimicrobial",
                "amoxicillin", "metronidazole", "ciprofloxacin", "vancomycin",
                "rifaximin", "neomycin", "ampicillin", "tetracycline",
                "doxycycline", "clindamycin", "azithromycin", "clarithromycin",
                "fluoroquinolone", "cephalosporin", "penicillin",
                "carbapenems", "colistin", "polymyxin", "linezolid",
                "daptomycin", "fidaxomicin",
            ],
            # ── Dietary pattern interventions ─────────────────────────────────
            "diet": [
                "diet", "dietary intervention",
                "mediterranean diet", "high-fiber diet", "high fiber diet",
                "plant-based diet", "plant based diet",
                "ketogenic diet", "low-carbohydrate diet", "low carb diet",
                "low-fat diet", "vegan diet", "vegetarian diet",
                "gluten-free diet", "dairy-free diet",
                "western diet", "high-fat diet",
                "caloric restriction", "calorie restriction",
                "intermittent fasting", "time-restricted eating",
                "time-restricted feeding", "fasting",
                "food supplementation", "nutritional intervention",
                "polyphenol", "quercetin", "resveratrol", "curcumin",
                "omega-3", "fish oil", "dha", "epa",
                "vitamin d", "vitamin b12", "vitamin k",
                "folate", "folic acid", "zinc", "iron",
                "magnesium", "calcium", "selenium",
            ],
            # ── Metabolite / SCFA supplementation ────────────────────────────
            "metabolite_supplementation": [
                "butyrate", "propionate", "acetate",
                "short-chain fatty acid", "scfa",
                "bile acid", "secondary bile acid",
                "tryptophan", "indole", "serotonin",
                "gaba", "dopamine",
            ],
            # ── Pharmaceutical drugs ──────────────────────────────────────────
            "drug_metabolic": [
                "metformin", "insulin", "glp-1",
                "semaglutide", "liraglutide", "exenatide",
                "statin", "atorvastatin", "rosuvastatin",
                "levothyroxine", "thyroid hormone",
            ],
            "drug_gastro": [
                "proton pump inhibitor", "ppi",
                "omeprazole", "pantoprazole", "esomeprazole",
                "laxative", "antidiarrheal",
                "mesalazine", "sulfasalazine", "budesonide",
            ],
            "drug_immune": [
                "immunosuppressant", "corticosteroid",
                "prednisolone", "dexamethasone",
                "infliximab", "adalimumab", "vedolizumab",
                "ustekinumab", "tofacitinib",
                "nsaid", "non-steroidal anti-inflammatory",
                "ibuprofen", "aspirin", "naproxen",
            ],
            "drug_oncology": [
                "chemotherapy", "immunotherapy",
                "checkpoint inhibitor", "pd-1", "pd-l1",
                "pembrolizumab", "nivolumab", "ipilimumab",
            ],
            "drug_contraceptive": [
                "oral contraceptive", "hormone therapy",
                "hormonal contraception", "birth control pill",
            ],
            # ── Lifestyle interventions ───────────────────────────────────────
            "exercise": [
                "exercise", "physical activity",
                "aerobic exercise", "resistance training",
                "endurance training", "high-intensity interval training", "hiit",
                "yoga", "sedentary",
            ],
            "lifestyle_other": [
                "sleep", "stress", "psychological stress",
                "mindfulness", "meditation",
                "smoking", "tobacco", "smoking cessation",
                "alcohol", "alcohol consumption",
            ],
            # ── Perinatal / early-life interventions ─────────────────────────
            "perinatal": [
                "breastfeeding", "breast milk", "human milk",
                "formula feeding", "infant formula",
                "cesarean section", "c-section", "caesarean",
                "vaginal delivery", "mode of delivery",
                "early-life antibiotic", "neonatal antibiotic",
            ],
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
