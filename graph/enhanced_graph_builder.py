"""
graph/enhanced_graph_builder.py
--------------------------------
Enhanced graph builder integrating semantic extractor and relationship reifier.

This module creates the enhanced knowledge graph by:
1. Extracting rich semantic relationships from papers using SemanticRelationshipExtractor
2. Creating reified claims from multiple papers using RelationshipReifier
3. Building EnhancedGraphEdge objects with embedded provenance
4. (Additive) Discovering open-world triples via LLMTripleExtractor when USE_LLM=true

Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 4.1
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict

from nlp.enriched_record import EnrichedPaperRecord
from graph.semantic_extractor import SemanticRelationshipExtractor
from graph.relationship_reifier import RelationshipReifier
from graph.semantic_relationships import SemanticRelationship, RelationType
from graph.reified_claims import ScientificClaim
from graph.provenance import ProvenanceMetadata
from graph.entity_normalizer import EntityNormalizer
from graph.llm_triple_extractor import LLMTripleExtractor
from graph.triple_promoter import TriplePromoter
from graph.triple_promotion_models import PaperMetadata, PromotedTriple, OpenWorldClaim


class EnhancedGraphEdge:
    """
    Graph edge with scientific semantics and complete provenance.
    
    This is the final edge representation that will be loaded into Neo4j,
    containing all semantic properties and embedded provenance metadata.
    
    Requirements: 2.1, 2.2, 2.3, 3.1, 3.2
    """
    
    def __init__(
        self,
        source: str,
        target: str,
        relation: str,
        properties: Dict[str, Any],
        provenance: ProvenanceMetadata,
        evidence_strength: str,
        confidence: float
    ):
        """
        Initialize an enhanced graph edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation: Relationship type (e.g., "REPORTS_ASSOCIATION")
            properties: Semantic properties dict
            provenance: Complete provenance metadata
            evidence_strength: "strong" | "moderate" | "weak"
            confidence: Extraction confidence [0.0, 1.0]
        """
        self.source = source
        self.target = target
        self.relation = relation
        self.properties = properties
        self.provenance = provenance
        self.evidence_strength = evidence_strength
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert edge to dictionary for Neo4j loading.
        
        Returns:
            Dictionary with all edge properties including embedded provenance
        """
        edge_dict = {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "evidence_strength": self.evidence_strength,
            "confidence": self.confidence,
        }
        
        # Add semantic properties
        edge_dict.update(self.properties)
        
        # Embed provenance metadata as edge properties
        edge_dict.update({
            "paper_id": self.provenance.paper_id,
            "section": self.provenance.section_type,
            "source_sentence": self.provenance.source_sentence,
            "sentence_offset": self.provenance.sentence_offset,
            "extraction_method": self.provenance.extraction_method,
            "extraction_timestamp": self.provenance.extraction_timestamp.isoformat(),
            "extractor_version": self.provenance.extractor_version,
            "llm_prompt_hash": self.provenance.llm_prompt_hash,
            "validation_status": self.provenance.validation_status,
            "surrounding_context": self.provenance.surrounding_context,
            "figure_table_ref": self.provenance.figure_table_ref,
        })
        
        return edge_dict
    
    def __repr__(self) -> str:
        return (
            f"EnhancedGraphEdge(source={self.source}, target={self.target}, "
            f"relation={self.relation}, confidence={self.confidence})"
        )


class EnhancedGraphBuilder:
    """
    Enhanced graph builder integrating semantic extractor and relationship reifier.
    
    This builder creates a scientific knowledge graph by:
    1. Extracting rich semantic relationships from papers
    2. Creating reified claims from multiple papers
    3. Building enhanced graph edges with embedded provenance
    
    Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 4.1
    """
    
    def __init__(
        self,
        extraction_method: str = "regex_ner",
        extractor_version: str = "1.0"
    ):
        """
        Initialize the enhanced graph builder.
        
        Args:
            extraction_method: Registered extraction method identifier
            extractor_version: Version of the extractor
        """
        self.semantic_extractor = SemanticRelationshipExtractor(
            extraction_method=extraction_method,
            extractor_version=extractor_version
        )
        self.relationship_reifier = RelationshipReifier()

        # Entity normalizer — grounds entities to canonical ontology IDs
        # before they are written to the graph
        self.entity_normalizer = EntityNormalizer()

        # LLM triple extractor for open-world relationship discovery (additive).
        # Active only when USE_LLM=true; no-ops otherwise.
        self.llm_triple_extractor = LLMTripleExtractor()
        # Open-world triples stored as plain dicts (not EnhancedGraphEdge objects)
        # because they don't share the rigid schema of the 3 canonical relation types.
        self.open_world_triples: List[Dict] = []

        # TriplePromoter — set optionally via set_triple_promoter().
        # When set, raw LLM triples are promoted to PromotedTriple objects.
        self.triple_promoter: Optional[TriplePromoter] = None
        # Promoted triples accumulated across all papers processed by this builder.
        self.promoted_triples: List[PromotedTriple] = []
        # OpenWorldClaim nodes aggregated after all papers are processed.
        self.open_world_claims: List[OpenWorldClaim] = []

        # Storage for relationships and claims
        self.relationships: List[SemanticRelationship] = []
        self.edges: List[EnhancedGraphEdge] = []
        self.claims: List[ScientificClaim] = []

        # Index for grouping relationships by (subject, predicate, object)
        self.relationship_index: Dict[Tuple[str, str, str], List[SemanticRelationship]] = defaultdict(list)

        # Cache of current paper's pre-grounded entities (set in process_paper)
        self._current_paper_entities: list = []
    
    def set_triple_promoter(self, promoter: TriplePromoter) -> None:
        """
        Set the TriplePromoter to use for promoting LLM-extracted triples.

        When set, triples extracted by LLMTripleExtractor are enriched with
        full provenance, entity normalization, evidence strength classification,
        and stored in self.promoted_triples alongside the raw self.open_world_triples.

        Args:
            promoter: Configured TriplePromoter instance
        """
        self.triple_promoter = promoter

    def process_paper(self, paper: EnrichedPaperRecord) -> List[EnhancedGraphEdge]:
        """
        Process a single paper and extract all relationships.
        """
        # Cache the paper's pre-grounded entities for use in _find_grounded_entity
        self._current_paper_entities = paper.entities
        paper_edges = []

        # Extract associations (Requirement 2.1)
        associations = self.semantic_extractor.extract_associations(paper)
        for rel in associations:
            edge = self._create_edge_from_relationship(rel)
            self._inject_paper_metadata(edge, paper)
            paper_edges.append(edge)
            self.relationships.append(rel)
            key = self._get_relationship_key(rel)
            self.relationship_index[key].append(rel)

        # Extract intervention effects (Requirement 2.2)
        interventions = self.semantic_extractor.extract_intervention_effects(paper)
        for rel in interventions:
            edge = self._create_edge_from_relationship(rel)
            self._inject_paper_metadata(edge, paper)
            paper_edges.append(edge)
            self.relationships.append(rel)
            key = self._get_relationship_key(rel)
            self.relationship_index[key].append(rel)

        # Extract methodology usage (Requirement 2.3)
        methodologies = self.semantic_extractor.extract_methodology_usage(paper)
        for rel in methodologies:
            edge = self._create_edge_from_relationship(rel)
            self._inject_paper_metadata(edge, paper)
            paper_edges.append(edge)
            self.relationships.append(rel)
            key = self._get_relationship_key(rel)
            self.relationship_index[key].append(rel)

        # Open-world triple extraction (additive, requires USE_LLM=true)
        self._process_open_world_relationships(paper)

        self.edges.extend(paper_edges)
        return paper_edges

    def _process_open_world_relationships(self, paper: EnrichedPaperRecord) -> None:
        """
        Extract open-world (subject, predicate, object) triples from the paper
        using the LLMTripleExtractor (additive — does not replace regex extraction).

        Prioritises results/discussion sections where causal claims are most
        likely to appear. Falls back to the abstract when no full-text sections
        are available.

        Results are appended to self.open_world_triples as plain dicts so they
        can later be loaded into Neo4j as RELATES_TO relationships with
        canonical_predicate, raw_predicate, subject_type, and object_type
        as properties.

        No-ops silently when USE_LLM != "true" or Ollama is unavailable.
        """
        if not self.llm_triple_extractor._available:
            return

        paper_id = paper.doi or paper.pmid or paper.title[:50] or "unknown"

        # Rank sections: results and discussion first, then others
        priority_order = ["results", "discussion", "abstract", "introduction", "other"]
        sections_by_type: Dict[str, List] = defaultdict(list)
        for section in paper.sections:
            sections_by_type[section.section_type].append(section)

        # Build ordered list of (section_type, content) pairs
        ordered_sections = []
        for stype in priority_order:
            for sec in sections_by_type.get(stype, []):
                ordered_sections.append((stype, sec.content))

        # Fall back to abstract when no sections parsed
        if not ordered_sections and paper.abstract:
            ordered_sections.append(("abstract", paper.abstract))

        for section_type, content in ordered_sections:
            try:
                triples = self.llm_triple_extractor.extract_triples(
                    text=content,
                    paper_id=paper_id,
                    section_type=section_type,
                )
                # Always keep raw triples for backward compatibility
                self.open_world_triples.extend(triples)

                # If a TriplePromoter is configured, promote the batch and
                # accumulate the enriched results for later claim aggregation.
                if self.triple_promoter is not None and triples:
                    paper_metadata = PaperMetadata(
                        paper_id=paper_id,
                        article_type=paper.article_type_normalized or "unknown",
                        publication_year=paper.publication_year,
                        sections_available=[s.section_type for s in paper.sections],
                    )
                    promoted = self.triple_promoter.promote_batch(triples, paper_metadata)
                    self.promoted_triples.extend(promoted)

            except Exception as exc:
                from loguru import logger
                logger.warning(
                    "[EnhancedGraphBuilder] open-world extraction failed for {} / {}: {}",
                    paper_id[:30],
                    section_type,
                    exc,
                )

        # After all sections for this paper, check for threshold-based predicate promotion
        if self.triple_promoter is not None:
            self.triple_promoter.check_predicate_promotion()

    def _inject_paper_metadata(self, edge: "EnhancedGraphEdge", paper: EnrichedPaperRecord):
        """Inject paper-level metadata into edge properties for Neo4j loading."""
        edge.properties["year"] = paper.publication_year
        edge.properties["article_type"] = paper.article_type_normalized
        edge.properties["data_availability"] = (
            paper.data_availability.status if paper.data_availability else "not_stated"
        )
        edge.properties["accession_numbers"] = (
            paper.data_availability.accession_numbers
            if paper.data_availability and paper.data_availability.accession_numbers
            else []
        )
    
    def process_papers(self, papers: List[EnrichedPaperRecord]) -> List[EnhancedGraphEdge]:
        """
        Process multiple papers and extract all relationships.
        
        Args:
            papers: List of enriched paper records
        
        Returns:
            List of all EnhancedGraphEdge objects
        """
        all_edges = []
        
        for paper in papers:
            paper_edges = self.process_paper(paper)
            all_edges.extend(paper_edges)

        # After all papers are processed, aggregate promoted triples into
        # OpenWorldClaim nodes (requires >= 2 distinct papers per triple key).
        if self.triple_promoter is not None and self.promoted_triples:
            self.open_world_claims = self.triple_promoter.aggregate_claims(
                self.promoted_triples
            )
        
        return all_edges
    
    def create_reified_claims(self) -> List[ScientificClaim]:
        """
        Create reified claims from relationships with the same (subject, predicate, object).

        After fix #1 in semantic_extractor.py:
          relationship.source_entity = taxon name   (e.g. "staphylococcus aureus")
          relationship.target_entity = disease name (e.g. "atopic dermatitis")
          relationship.provenance.paper_id = paper DOI — the evidence source

        The reified claim therefore reads:
          subject_entity  = taxon
          predicate       = "associated_with_increased" (etc.)
          object_entity   = disease
          supporting_papers = [doi:xxx, doi:yyy, ...]   ← from provenance

        Requirements: 4.1
        """
        claims = []

        for key, relationships in self.relationship_index.items():
            if len(relationships) < 1:
                continue

            # Get unique paper IDs from PROVENANCE (not from source_entity — that's
            # the taxon now, not the paper DOI).
            paper_ids = list(set(rel.provenance.paper_id for rel in relationships))

            provenance_list = [rel.provenance for rel in relationships]

            first_rel = relationships[0]
            claim_type = self._get_claim_type(first_rel.relation_type)

            p_value = first_rel.properties.get("p_value")

            # Infer article_type from provenance where available
            article_type = first_rel.properties.get("article_type")

            try:
                claim = self.relationship_reifier.reify_claim(
                    subject=first_rel.source_entity,       # taxon name
                    predicate=self._normalize_predicate(first_rel),
                    object_entity=first_rel.target_entity, # disease name
                    supporting_evidence=provenance_list,
                    claim_type=claim_type,
                    p_value=p_value,
                    article_type=article_type,
                )
                claims.append(claim)
            except ValueError:
                continue

        self.claims = claims
        return claims
    
    def get_all_edges(self) -> List[EnhancedGraphEdge]:
        """
        Get all enhanced graph edges.
        
        Returns:
            List of all EnhancedGraphEdge objects
        """
        return self.edges
    
    def get_all_claims(self) -> List[ScientificClaim]:
        """
        Get all reified claims.
        
        Returns:
            List of all ScientificClaim objects
        """
        return self.claims
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph construction.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_relationships": len(self.relationships),
            "total_edges": len(self.edges),
            "total_claims": len(self.claims),
            "associations": sum(
                1 for r in self.relationships 
                if r.relation_type == RelationType.REPORTS_ASSOCIATION
            ),
            "interventions": sum(
                1 for r in self.relationships 
                if r.relation_type == RelationType.REPORTS_INTERVENTION_EFFECT
            ),
            "methodologies": sum(
                1 for r in self.relationships 
                if r.relation_type == RelationType.USES_METHODOLOGY
            ),
            "unique_triples": len(self.relationship_index),
            "open_world_triples": len(self.open_world_triples),
            "promoted_triples": len(self.promoted_triples),
            "open_world_claims": len(self.open_world_claims),
        }

    def get_open_world_triples(self) -> List[Dict]:
        """
        Return all open-world triples discovered by the LLMTripleExtractor.

        Each triple is a dict with keys:
          subject, subject_type, predicate, canonical_predicate, predicate_category,
          is_novel_predicate, object, object_type, confidence, evidence,
          paper_id, section_type, extracted_at

        These are intended to be loaded into Neo4j as RELATES_TO relationships
        (or their canonical_predicate if recognized) with the raw predicate and
        entity type properties preserved.

        Returns:
            List of open-world triple dicts (empty when USE_LLM != "true")
        """
        return list(self.open_world_triples)
    
    # ========== Helper Methods ==========
    
    def _create_edge_from_relationship(
        self,
        relationship: SemanticRelationship
    ) -> EnhancedGraphEdge:
        """
        Create an EnhancedGraphEdge from a SemanticRelationship.

        Normalizes source and target entities via the ontology registry before
        creating the edge, so Neo4j nodes use canonical IDs instead of raw strings.

        Requirements: 3.1, 3.2 (embed provenance)
        """
        # ── Determine entity types for routing ────────────────────────────────
        if relationship.relation_type == RelationType.REPORTS_ASSOCIATION:
            source_type = "paper"     # source is always the paper
            target_type = "taxon"
        elif relationship.relation_type == RelationType.REPORTS_INTERVENTION_EFFECT:
            source_type = "paper"
            target_type = "taxon"
        elif relationship.relation_type == RelationType.USES_METHODOLOGY:
            source_type = "paper"
            target_type = "method"
        else:
            source_type = "paper"
            target_type = "unknown"

        # ── Normalize target entity (source is a paper DOI — no grounding needed)
        target_raw = relationship.target_entity
        if target_type != "paper" and target_type != "unknown":
            # Check if the entity was already grounded inline at Layer 2
            # by looking it up in the paper's entities list
            pre_grounded = self._find_grounded_entity(relationship.target_entity, target_type)
            if pre_grounded and pre_grounded.get("grounded"):
                # Use the pre-grounded result — no API call needed
                grounded = pre_grounded
                target_id = grounded.get("id") or f"ungrounded:{target_raw.lower()}"
                target_canonical = grounded.get("canonical_name") or target_raw
            else:
                # Fall back to normalizer (handles cache miss or Layer 2 without grounding)
                grounded = self.entity_normalizer.normalize(target_raw, target_type)
                target_id = grounded.get("id") or f"ungrounded:{target_raw.lower()}"
                target_canonical = grounded.get("canonical_name") or target_raw
        else:
            target_id = target_raw
            target_canonical = target_raw
            grounded = {"grounded": False, "confidence": 0.0, "source": "none", "ontology": None}

        # ── Build properties with grounding metadata ───────────────────────────
        props = relationship.properties.copy()
        props["target_canonical"] = target_canonical
        props["target_ontology_id"] = grounded.get("id")
        props["target_ontology"] = grounded.get("ontology")
        props["target_grounded"] = grounded.get("grounded", False)
        props["target_grounding_confidence"] = grounded.get("confidence", 0.0)
        props["target_grounding_source"] = grounded.get("source", "none")

        return EnhancedGraphEdge(
            source=relationship.source_entity,
            target=target_id,            # canonical ontology ID
            relation=relationship.relation_type.value,
            properties=props,
            provenance=relationship.provenance,
            evidence_strength=relationship.evidence_strength,
            confidence=relationship.extraction_confidence
        )
    
    def _get_relationship_key(
        self,
        relationship: SemanticRelationship
    ) -> Tuple[str, str, str]:
        """
        Get a key for indexing relationships by (subject, predicate, object).

        For REPORTS_ASSOCIATION:
          subject = taxon (source_entity)   — e.g. "faecalibacterium prausnitzii"
          object  = disease (target_entity) — e.g. "inflammatory bowel disease"

        For USES_METHODOLOGY:
          subject = paper DOI (source_entity)
          object  = method name (target_entity)

        The canonical ontology ID is preferred for the object key when available
        so that "h. pylori" and "Helicobacter pylori" merge into the same claim.

        Args:
            relationship: Semantic relationship

        Returns:
            Tuple of (subject, predicate, object)
        """
        predicate = self._normalize_predicate(relationship)
        # Use grounded canonical ontology ID as the target key if available,
        # so spelling variants of the same entity merge into one claim.
        canonical_target = (
            relationship.properties.get("target_ontology_id")
            or relationship.target_entity
        )
        return (
            relationship.source_entity,  # taxon for associations, paper for methodology
            predicate,
            canonical_target,            # disease for associations, method for methodology
        )
    
    def _normalize_predicate(self, relationship: SemanticRelationship) -> str:
        """
        Normalize predicate from relationship properties.
        
        For associations, include direction in predicate.
        For interventions, include intervention type and effect direction.
        For methodology, use relation type.
        
        Args:
            relationship: Semantic relationship
        
        Returns:
            Normalized predicate string
        """
        if relationship.relation_type == RelationType.REPORTS_ASSOCIATION:
            direction = relationship.properties.get("direction", "unknown")
            return f"associated_with_{direction}"
        
        elif relationship.relation_type == RelationType.REPORTS_INTERVENTION_EFFECT:
            intervention = relationship.properties.get("intervention_type", "unknown")
            direction = relationship.properties.get("effect_direction", "unknown")
            return f"{intervention}_effect_{direction}"
        
        elif relationship.relation_type == RelationType.USES_METHODOLOGY:
            return "uses_methodology"
        
        return relationship.relation_type.value
    
    def _get_claim_type(self, relation_type: RelationType) -> str:
        """
        Map RelationType to claim type.
        """
        if relation_type == RelationType.REPORTS_ASSOCIATION:
            return "association"
        elif relation_type == RelationType.REPORTS_INTERVENTION_EFFECT:
            return "intervention_effect"
        elif relation_type == RelationType.USES_METHODOLOGY:
            return "methodology_comparison"
        return "unknown"

    def _find_grounded_entity(self, entity_text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Look up pre-grounded entity data from the current paper's entities list.
        Returns a grounding dict if found and grounded, else None.
        """
        text_lower = entity_text.lower()
        for ent in self._current_paper_entities:
            if (ent.text.lower() == text_lower and
                    ent.label == entity_type and
                    ent.grounded):
                return {
                    "id": ent.ontology_id,
                    "canonical_name": ent.canonical_name,
                    "ontology": ent.ontology_name,
                    "grounded": ent.grounded,
                    "confidence": ent.grounding_confidence or 0.0,
                    "source": ent.grounding_source or "none",
                }
        return None
