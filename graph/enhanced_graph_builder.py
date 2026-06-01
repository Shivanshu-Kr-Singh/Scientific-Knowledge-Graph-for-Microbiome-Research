"""
graph/enhanced_graph_builder.py
--------------------------------
Enhanced graph builder integrating semantic extractor and relationship reifier.

This module creates the enhanced knowledge graph by:
1. Extracting rich semantic relationships from papers using SemanticRelationshipExtractor
2. Creating reified claims from multiple papers using RelationshipReifier
3. Building EnhancedGraphEdge objects with embedded provenance

Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 4.1
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict

from nlp.enriched_record import EnrichedPaperRecord
from graph.semantic_extractor import SemanticRelationshipExtractor
from graph.relationship_reifier import RelationshipReifier
from graph.semantic_relationships import SemanticRelationship, RelationType
from graph.reified_claims import ScientificClaim
from graph.provenance import ProvenanceMetadata


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
        
        # Storage for relationships and claims
        self.relationships: List[SemanticRelationship] = []
        self.edges: List[EnhancedGraphEdge] = []
        self.claims: List[ScientificClaim] = []
        
        # Index for grouping relationships by (subject, predicate, object)
        self.relationship_index: Dict[Tuple[str, str, str], List[SemanticRelationship]] = defaultdict(list)
    
    def process_paper(self, paper: EnrichedPaperRecord) -> List[EnhancedGraphEdge]:
        """
        Process a single paper and extract all relationships.
        """
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

        self.edges.extend(paper_edges)
        return paper_edges

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
        
        return all_edges
    
    def create_reified_claims(self) -> List[ScientificClaim]:
        """
        Create reified claims from relationships with the same (subject, predicate, object).
        
        This method aggregates evidence from multiple papers into scientific claims
        with consensus metrics and evidence strength classification.
        
        Requirement 4.1: Create reified claim nodes aggregating supporting evidence
        
        Returns:
            List of ScientificClaim objects
        """
        claims = []
        
        # For each unique (subject, predicate, object) triple
        for key, relationships in self.relationship_index.items():
            # Only create claims for relationships appearing in multiple papers
            # or with high confidence
            if len(relationships) < 1:
                continue
            
            # Get unique paper IDs
            paper_ids = list(set(rel.provenance.paper_id for rel in relationships))
            
            # Extract provenance metadata
            provenance_list = [rel.provenance for rel in relationships]
            
            # Determine claim type based on relation type
            first_rel = relationships[0]
            claim_type = self._get_claim_type(first_rel.relation_type)
            
            # Extract p-value and article type if available
            p_value = None
            article_type = None
            
            # Try to get p-value from properties
            if "p_value" in first_rel.properties:
                p_value = first_rel.properties["p_value"]
            
            # Create reified claim
            try:
                claim = self.relationship_reifier.reify_claim(
                    subject=first_rel.source_entity,
                    predicate=self._normalize_predicate(first_rel),
                    object_entity=first_rel.target_entity,
                    supporting_evidence=provenance_list,
                    claim_type=claim_type,
                    p_value=p_value,
                    article_type=article_type
                )
                claims.append(claim)
            except ValueError as e:
                # Skip invalid claims
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
        }
    
    # ========== Helper Methods ==========
    
    def _create_edge_from_relationship(
        self,
        relationship: SemanticRelationship
    ) -> EnhancedGraphEdge:
        """
        Create an EnhancedGraphEdge from a SemanticRelationship.
        
        Requirements: 3.1, 3.2 (embed provenance)
        
        Args:
            relationship: Semantic relationship with provenance
        
        Returns:
            EnhancedGraphEdge with embedded provenance
        """
        return EnhancedGraphEdge(
            source=relationship.source_entity,
            target=relationship.target_entity,
            relation=relationship.relation_type.value,
            properties=relationship.properties.copy(),
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
        
        Args:
            relationship: Semantic relationship
        
        Returns:
            Tuple of (subject, predicate, object)
        """
        predicate = self._normalize_predicate(relationship)
        return (
            relationship.source_entity,
            predicate,
            relationship.target_entity
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
        
        Args:
            relation_type: Relationship type
        
        Returns:
            Claim type string
        """
        if relation_type == RelationType.REPORTS_ASSOCIATION:
            return "association"
        elif relation_type == RelationType.REPORTS_INTERVENTION_EFFECT:
            return "intervention_effect"
        elif relation_type == RelationType.USES_METHODOLOGY:
            return "methodology_comparison"
        return "unknown"
