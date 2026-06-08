"""
graph/error_handler.py
----------------------
Error handling and recovery mechanisms for the knowledge graph system.

This module implements graceful error handling for:
1. Extraction failures (incomplete provenance)
2. Conflicting statistical measures
3. Entity normalization failures
4. Query timeouts
5. Conflicting claims

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict

from graph.semantic_relationships import SemanticRelationship
from graph.reified_claims import ScientificClaim
from graph.provenance import ProvenanceMetadata

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Handles errors gracefully and provides recovery mechanisms.
    
    This class manages:
    - Incomplete extraction queue for papers with missing provenance
    - Conflicting statistics flagging for papers with multiple measures
    - Curator review queue for ungrounded entities
    - Query timeout handling with partial results
    - Conflicting claims with CONFLICTS_WITH relationships
    
    Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
    """
    
    def __init__(
        self,
        incomplete_extraction_queue_path: Optional[str] = None,
        curator_review_queue_path: Optional[str] = None,
        query_log_path: Optional[str] = None
    ):
        """
        Initialize the error handler.
        
        Args:
            incomplete_extraction_queue_path: Path to store papers with incomplete extraction
            curator_review_queue_path: Path to store ungrounded entities for curator review
            query_log_path: Path to log query patterns for optimization
        """
        # Set default paths
        if incomplete_extraction_queue_path is None:
            incomplete_extraction_queue_path = "data/incomplete_extraction_queue.json"
        if curator_review_queue_path is None:
            curator_review_queue_path = "data/curator_review_queue.json"
        if query_log_path is None:
            query_log_path = "data/query_timeout_log.json"
        
        self.incomplete_extraction_queue_path = Path(incomplete_extraction_queue_path)
        self.curator_review_queue_path = Path(curator_review_queue_path)
        self.query_log_path = Path(query_log_path)
        
        # Ensure parent directories exist
        self.incomplete_extraction_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.curator_review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.query_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized ErrorHandler with queues at:\n"
            f"  - Incomplete extraction: {self.incomplete_extraction_queue_path}\n"
            f"  - Curator review: {self.curator_review_queue_path}\n"
            f"  - Query timeout log: {self.query_log_path}"
        )
    
    # ========== Requirement 15.1: Extraction Failure Handling ==========
    
    def handle_extraction_failure(
        self,
        paper_id: str,
        paper_title: Optional[str] = None,
        failure_reason: str = "Missing provenance data",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle extraction failures by logging warning and adding to incomplete_extraction queue.
        
        Requirement 15.1: WHEN extraction fails to capture provenance data,
        THE System SHALL log a warning and add the paper to an "incomplete_extraction"
        queue without creating a graph edge.
        
        Args:
            paper_id: Identifier of the paper (DOI, PMID, or title)
            paper_title: Title of the paper (optional)
            failure_reason: Reason for extraction failure
            details: Additional details about the failure
        """
        # Log warning
        logger.warning(
            f"Extraction failure for paper {paper_id}: {failure_reason}"
        )
        
        # Load existing queue
        existing_queue = self._load_json_queue(self.incomplete_extraction_queue_path)
        
        # Create entry
        entry = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "failure_reason": failure_reason,
            "details": details or {},
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_review"
        }
        
        # Add to queue
        existing_queue.append(entry)
        
        # Save queue
        self._save_json_queue(self.incomplete_extraction_queue_path, existing_queue)
        
        logger.info(
            f"Added paper {paper_id} to incomplete_extraction queue "
            f"(queue size: {len(existing_queue)})"
        )
    
    def validate_provenance_completeness(
        self,
        provenance: ProvenanceMetadata
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that provenance metadata is complete.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if not provenance.paper_id or not provenance.paper_id.strip():
            return False, "Missing paper_id"
        
        if not provenance.section_type or not provenance.section_type.strip():
            return False, "Missing section_type"
        
        if not provenance.source_sentence or not provenance.source_sentence.strip():
            return False, "Missing source_sentence"
        
        if not provenance.extraction_method or not provenance.extraction_method.strip():
            return False, "Missing extraction_method"
        
        if provenance.extraction_timestamp is None:
            return False, "Missing extraction_timestamp"
        
        if not (0.0 <= provenance.confidence_score <= 1.0):
            return False, f"Invalid confidence_score: {provenance.confidence_score}"
        
        return True, None
    
    # ========== Requirement 15.2: Conflicting Statistics Handling ==========
    
    def handle_conflicting_statistics(
        self,
        paper_id: str,
        relationships: List[SemanticRelationship],
        conflict_details: Optional[Dict[str, Any]] = None
    ) -> List[SemanticRelationship]:
        """
        Handle conflicting statistical measures by creating separate edges and flagging paper.
        
        Requirement 15.2: WHEN multiple conflicting statistical measures are found
        in the same paper, THE System SHALL create separate edges for each distinct
        claim and flag the paper with "conflicting_statistics".
        
        Args:
            paper_id: Identifier of the paper
            relationships: List of relationships with conflicting statistics
            conflict_details: Additional details about the conflicts
        
        Returns:
            List of relationships (unchanged, all will be created as separate edges)
        """
        # Log warning
        logger.warning(
            f"Conflicting statistics found in paper {paper_id}: "
            f"{len(relationships)} distinct claims"
        )
        
        # Flag the paper by adding a property to each relationship
        for rel in relationships:
            if not hasattr(rel, 'properties'):
                rel.properties = {}
            rel.properties["conflicting_statistics"] = True
            rel.properties["conflict_group_size"] = len(relationships)
        
        # Log the conflict for tracking
        conflict_entry = {
            "paper_id": paper_id,
            "relationship_count": len(relationships),
            "conflict_details": conflict_details or {},
            "detected_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Append to a conflicts log (separate from queues)
        conflicts_log_path = self.incomplete_extraction_queue_path.parent / "conflicting_statistics_log.json"
        existing_log = self._load_json_queue(conflicts_log_path)
        existing_log.append(conflict_entry)
        self._save_json_queue(conflicts_log_path, existing_log)
        
        logger.info(
            f"Flagged paper {paper_id} with conflicting_statistics. "
            f"Creating {len(relationships)} separate edges."
        )
        
        # Return all relationships - they will be created as separate edges
        return relationships
    
    def detect_conflicting_statistics(
        self,
        relationships: List[SemanticRelationship]
    ) -> Dict[str, List[SemanticRelationship]]:
        """
        Detect relationships with conflicting statistical measures.
        
        Groups relationships by (source, target, relation_type) and identifies
        groups with multiple distinct statistical measures.
        
        Args:
            relationships: List of relationships to analyze
        
        Returns:
            Dictionary mapping paper_id to list of conflicting relationships
        """
        # Group by paper_id and (source, target, relation_type)
        paper_groups = defaultdict(lambda: defaultdict(list))
        
        for rel in relationships:
            key = (rel.source_entity, rel.target_entity, rel.relation_type.value)
            paper_groups[rel.provenance.paper_id][key].append(rel)
        
        # Find conflicts
        conflicts = {}
        
        for paper_id, groups in paper_groups.items():
            for key, rels in groups.items():
                if len(rels) > 1:
                    # Check if they have different statistical measures
                    measures = set()
                    for rel in rels:
                        # Extract key statistical properties
                        p_value = rel.properties.get("p_value")
                        effect_size = rel.properties.get("effect_size")
                        direction = rel.properties.get("direction")
                        
                        measure_tuple = (p_value, effect_size, direction)
                        measures.add(measure_tuple)
                    
                    # If multiple distinct measures, it's a conflict
                    if len(measures) > 1:
                        if paper_id not in conflicts:
                            conflicts[paper_id] = []
                        conflicts[paper_id].extend(rels)
        
        return conflicts
    
    # ========== Requirement 15.3: Entity Normalization Failure Handling ==========
    
    def handle_entity_normalization_failure(
        self,
        entity_text: str,
        entity_type: str,
        failure_reason: str,
        temporary_id: str
    ) -> None:
        """
        Handle entity normalization failures by adding to curator review queue.
        
        Requirement 15.3: WHEN entity normalization fails, THE System SHALL
        create an "ungrounded" node with temporary ID and add to curator review queue.
        
        Note: The entity_normalizer.py already creates ungrounded nodes and logs
        failures to entity_normalization_failures table. This method provides
        additional curator review queue functionality.
        
        Args:
            entity_text: Original entity text
            entity_type: Type of entity (taxon, disease, etc.)
            failure_reason: Reason for normalization failure
            temporary_id: Temporary ID assigned to ungrounded node
        """
        # Load existing queue
        existing_queue = self._load_json_queue(self.curator_review_queue_path)
        
        # Create entry
        entry = {
            "entity_text": entity_text,
            "entity_type": entity_type,
            "temporary_id": temporary_id,
            "failure_reason": failure_reason,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_review",
            "curator_notes": None
        }
        
        # Check if already in queue
        existing_ids = {e["temporary_id"] for e in existing_queue}
        if temporary_id not in existing_ids:
            existing_queue.append(entry)
            
            # Save queue
            self._save_json_queue(self.curator_review_queue_path, existing_queue)
            
            logger.info(
                f"Added ungrounded entity '{entity_text}' ({entity_type}) "
                f"to curator review queue (queue size: {len(existing_queue)})"
            )
        else:
            logger.debug(
                f"Entity '{entity_text}' already in curator review queue"
            )
    
    # ========== Requirement 15.4: Query Timeout Handling ==========
    
    def handle_query_timeout(
        self,
        query_description: str,
        query_params: Dict[str, Any],
        partial_results: List[Dict[str, Any]],
        execution_time_ms: float,
        timeout_threshold_ms: float = 30000
    ) -> Dict[str, Any]:
        """
        Handle query timeouts by returning partial results with timeout flag.
        
        Requirement 15.4: WHEN a query times out, THE System SHALL cancel execution,
        return partial results with timeout flag, and log the query pattern for optimization.
        
        Args:
            query_description: Description of the query
            query_params: Query parameters
            partial_results: Partial results obtained before timeout
            execution_time_ms: Actual execution time in milliseconds
            timeout_threshold_ms: Timeout threshold (default: 30000ms = 30s)
        
        Returns:
            Dictionary with partial results and timeout metadata
        """
        # Log warning
        logger.warning(
            f"Query timeout after {execution_time_ms:.2f}ms "
            f"(threshold: {timeout_threshold_ms}ms): {query_description}"
        )
        
        # Log query pattern for optimization
        log_entry = {
            "query_description": query_description,
            "query_params": query_params,
            "execution_time_ms": execution_time_ms,
            "timeout_threshold_ms": timeout_threshold_ms,
            "partial_result_count": len(partial_results),
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "timeout"
        }
        
        # Append to query log
        existing_log = self._load_json_queue(self.query_log_path)
        existing_log.append(log_entry)
        self._save_json_queue(self.query_log_path, existing_log)
        
        logger.info(
            f"Logged timeout query pattern for optimization. "
            f"Returning {len(partial_results)} partial results."
        )
        
        # Return partial results with timeout flag
        return {
            "results": partial_results,
            "result_count": len(partial_results),
            "timeout": True,
            "execution_time_ms": execution_time_ms,
            "timeout_threshold_ms": timeout_threshold_ms,
            "message": f"Query timed out after {execution_time_ms:.2f}ms. Returning partial results."
        }
    
    # ========== Requirement 15.5: Conflicting Claims Handling ==========
    
    def handle_conflicting_claims(
        self,
        existing_claim: ScientificClaim,
        new_claim: ScientificClaim
    ) -> Tuple[ScientificClaim, ScientificClaim, Dict[str, Any]]:
        """
        Handle conflicting claims by creating separate claims with CONFLICTS_WITH relationship.
        
        Requirement 15.5: WHEN attempting to create a reified claim with opposite
        predicate to an existing claim, THE System SHALL create separate claims
        and link them with CONFLICTS_WITH relationship.
        
        Args:
            existing_claim: Existing scientific claim
            new_claim: New claim with opposite predicate
        
        Returns:
            Tuple of (existing_claim, new_claim, conflict_relationship)
        """
        # Log the conflict
        logger.warning(
            f"Conflicting claims detected:\n"
            f"  Existing: {existing_claim.subject_entity} "
            f"{existing_claim.predicate} {existing_claim.object_entity}\n"
            f"  New: {new_claim.subject_entity} "
            f"{new_claim.predicate} {new_claim.object_entity}"
        )
        
        # Create CONFLICTS_WITH relationship metadata
        conflict_relationship = {
            "relationship_type": "CONFLICTS_WITH",
            "source_claim_id": existing_claim.claim_id,
            "target_claim_id": new_claim.claim_id,
            "conflict_type": "opposite_predicate",
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "subject_entity": existing_claim.subject_entity,
            "object_entity": existing_claim.object_entity,
            "existing_predicate": existing_claim.predicate,
            "new_predicate": new_claim.predicate,
            "existing_supporting_papers": len(existing_claim.supporting_papers),
            "new_supporting_papers": len(new_claim.supporting_papers)
        }
        
        logger.info(
            f"Created CONFLICTS_WITH relationship between claims "
            f"{existing_claim.claim_id} and {new_claim.claim_id}"
        )
        
        return existing_claim, new_claim, conflict_relationship
    
    def detect_opposite_predicates(
        self,
        predicate1: str,
        predicate2: str
    ) -> bool:
        """
        Detect if two predicates are opposite.
        
        Args:
            predicate1: First predicate
            predicate2: Second predicate
        
        Returns:
            True if predicates are opposite, False otherwise
        """
        pred1 = predicate1.lower()
        pred2 = predicate2.lower()
        
        # Check for increased/decreased opposition
        if ("increased" in pred1 and "decreased" in pred2) or \
           ("decreased" in pred1 and "increased" in pred2):
            return True
        
        # Check for positive/negative opposition
        if ("positive" in pred1 and "negative" in pred2) or \
           ("negative" in pred1 and "positive" in pred2):
            return True
        
        # Check for up/down opposition
        if ("up" in pred1 and "down" in pred2) or \
           ("down" in pred1 and "up" in pred2):
            return True
        
        # Check for higher/lower opposition
        if ("higher" in pred1 and "lower" in pred2) or \
           ("lower" in pred1 and "higher" in pred2):
            return True
        
        return False
    
    # ========== Helper Methods ==========
    
    def _load_json_queue(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON queue from file."""
        if not path.exists():
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load queue from {path}: {e}")
            return []
    
    def _save_json_queue(self, path: Path, queue: List[Dict[str, Any]]) -> None:
        """Save JSON queue to file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save queue to {path}: {e}")
            raise
    
    # ========== Queue Statistics ==========
    
    def get_incomplete_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the incomplete extraction queue."""
        queue = self._load_json_queue(self.incomplete_extraction_queue_path)
        
        # Count by failure reason
        reason_counts = defaultdict(int)
        for entry in queue:
            reason = entry.get("failure_reason", "unknown")
            reason_counts[reason] += 1
        
        return {
            "queue_size": len(queue),
            "queue_path": str(self.incomplete_extraction_queue_path),
            "exists": self.incomplete_extraction_queue_path.exists(),
            "failure_reason_counts": dict(reason_counts)
        }
    
    def get_curator_review_stats(self) -> Dict[str, Any]:
        """Get statistics about the curator review queue."""
        queue = self._load_json_queue(self.curator_review_queue_path)
        
        # Count by entity type
        type_counts = defaultdict(int)
        for entry in queue:
            entity_type = entry.get("entity_type", "unknown")
            type_counts[entity_type] += 1
        
        return {
            "queue_size": len(queue),
            "queue_path": str(self.curator_review_queue_path),
            "exists": self.curator_review_queue_path.exists(),
            "entity_type_counts": dict(type_counts)
        }
    
    def get_query_timeout_stats(self) -> Dict[str, Any]:
        """Get statistics about query timeouts."""
        log = self._load_json_queue(self.query_log_path)
        
        if not log:
            return {
                "timeout_count": 0,
                "log_path": str(self.query_log_path),
                "exists": self.query_log_path.exists()
            }
        
        # Calculate statistics
        timeout_count = len(log)
        avg_execution_time = sum(e["execution_time_ms"] for e in log) / timeout_count
        
        # Count by query description
        query_counts = defaultdict(int)
        for entry in log:
            query_desc = entry.get("query_description", "unknown")
            query_counts[query_desc] += 1
        
        return {
            "timeout_count": timeout_count,
            "log_path": str(self.query_log_path),
            "exists": self.query_log_path.exists(),
            "avg_execution_time_ms": avg_execution_time,
            "query_description_counts": dict(query_counts)
        }
