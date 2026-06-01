"""
EntityMerger — atomic Neo4j merge operations with relationship deduplication and rollback.

Merge algorithm:
  1. Verify source and target nodes have the same entity_type (reject if different)
  2. Transfer all inbound and outbound relationships from source to target
  3. Deduplicate: if a transferred relationship duplicates an existing one
     (same type, same counterpart, same direction), keep the higher-confidence one
  4. Delete the source node
  5. Write MergeLogEntry to audit log

If any step fails: roll back all changes, write MergeRollbackEntry, return error.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from neo4j import GraphDatabase, Driver, Transaction

from entity_resolution.models import MergeLogEntry, MergeRollbackEntry

logger = logging.getLogger(__name__)


class EntityMerger:
    """
    Atomic Neo4j merge operations with rollback.

    Merge algorithm:
    1. Verify source and target nodes have the same entity_type (reject if different)
    2. Transfer all inbound and outbound relationships from source to target
    3. Deduplicate: if a transferred relationship duplicates an existing one
       (same type, same counterpart, same direction), keep the higher-confidence one
    4. Delete the source node
    5. Write MergeLogEntry to audit log

    If any step fails: roll back all changes, write MergeRollbackEntry, return error.

    Preconditions for merge():
    - source_node_id and target_canonical_id exist in Neo4j
    - Both nodes have the same entity_type

    Postconditions for merge():
    - On success: source node deleted, all relationships on target node
    - On failure: graph is in pre-merge state (rollback applied)
    - MergeLogEntry or MergeRollbackEntry written in all cases

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Connect to Neo4j.

        Parameters
        ----------
        uri:      Neo4j bolt URI, e.g. "bolt://localhost:7687"
        user:     Neo4j username
        password: Neo4j password
        """
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        # In-memory audit log accessible for testing
        self.merge_log: List[MergeLogEntry] = []
        self.rollback_log: List[MergeRollbackEntry] = []

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_canonical_node(
        self,
        canonical_id: str,
        entity_type: str,
        primary_name: str,
    ) -> str:
        """
        Get or create the canonical node in Neo4j.

        Uses MERGE on the ``canonical_id`` property to guarantee idempotency.
        Returns the Neo4j internal node element ID (string).

        Requirements: 6.1
        """
        with self._driver.session() as session:
            result = session.execute_write(
                self._merge_canonical_node,
                canonical_id,
                entity_type,
                primary_name,
            )
        return result

    def merge(
        self,
        source_node_id: str,
        target_canonical_id: str,
        triggering_surface_form: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Merge source node into target canonical node.

        Steps (all inside a single explicit transaction):
          1. Verify entity_type match — reject with type-conflict log if different
          2. Get all relationships of source node
          3. Transfer relationships to target, deduplicating
          4. Delete source node
          5. Write MergeLogEntry

        On any exception: rollback, write MergeRollbackEntry.

        Returns (success, error_message_or_None).

        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                # ----------------------------------------------------------------
                # Step 1: Verify entity_type match
                # ----------------------------------------------------------------
                source_type, target_type = self._get_entity_types(
                    tx, source_node_id, target_canonical_id
                )

                if source_type != target_type:
                    # Requirement 6.5: reject merge of different entity types
                    error_msg = (
                        f"Type conflict: source node '{source_node_id}' has "
                        f"entity_type='{source_type}', target canonical node "
                        f"'{target_canonical_id}' has entity_type='{target_type}'. "
                        f"Merge rejected."
                    )
                    logger.error(
                        "EntityMerger type conflict: source_node_id=%s "
                        "source_type=%s target_canonical_id=%s target_type=%s",
                        source_node_id,
                        source_type,
                        target_canonical_id,
                        target_type,
                    )
                    tx.rollback()
                    rollback_entry = MergeRollbackEntry(
                        source_node_ids=[source_node_id],
                        target_canonical_id=target_canonical_id,
                        failed_step="type_check",
                        error_message=error_msg,
                        timestamp=datetime.now(timezone.utc),
                    )
                    self.rollback_log.append(rollback_entry)
                    return False, error_msg

                # ----------------------------------------------------------------
                # Step 2: Get all relationships of source node
                # ----------------------------------------------------------------
                source_rels = self._get_all_relationships(tx, source_node_id)

                # ----------------------------------------------------------------
                # Step 3: Transfer relationships to target, deduplicating
                # ----------------------------------------------------------------
                transferred, deduplicated = self._transfer_relationships(
                    tx, source_node_id, target_canonical_id, source_rels
                )

                # ----------------------------------------------------------------
                # Step 4: Delete source node (relationships already removed above)
                # ----------------------------------------------------------------
                self._delete_node(tx, source_node_id)

                # ----------------------------------------------------------------
                # Step 5: Write MergeLogEntry
                # ----------------------------------------------------------------
                log_entry = MergeLogEntry(
                    source_node_ids=[source_node_id],
                    target_canonical_id=target_canonical_id,
                    triggering_resolution=triggering_surface_form,
                    timestamp=datetime.now(timezone.utc),
                    relationships_transferred=transferred,
                    relationships_deduplicated=deduplicated,
                )
                self.merge_log.append(log_entry)
                logger.info(
                    "EntityMerger merge success: source=%s target=%s "
                    "transferred=%d deduplicated=%d",
                    source_node_id,
                    target_canonical_id,
                    transferred,
                    deduplicated,
                )

                tx.commit()
                return True, None

            except Exception as exc:  # noqa: BLE001
                # Requirement 6.6: rollback on any failure
                try:
                    tx.rollback()
                except Exception:  # noqa: BLE001
                    pass  # best-effort rollback

                error_msg = str(exc)
                # Determine which step failed based on exception context
                failed_step = getattr(exc, "_merger_step", "unknown")

                rollback_entry = MergeRollbackEntry(
                    source_node_ids=[source_node_id],
                    target_canonical_id=target_canonical_id,
                    failed_step=failed_step,
                    error_message=error_msg,
                    timestamp=datetime.now(timezone.utc),
                )
                self.rollback_log.append(rollback_entry)
                logger.error(
                    "EntityMerger merge failed and rolled back: source=%s "
                    "target=%s step=%s error=%s",
                    source_node_id,
                    target_canonical_id,
                    failed_step,
                    error_msg,
                )
                return False, error_msg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_canonical_node(
        tx: Transaction,
        canonical_id: str,
        entity_type: str,
        primary_name: str,
    ) -> str:
        """
        MERGE a canonical node on ``canonical_id`` property.
        Returns the Neo4j element ID of the node.
        """
        result = tx.run(
            """
            MERGE (n:Entity {canonical_id: $canonical_id})
            ON CREATE SET
                n.entity_type  = $entity_type,
                n.primary_name = $primary_name
            RETURN elementId(n) AS node_id
            """,
            canonical_id=canonical_id,
            entity_type=entity_type,
            primary_name=primary_name,
        )
        record = result.single()
        if record is None:
            raise RuntimeError(
                f"ensure_canonical_node: MERGE returned no record for "
                f"canonical_id='{canonical_id}'"
            )
        return record["node_id"]

    @staticmethod
    def _get_entity_types(
        tx: Transaction,
        source_node_id: str,
        target_canonical_id: str,
    ) -> Tuple[str, str]:
        """
        Retrieve entity_type for both source (by elementId) and target (by canonical_id).

        Raises RuntimeError if either node is not found.
        """
        # Source node — looked up by Neo4j element ID
        src_result = tx.run(
            """
            MATCH (n)
            WHERE elementId(n) = $node_id
            RETURN n.entity_type AS entity_type
            """,
            node_id=source_node_id,
        )
        src_record = src_result.single()
        if src_record is None:
            err = RuntimeError(
                f"Source node with elementId='{source_node_id}' not found in Neo4j"
            )
            err._merger_step = "type_check"  # type: ignore[attr-defined]
            raise err
        source_type: str = src_record["entity_type"]

        # Target node — looked up by canonical_id property
        tgt_result = tx.run(
            """
            MATCH (n:Entity {canonical_id: $canonical_id})
            RETURN n.entity_type AS entity_type
            """,
            canonical_id=target_canonical_id,
        )
        tgt_record = tgt_result.single()
        if tgt_record is None:
            err = RuntimeError(
                f"Target canonical node with canonical_id='{target_canonical_id}' "
                f"not found in Neo4j"
            )
            err._merger_step = "type_check"  # type: ignore[attr-defined]
            raise err
        target_type: str = tgt_record["entity_type"]

        return source_type, target_type

    @staticmethod
    def _get_all_relationships(
        tx: Transaction,
        source_node_id: str,
    ) -> list:
        """
        Return all relationships (inbound and outbound) for the source node.

        Each entry is a dict with keys:
          rel_id, rel_type, direction ("outbound"|"inbound"),
          counterpart_id (elementId of the other node),
          confidence (float, defaults to 0.0 if not set)
        """
        result = tx.run(
            """
            MATCH (src)
            WHERE elementId(src) = $node_id
            OPTIONAL MATCH (src)-[r]->(other)
            RETURN
                elementId(r)     AS rel_id,
                type(r)          AS rel_type,
                'outbound'       AS direction,
                elementId(other) AS counterpart_id,
                coalesce(r.confidence, 0.0) AS confidence
            UNION ALL
            MATCH (src)
            WHERE elementId(src) = $node_id
            OPTIONAL MATCH (other)-[r]->(src)
            RETURN
                elementId(r)     AS rel_id,
                type(r)          AS rel_type,
                'inbound'        AS direction,
                elementId(other) AS counterpart_id,
                coalesce(r.confidence, 0.0) AS confidence
            """,
            node_id=source_node_id,
        )
        rels = []
        for record in result:
            if record["rel_id"] is None:
                continue  # OPTIONAL MATCH returned no relationship
            rels.append(
                {
                    "rel_id": record["rel_id"],
                    "rel_type": record["rel_type"],
                    "direction": record["direction"],
                    "counterpart_id": record["counterpart_id"],
                    "confidence": float(record["confidence"]),
                }
            )
        return rels

    @staticmethod
    def _transfer_relationships(
        tx: Transaction,
        source_node_id: str,
        target_canonical_id: str,
        source_rels: list,
    ) -> Tuple[int, int]:
        """
        Transfer all source relationships to the target canonical node,
        deduplicating where the same (type, counterpart, direction) already
        exists on the target — keeping the higher-confidence relationship.

        Returns (transferred_count, deduplicated_count).

        Requirements: 6.3
        """
        if not source_rels:
            return 0, 0

        # Fetch existing relationships on the target node so we can deduplicate
        existing_result = tx.run(
            """
            MATCH (tgt:Entity {canonical_id: $canonical_id})
            OPTIONAL MATCH (tgt)-[r]->(other)
            RETURN
                elementId(r)     AS rel_id,
                type(r)          AS rel_type,
                'outbound'       AS direction,
                elementId(other) AS counterpart_id,
                coalesce(r.confidence, 0.0) AS confidence
            UNION ALL
            MATCH (tgt:Entity {canonical_id: $canonical_id})
            OPTIONAL MATCH (other)-[r]->(tgt)
            RETURN
                elementId(r)     AS rel_id,
                type(r)          AS rel_type,
                'inbound'        AS direction,
                elementId(other) AS counterpart_id,
                coalesce(r.confidence, 0.0) AS confidence
            """,
            canonical_id=target_canonical_id,
        )

        # Build a map: (rel_type, counterpart_id, direction) -> {rel_id, confidence}
        existing_map: dict = {}
        for record in existing_result:
            if record["rel_id"] is None:
                continue
            key = (record["rel_type"], record["counterpart_id"], record["direction"])
            existing_map[key] = {
                "rel_id": record["rel_id"],
                "confidence": float(record["confidence"]),
            }

        transferred = 0
        deduplicated = 0

        for rel in source_rels:
            key = (rel["rel_type"], rel["counterpart_id"], rel["direction"])
            src_confidence = rel["confidence"]

            if key in existing_map:
                # Duplicate detected — keep higher confidence
                existing_confidence = existing_map[key]["confidence"]
                if src_confidence > existing_confidence:
                    # Replace existing with source (higher confidence)
                    existing_rel_id = existing_map[key]["rel_id"]
                    EntityMerger._delete_relationship_by_id(tx, existing_rel_id)
                    EntityMerger._create_relationship(
                        tx,
                        target_canonical_id,
                        rel["rel_type"],
                        rel["counterpart_id"],
                        rel["direction"],
                        src_confidence,
                    )
                    existing_map[key] = {
                        "rel_id": None,  # newly created, id unknown
                        "confidence": src_confidence,
                    }
                # else: existing has higher or equal confidence — discard source rel
                deduplicated += 1
            else:
                # No duplicate — create new relationship on target
                EntityMerger._create_relationship(
                    tx,
                    target_canonical_id,
                    rel["rel_type"],
                    rel["counterpart_id"],
                    rel["direction"],
                    src_confidence,
                )
                existing_map[key] = {"rel_id": None, "confidence": src_confidence}
                transferred += 1

        return transferred, deduplicated

    @staticmethod
    def _create_relationship(
        tx: Transaction,
        canonical_id: str,
        rel_type: str,
        counterpart_element_id: str,
        direction: str,
        confidence: float,
    ) -> None:
        """Create a relationship between the canonical node and a counterpart node."""
        if direction == "outbound":
            tx.run(
                f"""
                MATCH (tgt:Entity {{canonical_id: $canonical_id}})
                MATCH (other) WHERE elementId(other) = $counterpart_id
                MERGE (tgt)-[r:`{rel_type}`]->(other)
                SET r.confidence = $confidence
                """,
                canonical_id=canonical_id,
                counterpart_id=counterpart_element_id,
                confidence=confidence,
            )
        else:  # inbound
            tx.run(
                f"""
                MATCH (tgt:Entity {{canonical_id: $canonical_id}})
                MATCH (other) WHERE elementId(other) = $counterpart_id
                MERGE (other)-[r:`{rel_type}`]->(tgt)
                SET r.confidence = $confidence
                """,
                canonical_id=canonical_id,
                counterpart_id=counterpart_element_id,
                confidence=confidence,
            )

    @staticmethod
    def _delete_relationship_by_id(tx: Transaction, rel_id: str) -> None:
        """Delete a relationship by its Neo4j element ID."""
        tx.run(
            """
            MATCH ()-[r]->()
            WHERE elementId(r) = $rel_id
            DELETE r
            """,
            rel_id=rel_id,
        )

    @staticmethod
    def _delete_node(tx: Transaction, node_id: str) -> None:
        """
        Delete a node by its Neo4j element ID.
        All relationships must have been removed before calling this.
        """
        tx.run(
            """
            MATCH (n)
            WHERE elementId(n) = $node_id
            DETACH DELETE n
            """,
            node_id=node_id,
        )
