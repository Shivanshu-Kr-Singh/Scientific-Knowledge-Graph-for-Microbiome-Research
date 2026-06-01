"""
graph/incremental_processor.py
-------------------------------
Incremental processing system for tracking processed papers and enabling
resumable extraction pipelines.

This module implements:
1. Tracking which papers have already been processed in the database
2. Identifying new papers added since the last extraction run
3. Only extracting relationships from new/unprocessed papers
4. Updating existing reified claims incrementally when new evidence is added
5. Maintaining processing state to enable resumable extraction pipelines

Requirements: 17.4
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Set, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from nlp.enriched_record import EnrichedPaperRecord
from graph.enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphEdge
from graph.reified_claims import ScientificClaim
from graph.relationship_reifier import RelationshipReifier


logger = logging.getLogger(__name__)


@dataclass
class ProcessingState:
    """
    State information for a processing run.
    
    Attributes:
        run_id: Unique identifier for this processing run
        started_at: Timestamp when processing started
        completed_at: Timestamp when processing completed (None if in progress)
        papers_processed: Number of papers processed in this run
        edges_created: Number of edges created in this run
        claims_updated: Number of claims updated in this run
        status: Status of the run (in_progress, completed, failed)
    """
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    papers_processed: int = 0
    edges_created: int = 0
    claims_updated: int = 0
    status: str = "in_progress"  # in_progress | completed | failed


class IncrementalProcessor:
    """
    Tracks processed papers and enables incremental extraction.
    
    This class maintains a SQLite database that tracks:
    - Which papers have been processed
    - When each paper was processed
    - Processing run history
    - Extraction statistics
    
    Requirements: 17.4
    """
    
    def __init__(self, db_path: str = "data/processed_papers.db"):
        """
        Initialize the incremental processor.
        
        Args:
            db_path: Path to SQLite database for tracking processed papers
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized incremental processor with database: {self.db_path}")
    
    def _init_database(self):
        """
        Create database tables if they don't exist.
        
        Tables:
        - processed_papers: Tracks which papers have been processed
        - processing_runs: Tracks processing run history
        - claim_updates: Tracks when claims were updated with new evidence
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Table for tracking processed papers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_papers (
                paper_id TEXT PRIMARY KEY,
                doi TEXT,
                pmid TEXT,
                title TEXT,
                processed_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                edges_extracted INTEGER DEFAULT 0,
                extraction_method TEXT,
                extractor_version TEXT,
                content_hash TEXT,
                FOREIGN KEY (run_id) REFERENCES processing_runs(run_id)
            )
        """)
        
        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_papers_doi 
            ON processed_papers(doi)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_papers_pmid 
            ON processed_papers(pmid)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_papers_run_id 
            ON processed_papers(run_id)
        """)
        
        # Table for tracking processing runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                papers_processed INTEGER DEFAULT 0,
                edges_created INTEGER DEFAULT 0,
                claims_updated INTEGER DEFAULT 0,
                status TEXT DEFAULT 'in_progress',
                extraction_method TEXT,
                extractor_version TEXT,
                notes TEXT
            )
        """)
        
        # Table for tracking claim updates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claim_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                update_type TEXT NOT NULL,
                previous_confidence REAL,
                new_confidence REAL,
                FOREIGN KEY (run_id) REFERENCES processing_runs(run_id)
            )
        """)
        
        # Index for claim updates
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_claim_updates_claim_id 
            ON claim_updates(claim_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_claim_updates_run_id 
            ON claim_updates(run_id)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Initialized incremental processing database")
    
    def is_paper_processed(self, paper: EnrichedPaperRecord) -> bool:
        """
        Check if a paper has already been processed.
        
        A paper is considered processed if it exists in the processed_papers
        table with the same content hash (to detect updates).
        
        Args:
            paper: EnrichedPaperRecord to check
        
        Returns:
            True if paper has been processed, False otherwise
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get paper identifier (prefer DOI, fallback to PMID or title)
        paper_id = self._get_paper_id(paper)
        
        # Check if paper exists with same content hash
        cursor.execute("""
            SELECT content_hash 
            FROM processed_papers 
            WHERE paper_id = ?
        """, (paper_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return False
        
        # Check if content has changed
        stored_hash = result[0]
        current_hash = self._compute_content_hash(paper)
        
        if stored_hash != current_hash:
            logger.info(
                f"Paper {paper_id} has been updated (hash changed), "
                f"will re-process"
            )
            return False
        
        return True
    
    def get_unprocessed_papers(
        self,
        papers: List[EnrichedPaperRecord]
    ) -> List[EnrichedPaperRecord]:
        """
        Filter papers to only include those that haven't been processed.
        
        This is the main method for incremental processing - it identifies
        new papers that need extraction.
        
        Args:
            papers: List of all papers
        
        Returns:
            List of papers that haven't been processed yet
        """
        unprocessed = []
        
        for paper in papers:
            if not self.is_paper_processed(paper):
                unprocessed.append(paper)
        
        logger.info(
            f"Found {len(unprocessed)} unprocessed papers out of {len(papers)} total"
        )
        
        return unprocessed
    
    def mark_paper_processed(
        self,
        paper: EnrichedPaperRecord,
        run_id: str,
        edges_extracted: int,
        extraction_method: str,
        extractor_version: str
    ):
        """
        Mark a paper as processed in the database.
        
        Args:
            paper: Paper that was processed
            run_id: ID of the processing run
            edges_extracted: Number of edges extracted from this paper
            extraction_method: Extraction method used
            extractor_version: Version of the extractor
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        paper_id = self._get_paper_id(paper)
        content_hash = self._compute_content_hash(paper)
        
        # Insert or replace (in case of re-processing)
        cursor.execute("""
            INSERT OR REPLACE INTO processed_papers (
                paper_id, doi, pmid, title, processed_at, run_id,
                edges_extracted, extraction_method, extractor_version,
                content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_id,
            paper.doi,
            paper.pmid,
            paper.title,
            datetime.now().isoformat(),
            run_id,
            edges_extracted,
            extraction_method,
            extractor_version,
            content_hash
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Marked paper {paper_id} as processed")
    
    def start_processing_run(
        self,
        extraction_method: str,
        extractor_version: str,
        notes: Optional[str] = None
    ) -> ProcessingState:
        """
        Start a new processing run and return its state.
        
        Args:
            extraction_method: Extraction method being used
            extractor_version: Version of the extractor
            notes: Optional notes about this run
        
        Returns:
            ProcessingState for this run
        """
        import time
        # Use microseconds to avoid duplicate run_ids
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000000) % 1000000}"
        started_at = datetime.now()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO processing_runs (
                run_id, started_at, status, extraction_method,
                extractor_version, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            started_at.isoformat(),
            "in_progress",
            extraction_method,
            extractor_version,
            notes
        ))
        
        conn.commit()
        conn.close()
        
        state = ProcessingState(
            run_id=run_id,
            started_at=started_at,
            status="in_progress"
        )
        
        logger.info(f"Started processing run: {run_id}")
        
        return state
    
    def complete_processing_run(
        self,
        state: ProcessingState,
        status: str = "completed"
    ):
        """
        Mark a processing run as completed.
        
        Args:
            state: ProcessingState to complete
            status: Final status (completed or failed)
        """
        state.completed_at = datetime.now()
        state.status = status
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE processing_runs
            SET completed_at = ?,
                papers_processed = ?,
                edges_created = ?,
                claims_updated = ?,
                status = ?
            WHERE run_id = ?
        """, (
            state.completed_at.isoformat(),
            state.papers_processed,
            state.edges_created,
            state.claims_updated,
            state.status,
            state.run_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(
            f"Completed processing run {state.run_id}: "
            f"{state.papers_processed} papers, "
            f"{state.edges_created} edges, "
            f"{state.claims_updated} claims updated"
        )
    
    def record_claim_update(
        self,
        claim_id: str,
        paper_id: str,
        run_id: str,
        update_type: str,
        previous_confidence: float,
        new_confidence: float
    ):
        """
        Record that a claim was updated with new evidence.
        
        Args:
            claim_id: ID of the claim that was updated
            paper_id: ID of the paper that provided new evidence
            run_id: ID of the processing run
            update_type: Type of update (supporting | contradicting)
            previous_confidence: Confidence before update
            new_confidence: Confidence after update
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO claim_updates (
                claim_id, paper_id, updated_at, run_id, update_type,
                previous_confidence, new_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            claim_id,
            paper_id,
            datetime.now().isoformat(),
            run_id,
            update_type,
            previous_confidence,
            new_confidence
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(
            f"Recorded claim update: {claim_id} from paper {paper_id}, "
            f"confidence {previous_confidence:.3f} -> {new_confidence:.3f}"
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed papers and runs.
        
        Returns:
            Dictionary with processing statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total processed papers
        cursor.execute("SELECT COUNT(*) FROM processed_papers")
        total_papers = cursor.fetchone()[0]
        
        # Total processing runs
        cursor.execute("SELECT COUNT(*) FROM processing_runs")
        total_runs = cursor.fetchone()[0]
        
        # Completed runs
        cursor.execute(
            "SELECT COUNT(*) FROM processing_runs WHERE status = 'completed'"
        )
        completed_runs = cursor.fetchone()[0]
        
        # Total edges extracted
        cursor.execute("SELECT SUM(edges_extracted) FROM processed_papers")
        total_edges = cursor.fetchone()[0] or 0
        
        # Total claim updates
        cursor.execute("SELECT COUNT(*) FROM claim_updates")
        total_claim_updates = cursor.fetchone()[0]
        
        # Last processing run
        cursor.execute("""
            SELECT run_id, started_at, completed_at, papers_processed, status
            FROM processing_runs
            ORDER BY started_at DESC
            LIMIT 1
        """)
        last_run = cursor.fetchone()
        
        conn.close()
        
        stats = {
            "total_papers_processed": total_papers,
            "total_processing_runs": total_runs,
            "completed_runs": completed_runs,
            "total_edges_extracted": total_edges,
            "total_claim_updates": total_claim_updates,
            "last_run": None
        }
        
        if last_run:
            stats["last_run"] = {
                "run_id": last_run[0],
                "started_at": last_run[1],
                "completed_at": last_run[2],
                "papers_processed": last_run[3],
                "status": last_run[4]
            }
        
        return stats
    
    def _get_paper_id(self, paper: EnrichedPaperRecord) -> str:
        """
        Get a unique identifier for a paper.
        
        Priority: DOI > PMID > title hash
        
        Args:
            paper: Paper to get ID for
        
        Returns:
            Unique paper identifier
        """
        if paper.doi:
            return f"doi:{paper.doi.lower().strip()}"
        if paper.pmid:
            return f"pmid:{paper.pmid}"
        # Fallback to title hash
        import hashlib
        title_hash = hashlib.md5(paper.title.lower().encode()).hexdigest()[:16]
        return f"title:{title_hash}"
    
    def _compute_content_hash(self, paper: EnrichedPaperRecord) -> str:
        """
        Compute a hash of the paper's content to detect updates.
        
        Uses title + abstract to detect if paper content has changed.
        
        Args:
            paper: Paper to hash
        
        Returns:
            MD5 hash of paper content
        """
        import hashlib
        
        content = f"{paper.title}|{paper.abstract or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_last_processing_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the last completed processing run.
        
        Returns:
            Datetime of last processing run, or None if no runs exist
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT completed_at
            FROM processing_runs
            WHERE status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        
        return None
    
    def get_papers_since(
        self,
        since: datetime,
        papers: List[EnrichedPaperRecord]
    ) -> List[EnrichedPaperRecord]:
        """
        Get papers that were added after a specific timestamp.
        
        This is useful for incremental processing based on paper ingestion time.
        
        Args:
            since: Timestamp to filter from
            papers: List of all papers
        
        Returns:
            Papers added after the timestamp (inclusive)
        """
        new_papers = []
        
        for paper in papers:
            # Check if paper has a fetched_at timestamp
            if paper.fetched_at:
                try:
                    fetched_time = datetime.fromisoformat(paper.fetched_at)
                    if fetched_time >= since:
                        new_papers.append(paper)
                except (ValueError, TypeError):
                    # If timestamp parsing fails, include the paper
                    new_papers.append(paper)
            else:
                # If no timestamp, include the paper
                new_papers.append(paper)
        
        logger.info(
            f"Found {len(new_papers)} papers added since {since.isoformat()}"
        )
        
        return new_papers


class IncrementalExtractionPipeline:
    """
    Pipeline that performs incremental extraction on new papers only.
    
    This pipeline:
    1. Identifies unprocessed papers
    2. Extracts relationships from new papers only
    3. Updates existing reified claims with new evidence
    4. Tracks processing state for resumability
    
    Requirements: 17.4
    """
    
    def __init__(
        self,
        processor: IncrementalProcessor,
        extraction_method: str = "regex_ner",
        extractor_version: str = "1.0"
    ):
        """
        Initialize the incremental extraction pipeline.
        
        Args:
            processor: IncrementalProcessor for tracking state
            extraction_method: Extraction method to use
            extractor_version: Version of the extractor
        """
        self.processor = processor
        self.extraction_method = extraction_method
        self.extractor_version = extractor_version
        
        logger.info("Initialized incremental extraction pipeline")
    
    def run(
        self,
        all_papers: List[EnrichedPaperRecord],
        existing_claims: Optional[List[ScientificClaim]] = None
    ) -> Dict[str, Any]:
        """
        Run incremental extraction on new papers only.
        
        This method:
        1. Filters to unprocessed papers
        2. Extracts relationships from new papers
        3. Updates existing claims with new evidence
        4. Tracks processing state
        
        Args:
            all_papers: List of all papers (processed and unprocessed)
            existing_claims: Optional list of existing claims to update
        
        Returns:
            Dictionary with extraction results and statistics
        """
        # Start processing run
        state = self.processor.start_processing_run(
            extraction_method=self.extraction_method,
            extractor_version=self.extractor_version,
            notes="Incremental extraction run"
        )
        
        try:
            # Filter to unprocessed papers
            unprocessed_papers = self.processor.get_unprocessed_papers(all_papers)
            
            if not unprocessed_papers:
                logger.info("No new papers to process")
                self.processor.complete_processing_run(state, status="completed")
                return {
                    "status": "no_new_papers",
                    "papers_processed": 0,
                    "edges_created": 0,
                    "claims_updated": 0
                }
            
            logger.info(f"Processing {len(unprocessed_papers)} new papers")
            
            # Extract relationships from new papers
            builder = EnhancedGraphBuilder(
                extraction_method=self.extraction_method,
                extractor_version=self.extractor_version
            )
            
            new_edges = builder.process_papers(unprocessed_papers)
            
            # Mark papers as processed
            for paper in unprocessed_papers:
                paper_id = self.processor._get_paper_id(paper)
                # Count edges for this paper
                paper_edges = [
                    e for e in new_edges
                    if e.source == paper_id or paper_id in e.source
                ]
                
                self.processor.mark_paper_processed(
                    paper=paper,
                    run_id=state.run_id,
                    edges_extracted=len(paper_edges),
                    extraction_method=self.extraction_method,
                    extractor_version=self.extractor_version
                )
            
            # Update existing claims with new evidence
            claims_updated = 0
            if existing_claims:
                claims_updated = self._update_existing_claims(
                    existing_claims=existing_claims,
                    new_edges=new_edges,
                    run_id=state.run_id
                )
            
            # Update state
            state.papers_processed = len(unprocessed_papers)
            state.edges_created = len(new_edges)
            state.claims_updated = claims_updated
            
            # Complete processing run
            self.processor.complete_processing_run(state, status="completed")
            
            logger.info(
                f"Incremental extraction completed: "
                f"{state.papers_processed} papers, "
                f"{state.edges_created} edges, "
                f"{state.claims_updated} claims updated"
            )
            
            return {
                "status": "success",
                "run_id": state.run_id,
                "papers_processed": state.papers_processed,
                "edges_created": state.edges_created,
                "claims_updated": state.claims_updated,
                "new_edges": new_edges
            }
            
        except Exception as e:
            logger.error(f"Error in incremental extraction: {e}", exc_info=True)
            self.processor.complete_processing_run(state, status="failed")
            raise
    
    def _update_existing_claims(
        self,
        existing_claims: List[ScientificClaim],
        new_edges: List[EnhancedGraphEdge],
        run_id: str
    ) -> int:
        """
        Update existing reified claims with new evidence from edges.
        
        Args:
            existing_claims: List of existing claims
            new_edges: New edges to incorporate
            run_id: Processing run ID
        
        Returns:
            Number of claims updated
        """
        reifier = RelationshipReifier()
        claims_updated = 0
        
        # Build index of existing claims by (subject, predicate, object)
        claim_index = {}
        for claim in existing_claims:
            key = (claim.subject_entity, claim.predicate, claim.object_entity)
            claim_index[key] = claim
        
        # Process new edges and update matching claims
        for edge in new_edges:
            # Create a key from the edge
            key = (edge.source, edge.relation, edge.target)
            
            if key in claim_index:
                claim = claim_index[key]
                previous_confidence = claim.consensus_confidence
                
                # Update claim with new evidence
                # Determine if this edge supports or contradicts
                supports = True  # Default to supporting
                
                # Check if direction contradicts existing claim
                if hasattr(edge, 'direction') and edge.direction:
                    # Simple heuristic: if most existing papers have opposite direction
                    if claim.effect_direction_consistency > 0.5:
                        # Majority direction is established
                        # Check if new edge contradicts
                        # This is simplified - in practice would need more logic
                        pass
                
                updated_claim = reifier.update_claim_with_new_evidence(
                    claim=claim,
                    new_evidence=edge.provenance,
                    supports=supports
                )
                
                # Record the update
                self.processor.record_claim_update(
                    claim_id=claim.claim_id,
                    paper_id=edge.source,
                    run_id=run_id,
                    update_type="supporting" if supports else "contradicting",
                    previous_confidence=previous_confidence,
                    new_confidence=updated_claim.consensus_confidence
                )
                
                claims_updated += 1
        
        logger.info(f"Updated {claims_updated} existing claims with new evidence")
        
        return claims_updated
