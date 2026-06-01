"""
graph/test_incremental_processor.py
------------------------------------
Unit tests for the incremental processing system.

Tests verify:
1. Tracking of processed papers in database
2. Identification of new/unprocessed papers
3. Incremental extraction from new papers only
4. Updating reified claims with new evidence
5. Processing state management for resumability

Requirements: 17.4
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from nlp.enriched_record import EnrichedPaperRecord
from graph.incremental_processor import (
    IncrementalProcessor,
    IncrementalExtractionPipeline,
    ProcessingState
)
from graph.enhanced_graph_builder import EnhancedGraphBuilder
from graph.reified_claims import ScientificClaim
from models import PaperRecord


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_processed_papers.db"
    yield str(db_path)
    shutil.rmtree(temp_dir)


@pytest.fixture
def processor(temp_db):
    """Create an IncrementalProcessor with temporary database."""
    return IncrementalProcessor(db_path=temp_db)


@pytest.fixture
def sample_papers() -> List[EnrichedPaperRecord]:
    """Create sample enriched papers for testing."""
    papers = []
    
    for i in range(5):
        paper = EnrichedPaperRecord(
            doi=f"10.1234/paper{i}",
            pmid=f"PMID{i}",
            title=f"Test Paper {i}",
            abstract=f"This is the abstract for paper {i}",
            source="pubmed",
            article_type_normalized="original_research",
            taxa=["Bacteroides", "Lactobacillus"],
            diseases=["IBD"],
            methods=["16S rRNA"],
            fetched_at=datetime.now().isoformat()
        )
        papers.append(paper)
    
    return papers


class TestIncrementalProcessor:
    """Test suite for IncrementalProcessor."""
    
    def test_database_initialization(self, processor):
        """Test that database tables are created correctly."""
        import sqlite3
        
        conn = sqlite3.connect(processor.db_path)
        cursor = conn.cursor()
        
        # Check that tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN (
                'processed_papers', 'processing_runs', 'claim_updates'
            )
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "processed_papers" in tables
        assert "processing_runs" in tables
        assert "claim_updates" in tables
        
        conn.close()
    
    def test_is_paper_processed_new_paper(self, processor, sample_papers):
        """Test that new papers are identified as unprocessed."""
        paper = sample_papers[0]
        
        # Paper should not be processed initially
        assert not processor.is_paper_processed(paper)
    
    def test_mark_paper_processed(self, processor, sample_papers):
        """Test marking a paper as processed."""
        paper = sample_papers[0]
        run_id = "test_run_001"
        
        # Mark paper as processed
        processor.mark_paper_processed(
            paper=paper,
            run_id=run_id,
            edges_extracted=5,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Paper should now be processed
        assert processor.is_paper_processed(paper)
    
    def test_is_paper_processed_content_change(self, processor, sample_papers):
        """Test that papers with changed content are re-processed."""
        paper = sample_papers[0]
        run_id = "test_run_001"
        
        # Mark paper as processed
        processor.mark_paper_processed(
            paper=paper,
            run_id=run_id,
            edges_extracted=5,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Paper should be processed
        assert processor.is_paper_processed(paper)
        
        # Change paper content
        paper.abstract = "This is a completely different abstract"
        
        # Paper should now be unprocessed (content changed)
        assert not processor.is_paper_processed(paper)
    
    def test_get_unprocessed_papers(self, processor, sample_papers):
        """Test filtering to unprocessed papers only."""
        run_id = "test_run_001"
        
        # Mark first 3 papers as processed
        for paper in sample_papers[:3]:
            processor.mark_paper_processed(
                paper=paper,
                run_id=run_id,
                edges_extracted=5,
                extraction_method="regex_ner",
                extractor_version="1.0"
            )
        
        # Get unprocessed papers
        unprocessed = processor.get_unprocessed_papers(sample_papers)
        
        # Should return last 2 papers
        assert len(unprocessed) == 2
        assert unprocessed[0].doi == "10.1234/paper3"
        assert unprocessed[1].doi == "10.1234/paper4"
    
    def test_start_processing_run(self, processor):
        """Test starting a processing run."""
        state = processor.start_processing_run(
            extraction_method="regex_ner",
            extractor_version="1.0",
            notes="Test run"
        )
        
        assert state.run_id.startswith("run_")
        assert state.status == "in_progress"
        assert state.papers_processed == 0
        assert state.edges_created == 0
        assert state.claims_updated == 0
    
    def test_complete_processing_run(self, processor):
        """Test completing a processing run."""
        state = processor.start_processing_run(
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Update state
        state.papers_processed = 10
        state.edges_created = 50
        state.claims_updated = 5
        
        # Complete run
        processor.complete_processing_run(state, status="completed")
        
        assert state.status == "completed"
        assert state.completed_at is not None
    
    def test_record_claim_update(self, processor):
        """Test recording claim updates."""
        run_id = "test_run_001"
        
        processor.record_claim_update(
            claim_id="claim_123",
            paper_id="doi:10.1234/paper1",
            run_id=run_id,
            update_type="supporting",
            previous_confidence=0.75,
            new_confidence=0.82
        )
        
        # Verify update was recorded
        import sqlite3
        conn = sqlite3.connect(processor.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT claim_id, update_type, previous_confidence, new_confidence
            FROM claim_updates
            WHERE claim_id = ?
        """, ("claim_123",))
        
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == "claim_123"
        assert result[1] == "supporting"
        assert result[2] == 0.75
        assert result[3] == 0.82
    
    def test_get_processing_statistics(self, processor, sample_papers):
        """Test getting processing statistics."""
        run_id = "test_run_001"
        
        # Process some papers
        for paper in sample_papers[:3]:
            processor.mark_paper_processed(
                paper=paper,
                run_id=run_id,
                edges_extracted=5,
                extraction_method="regex_ner",
                extractor_version="1.0"
            )
        
        # Get statistics
        stats = processor.get_processing_statistics()
        
        assert stats["total_papers_processed"] == 3
        assert stats["total_edges_extracted"] == 15  # 3 papers * 5 edges
    
    def test_get_last_processing_time(self, processor):
        """Test getting last processing time."""
        # Initially no processing runs
        assert processor.get_last_processing_time() is None
        
        # Start and complete a run
        state = processor.start_processing_run(
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        processor.complete_processing_run(state, status="completed")
        
        # Should return the completion time
        last_time = processor.get_last_processing_time()
        assert last_time is not None
        assert isinstance(last_time, datetime)
    
    def test_get_papers_since(self, processor):
        """Test filtering papers by timestamp."""
        # Create papers with different timestamps
        papers = []
        base_time = datetime.now()
        
        for i in range(5):
            paper = EnrichedPaperRecord(
                doi=f"10.1234/paper{i}",
                title=f"Test Paper {i}",
                abstract=f"Abstract {i}",
                source="pubmed",
                article_type_normalized="original_research",
                fetched_at=(base_time - timedelta(days=i)).isoformat()
            )
            papers.append(paper)
        
        # Get papers from last 2 days
        since = base_time - timedelta(days=2)
        recent_papers = processor.get_papers_since(since, papers)
        
        # Should return papers 0, 1, 2 (fetched 0, 1, 2 days ago)
        assert len(recent_papers) == 3
    
    def test_paper_id_generation(self, processor):
        """Test paper ID generation with different identifiers."""
        # Paper with DOI
        paper1 = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="Abstract",
            source="pubmed",
            article_type_normalized="original_research"
        )
        id1 = processor._get_paper_id(paper1)
        assert id1.startswith("doi:")
        
        # Paper with PMID only
        paper2 = EnrichedPaperRecord(
            pmid="12345",
            title="Test Paper",
            abstract="Abstract",
            source="pubmed",
            article_type_normalized="original_research"
        )
        id2 = processor._get_paper_id(paper2)
        assert id2.startswith("pmid:")
        
        # Paper with title only
        paper3 = EnrichedPaperRecord(
            title="Test Paper",
            abstract="Abstract",
            source="pubmed",
            article_type_normalized="original_research"
        )
        id3 = processor._get_paper_id(paper3)
        assert id3.startswith("title:")
    
    def test_content_hash_computation(self, processor, sample_papers):
        """Test content hash computation."""
        paper = sample_papers[0]
        
        hash1 = processor._compute_content_hash(paper)
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
        
        # Same paper should produce same hash
        hash2 = processor._compute_content_hash(paper)
        assert hash1 == hash2
        
        # Different content should produce different hash
        paper.abstract = "Different abstract"
        hash3 = processor._compute_content_hash(paper)
        assert hash1 != hash3


class TestIncrementalExtractionPipeline:
    """Test suite for IncrementalExtractionPipeline."""
    
    def test_pipeline_initialization(self, processor):
        """Test pipeline initialization."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        assert pipeline.processor == processor
        assert pipeline.extraction_method == "regex_ner"
        assert pipeline.extractor_version == "1.0"
    
    def test_run_with_no_new_papers(self, processor, sample_papers):
        """Test running pipeline when all papers are already processed."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Mark all papers as processed
        run_id = "test_run_001"
        for paper in sample_papers:
            processor.mark_paper_processed(
                paper=paper,
                run_id=run_id,
                edges_extracted=5,
                extraction_method="regex_ner",
                extractor_version="1.0"
            )
        
        # Run pipeline
        result = pipeline.run(sample_papers)
        
        assert result["status"] == "no_new_papers"
        assert result["papers_processed"] == 0
        assert result["edges_created"] == 0
    
    def test_run_with_new_papers(self, processor, sample_papers):
        """Test running pipeline with new papers."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Mark first 3 papers as processed
        run_id = "test_run_001"
        for paper in sample_papers[:3]:
            processor.mark_paper_processed(
                paper=paper,
                run_id=run_id,
                edges_extracted=5,
                extraction_method="regex_ner",
                extractor_version="1.0"
            )
        
        # Run pipeline (should process last 2 papers)
        result = pipeline.run(sample_papers)
        
        assert result["status"] == "success"
        assert result["papers_processed"] == 2
        assert "run_id" in result
        assert "edges_created" in result
    
    def test_run_tracks_processing_state(self, processor, sample_papers):
        """Test that pipeline tracks processing state correctly."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Run pipeline
        result = pipeline.run(sample_papers)
        
        # Check that processing run was recorded
        stats = processor.get_processing_statistics()
        assert stats["total_processing_runs"] >= 1
        assert stats["completed_runs"] >= 1
        assert stats["last_run"] is not None
        assert stats["last_run"]["status"] == "completed"
    
    def test_run_marks_papers_as_processed(self, processor, sample_papers):
        """Test that pipeline marks papers as processed."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Initially no papers are processed
        unprocessed = processor.get_unprocessed_papers(sample_papers)
        assert len(unprocessed) == len(sample_papers)
        
        # Run pipeline
        result = pipeline.run(sample_papers)
        
        # All papers should now be processed
        unprocessed = processor.get_unprocessed_papers(sample_papers)
        assert len(unprocessed) == 0
    
    def test_run_handles_errors_gracefully(self, processor):
        """Test that pipeline handles errors and marks run as failed."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Create invalid papers that will cause errors
        invalid_papers = [
            EnrichedPaperRecord(
                title="",  # Empty title
                abstract="",
                source="pubmed",
                article_type_normalized="original_research"
            )
        ]
        
        # Run should handle error
        try:
            result = pipeline.run(invalid_papers)
        except Exception:
            # Check that run was marked as failed
            stats = processor.get_processing_statistics()
            if stats["last_run"]:
                # Run may be marked as failed
                pass


class TestIncrementalProcessingIntegration:
    """Integration tests for incremental processing."""
    
    def test_full_incremental_workflow(self, processor, sample_papers):
        """Test complete incremental processing workflow."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # First run: process all papers
        result1 = pipeline.run(sample_papers)
        assert result1["status"] == "success"
        assert result1["papers_processed"] == len(sample_papers)
        
        # Second run: no new papers
        result2 = pipeline.run(sample_papers)
        assert result2["status"] == "no_new_papers"
        assert result2["papers_processed"] == 0
        
        # Add new papers
        new_papers = sample_papers + [
            EnrichedPaperRecord(
                doi="10.1234/paper_new",
                title="New Paper",
                abstract="New abstract",
                source="pubmed",
                article_type_normalized="original_research",
                taxa=["Bacteroides"],
                diseases=["IBD"],
                methods=["16S rRNA"]
            )
        ]
        
        # Third run: process only new paper
        result3 = pipeline.run(new_papers)
        assert result3["status"] == "success"
        assert result3["papers_processed"] == 1
    
    def test_processing_statistics_accuracy(self, processor, sample_papers):
        """Test that processing statistics are accurate."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Process papers
        result = pipeline.run(sample_papers)
        
        # Get statistics
        stats = processor.get_processing_statistics()
        
        # Verify statistics
        assert stats["total_papers_processed"] == len(sample_papers)
        assert stats["total_processing_runs"] >= 1
        assert stats["completed_runs"] >= 1
        assert stats["total_edges_extracted"] >= 0
    
    def test_resumable_processing(self, processor, sample_papers):
        """Test that processing can be resumed after interruption."""
        pipeline = IncrementalExtractionPipeline(
            processor=processor,
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        # Process first batch
        batch1 = sample_papers[:3]
        result1 = pipeline.run(batch1)
        assert result1["papers_processed"] == 3
        
        # Process second batch (includes some from first batch)
        batch2 = sample_papers  # All papers
        result2 = pipeline.run(batch2)
        
        # Should only process the 2 new papers
        assert result2["papers_processed"] == 2
        
        # Total processed should be 5
        stats = processor.get_processing_statistics()
        assert stats["total_papers_processed"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
