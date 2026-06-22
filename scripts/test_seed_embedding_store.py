"""
Unit tests for scripts/seed_embedding_store.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.seed_embedding_store import BackfillSeeder


@pytest.fixture
def tmp_audit_dir(tmp_path):
    """Create a temporary audit directory with sample data."""
    audit_dir = tmp_path / "data" / "audit"
    audit_dir.mkdir(parents=True)
    return audit_dir


@pytest.fixture
def sample_kept_records():
    """Sample kept.json records."""
    return [
        {
            "title": "Paper A",
            "doi": "10.1000/a",
            "pmid": "111",
            "abstract": "This is a relevant paper about genomics.",
            "decision": "keep",
        },
        {
            "title": "Paper B no abstract",
            "doi": "10.1000/b",
            "pmid": "222",
            "abstract": "",
            "decision": "keep",
        },
    ]


@pytest.fixture
def sample_rejected_records():
    """Sample rejected.json records."""
    return [
        {
            "title": "Paper C",
            "doi": "10.1000/c",
            "pmid": "333",
            "abstract": "This paper is not relevant.",
            "decision": "reject",
        },
    ]


@pytest.fixture
def sample_llm_verified_records():
    """Sample llm_verified.json records."""
    return [
        {
            "title": "Paper D verified keep",
            "doi": "10.1000/d",
            "pmid": "444",
            "abstract": "Verified relevant paper.",
            "keep": True,
            "confidence": 0.95,
        },
        {
            "title": "Paper E verified reject",
            "doi": "10.1000/e",
            "pmid": "555",
            "abstract": "Verified irrelevant paper.",
            "keep": False,
            "confidence": 0.90,
        },
        {
            "title": "Paper F no abstract",
            "doi": "10.1000/f",
            "pmid": "666",
            "abstract": None,
            "keep": True,
            "confidence": 0.80,
        },
    ]


class TestBackfillSeeder:
    """Tests for BackfillSeeder."""

    @patch("scripts.seed_embedding_store.EmbeddingModel")
    @patch("scripts.seed_embedding_store.EmbeddingStore")
    def test_run_processes_all_files(
        self, mock_store_cls, mock_model_cls,
        tmp_audit_dir, sample_kept_records, sample_rejected_records, sample_llm_verified_records
    ):
        """Test that run() reads all audit files and produces correct stats."""
        # Write sample audit files
        (tmp_audit_dir / "kept.json").write_text(json.dumps(sample_kept_records))
        (tmp_audit_dir / "rejected.json").write_text(json.dumps(sample_rejected_records))
        (tmp_audit_dir / "llm_verified.json").write_text(json.dumps(sample_llm_verified_records))

        # Mock model to return a fake embedding
        mock_model = MagicMock()
        mock_model.encode_paper.return_value = np.zeros(384, dtype=np.float32)
        mock_model_cls.return_value = mock_model

        # Mock store - nothing pre-existing
        mock_store = MagicMock()
        mock_store.contains.return_value = False
        mock_store_cls.return_value = mock_store

        with patch("scripts.seed_embedding_store.AUDIT_DIR", tmp_audit_dir):
            seeder = BackfillSeeder()
            stats = seeder.run()

        # Paper A: kept, has abstract → positive
        # Paper B: kept, no abstract → skipped
        # Paper C: rejected, has abstract → negative
        # Paper D: llm keep=True, has abstract → positive
        # Paper E: llm keep=False, has abstract → negative
        # Paper F: llm keep=True, no abstract → skipped
        assert stats["positive_added"] == 2  # Paper A + Paper D
        assert stats["negative_added"] == 2  # Paper C + Paper E
        assert stats["skipped_no_abstract"] == 2  # Paper B + Paper F
        assert stats["skipped_duplicate"] == 0

    @patch("scripts.seed_embedding_store.EmbeddingModel")
    @patch("scripts.seed_embedding_store.EmbeddingStore")
    def test_deduplication_skips_existing(
        self, mock_store_cls, mock_model_cls, tmp_audit_dir
    ):
        """Test that records already in the store are skipped."""
        records = [
            {"title": "Existing Paper", "doi": "10.1000/existing", "pmid": "999", "abstract": "Some text."}
        ]
        (tmp_audit_dir / "kept.json").write_text(json.dumps(records))
        (tmp_audit_dir / "rejected.json").write_text("[]")
        (tmp_audit_dir / "llm_verified.json").write_text("[]")

        mock_model = MagicMock()
        mock_model.encode_paper.return_value = np.zeros(384, dtype=np.float32)
        mock_model_cls.return_value = mock_model

        # Store reports it already contains this paper
        mock_store = MagicMock()
        mock_store.contains.return_value = True
        mock_store_cls.return_value = mock_store

        with patch("scripts.seed_embedding_store.AUDIT_DIR", tmp_audit_dir):
            seeder = BackfillSeeder()
            stats = seeder.run()

        assert stats["skipped_duplicate"] == 1
        assert stats["positive_added"] == 0
        mock_store.append.assert_not_called()

    @patch("scripts.seed_embedding_store.EmbeddingModel")
    @patch("scripts.seed_embedding_store.EmbeddingStore")
    def test_exits_when_no_abstracts(
        self, mock_store_cls, mock_model_cls, tmp_audit_dir
    ):
        """Test sys.exit is called when no records have abstracts."""
        records = [
            {"title": "No Abstract Paper", "doi": "10.1000/x", "pmid": "111", "abstract": ""},
            {"title": "Also No Abstract", "doi": "10.1000/y", "pmid": "222", "abstract": None},
        ]
        (tmp_audit_dir / "kept.json").write_text(json.dumps(records))
        (tmp_audit_dir / "rejected.json").write_text("[]")
        (tmp_audit_dir / "llm_verified.json").write_text("[]")

        mock_model_cls.return_value = MagicMock()
        mock_store_cls.return_value = MagicMock()

        with patch("scripts.seed_embedding_store.AUDIT_DIR", tmp_audit_dir):
            seeder = BackfillSeeder()
            with pytest.raises(SystemExit):
                seeder.run()

    @patch("scripts.seed_embedding_store.EmbeddingModel")
    @patch("scripts.seed_embedding_store.EmbeddingStore")
    def test_handles_missing_files_gracefully(
        self, mock_store_cls, mock_model_cls, tmp_audit_dir
    ):
        """Test that missing audit files don't crash the script."""
        # Only create kept.json with one valid record
        records = [{"title": "Valid", "doi": "10.1000/v", "pmid": "100", "abstract": "Has abstract."}]
        (tmp_audit_dir / "kept.json").write_text(json.dumps(records))
        # rejected.json and llm_verified.json don't exist

        mock_model = MagicMock()
        mock_model.encode_paper.return_value = np.zeros(384, dtype=np.float32)
        mock_model_cls.return_value = mock_model

        mock_store = MagicMock()
        mock_store.contains.return_value = False
        mock_store_cls.return_value = mock_store

        with patch("scripts.seed_embedding_store.AUDIT_DIR", tmp_audit_dir):
            seeder = BackfillSeeder()
            stats = seeder.run()

        assert stats["positive_added"] == 1
        assert stats["negative_added"] == 0

    def test_has_abstract_helper(self):
        """Test the _has_abstract static method."""
        assert BackfillSeeder._has_abstract({"abstract": "Some text"}) is True
        assert BackfillSeeder._has_abstract({"abstract": ""}) is False
        assert BackfillSeeder._has_abstract({"abstract": None}) is False
        assert BackfillSeeder._has_abstract({"abstract": "   "}) is False
        assert BackfillSeeder._has_abstract({}) is False
