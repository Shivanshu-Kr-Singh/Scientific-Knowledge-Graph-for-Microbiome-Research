#!/usr/bin/env python3
"""
scripts/test_backup_rollback.py
--------------------------------
Unit tests for Neo4j backup and rollback mechanisms.

Requirements: 16.4
"""

import os
import sys
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backup_neo4j import Neo4jBackupManager
from scripts.rollback_neo4j import Neo4jRollbackManager


class TestNeo4jBackupManager:
    """Test suite for Neo4jBackupManager."""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create a temporary backup directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        driver.session.return_value.__exit__.return_value = None
        return driver
    
    @pytest.fixture
    def backup_manager(self, temp_backup_dir, mock_driver):
        """Create a backup manager with mocked driver."""
        with patch('scripts.backup_neo4j.GraphDatabase.driver', return_value=mock_driver):
            manager = Neo4jBackupManager(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                backup_dir=temp_backup_dir
            )
            yield manager
            manager.close()
    
    def test_initialization(self, temp_backup_dir):
        """Test backup manager initialization."""
        with patch('scripts.backup_neo4j.GraphDatabase.driver'):
            manager = Neo4jBackupManager(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                backup_dir=temp_backup_dir
            )
            
            assert manager.neo4j_uri == "bolt://localhost:7687"
            assert manager.neo4j_user == "neo4j"
            assert manager.backup_dir == Path(temp_backup_dir)
            assert manager.backup_dir.exists()
            
            manager.close()
    
    def test_get_database_stats(self, backup_manager, mock_driver):
        """Test gathering database statistics."""
        # Mock session
        session = mock_driver.session.return_value.__enter__.return_value
        
        # Mock labels query
        labels_result = MagicMock()
        labels_result.__iter__.return_value = [
            {"label": "Paper"},
            {"label": "Taxon"},
        ]
        session.run.return_value = labels_result
        
        # Mock count queries
        def run_side_effect(query):
            if "MATCH (n:Paper)" in query:
                result = MagicMock()
                result.single.return_value = {"count": 100}
                return result
            elif "MATCH (n:Taxon)" in query:
                result = MagicMock()
                result.single.return_value = {"count": 50}
                return result
            elif "db.labels()" in query:
                return labels_result
            elif "db.relationshipTypes()" in query:
                rel_result = MagicMock()
                rel_result.__iter__.return_value = [{"relationshipType": "HAS_TAXON"}]
                return rel_result
            elif "MATCH ()-[r:HAS_TAXON]->()" in query:
                result = MagicMock()
                result.single.return_value = {"count": 200}
                return result
            return MagicMock()
        
        session.run.side_effect = run_side_effect
        
        # Get stats
        stats = backup_manager.get_database_stats()
        
        # Verify stats
        assert stats["nodes"]["Paper"] == 100
        assert stats["nodes"]["Taxon"] == 50
        assert stats["total_nodes"] == 150
        assert stats["relationships"]["HAS_TAXON"] == 200
        assert stats["total_relationships"] == 200
    
    def test_format_value(self, backup_manager):
        """Test Cypher value formatting."""
        # Test different value types
        assert backup_manager._format_value(None) == "null"
        assert backup_manager._format_value(True) == "true"
        assert backup_manager._format_value(False) == "false"
        assert backup_manager._format_value(42) == "42"
        assert backup_manager._format_value(3.14) == "3.14"
        assert backup_manager._format_value("hello") == '"hello"'
        assert backup_manager._format_value("quote\"test") == '"quote\\"test"'
        assert backup_manager._format_value([1, 2, 3]) == "[1, 2, 3]"
        assert backup_manager._format_value({"key": "value"}) == '{key: "value"}'
    
    def test_export_to_cypher(self, backup_manager, mock_driver, temp_backup_dir):
        """Test exporting database to Cypher statements."""
        # Mock session
        session = mock_driver.session.return_value.__enter__.return_value
        
        # Mock labels query
        labels_result = MagicMock()
        labels_result.__iter__.return_value = [{"label": "Paper"}]
        
        # Mock nodes query
        node_result = MagicMock()
        node_mock = MagicMock()
        node_mock.__iter__.return_value = [("id", "paper1"), ("title", "Test Paper")]
        node_result.__iter__.return_value = [{"n": node_mock}]
        
        # Mock relationship types query
        rel_types_result = MagicMock()
        rel_types_result.__iter__.return_value = [{"relationshipType": "HAS_TAXON"}]
        
        # Mock relationships query
        rel_result = MagicMock()
        rel_result.__iter__.return_value = [
            {
                "source_id": "paper1",
                "target_id": "taxon1",
                "props": {"confidence": 0.85},
                "source_label": "Paper",
                "target_label": "Taxon",
            }
        ]
        
        def run_side_effect(query):
            if "db.labels()" in query:
                return labels_result
            elif "MATCH (n:Paper)" in query:
                return node_result
            elif "db.relationshipTypes()" in query:
                return rel_types_result
            elif "MATCH (source)-[r:HAS_TAXON]->(target)" in query:
                return rel_result
            return MagicMock()
        
        session.run.side_effect = run_side_effect
        
        # Export to Cypher
        cypher_file = backup_manager.export_to_cypher("test_backup")
        
        # Verify file was created
        assert cypher_file.exists()
        assert cypher_file.name == "test_backup_export.cypher"
        
        # Verify file content
        content = cypher_file.read_text()
        assert "Neo4j Database Backup" in content
        assert "CREATE (:Paper" in content
        assert "HAS_TAXON" in content
    
    def test_create_backup(self, backup_manager, mock_driver, temp_backup_dir):
        """Test creating a complete backup."""
        # Mock session
        session = mock_driver.session.return_value.__enter__.return_value
        
        # Mock database stats
        labels_result = MagicMock()
        labels_result.__iter__.return_value = [{"label": "Paper"}]
        
        count_result = MagicMock()
        count_result.single.return_value = {"count": 10}
        
        rel_types_result = MagicMock()
        rel_types_result.__iter__.return_value = [{"relationshipType": "HAS_TAXON"}]
        
        node_result = MagicMock()
        node_result.__iter__.return_value = []
        
        rel_result = MagicMock()
        rel_result.__iter__.return_value = []
        
        def run_side_effect(query):
            if "db.labels()" in query:
                return labels_result
            elif "count(n)" in query:
                return count_result
            elif "db.relationshipTypes()" in query:
                return rel_types_result
            elif "count(r)" in query:
                return count_result
            elif "MATCH (n:" in query:
                return node_result
            elif "MATCH (source)-[r:" in query:
                return rel_result
            return MagicMock()
        
        session.run.side_effect = run_side_effect
        
        # Create backup
        metadata = backup_manager.create_backup(backup_name="test_backup")
        
        # Verify metadata
        assert metadata["backup_name"] == "test_backup"
        assert "timestamp" in metadata
        assert metadata["neo4j_uri"] == "bolt://localhost:7687"
        assert "statistics" in metadata
        assert metadata["backup_method"] == "cypher_export"
        
        # Verify files were created
        backup_dir = Path(temp_backup_dir)
        assert (backup_dir / "test_backup_export.cypher").exists()
        assert (backup_dir / "test_backup_metadata.json").exists()
        assert (backup_dir / "latest_backup.json").exists()
    
    def test_list_backups(self, backup_manager, temp_backup_dir):
        """Test listing available backups."""
        # Create mock backup metadata files
        backup_dir = Path(temp_backup_dir)
        
        metadata1 = {
            "backup_name": "backup1",
            "timestamp": "2026-05-27T10:00:00+00:00",
        }
        with open(backup_dir / "backup1_metadata.json", 'w') as f:
            json.dump(metadata1, f)
        
        metadata2 = {
            "backup_name": "backup2",
            "timestamp": "2026-05-27T11:00:00+00:00",
        }
        with open(backup_dir / "backup2_metadata.json", 'w') as f:
            json.dump(metadata2, f)
        
        # List backups
        backups = backup_manager.list_backups()
        
        # Verify backups are listed (newest first)
        assert len(backups) == 2
        assert backups[0]["backup_name"] == "backup2"
        assert backups[1]["backup_name"] == "backup1"
    
    def test_verify_backup(self, backup_manager, temp_backup_dir):
        """Test backup verification."""
        backup_dir = Path(temp_backup_dir)
        
        # Create valid backup
        metadata = {
            "backup_name": "valid_backup",
            "cypher_file": "valid_backup_export.cypher",
        }
        with open(backup_dir / "valid_backup_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        with open(backup_dir / "valid_backup_export.cypher", 'w') as f:
            f.write("CREATE (:Test);")
        
        # Verify valid backup
        assert backup_manager.verify_backup("valid_backup") is True
        
        # Test missing metadata
        assert backup_manager.verify_backup("nonexistent") is False
        
        # Test missing Cypher file
        metadata_invalid = {
            "backup_name": "invalid_backup",
            "cypher_file": "missing.cypher",
        }
        with open(backup_dir / "invalid_backup_metadata.json", 'w') as f:
            json.dump(metadata_invalid, f)
        
        assert backup_manager.verify_backup("invalid_backup") is False


class TestNeo4jRollbackManager:
    """Test suite for Neo4jRollbackManager."""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create a temporary backup directory with test backups."""
        temp_dir = tempfile.mkdtemp()
        
        # Create test backup
        backup_dir = Path(temp_dir)
        
        metadata = {
            "backup_name": "test_backup",
            "timestamp": "2026-05-27T10:00:00+00:00",
            "cypher_file": "test_backup_export.cypher",
            "statistics": {
                "nodes": {"Paper": 10},
                "relationships": {"HAS_TAXON": 20},
                "total_nodes": 10,
                "total_relationships": 20,
            }
        }
        
        with open(backup_dir / "test_backup_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        with open(backup_dir / "test_backup_export.cypher", 'w') as f:
            f.write("CREATE (:Paper {id: 'test'});")
        
        with open(backup_dir / "latest_backup.json", 'w') as f:
            json.dump(metadata, f)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        driver.session.return_value.__exit__.return_value = None
        return driver
    
    @pytest.fixture
    def rollback_manager(self, temp_backup_dir, mock_driver):
        """Create a rollback manager with mocked driver."""
        with patch('scripts.rollback_neo4j.GraphDatabase.driver', return_value=mock_driver):
            manager = Neo4jRollbackManager(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                backup_dir=temp_backup_dir
            )
            yield manager
            manager.close()
    
    def test_initialization(self, temp_backup_dir):
        """Test rollback manager initialization."""
        with patch('scripts.rollback_neo4j.GraphDatabase.driver'):
            manager = Neo4jRollbackManager(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                backup_dir=temp_backup_dir
            )
            
            assert manager.neo4j_uri == "bolt://localhost:7687"
            assert manager.neo4j_user == "neo4j"
            assert manager.backup_dir == Path(temp_backup_dir)
            
            manager.close()
    
    def test_initialization_missing_backup_dir(self):
        """Test initialization with missing backup directory."""
        with patch('scripts.rollback_neo4j.GraphDatabase.driver'):
            with pytest.raises(FileNotFoundError):
                Neo4jRollbackManager(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    backup_dir="/nonexistent/path"
                )
    
    def test_list_backups(self, rollback_manager, temp_backup_dir):
        """Test listing available backups."""
        backups = rollback_manager.list_backups()
        
        assert len(backups) == 1
        assert backups[0]["backup_name"] == "test_backup"
    
    def test_get_latest_backup(self, rollback_manager):
        """Test getting latest backup."""
        latest = rollback_manager.get_latest_backup()
        
        assert latest is not None
        assert latest["backup_name"] == "test_backup"
    
    def test_verify_backup(self, rollback_manager):
        """Test backup verification."""
        assert rollback_manager.verify_backup("test_backup") is True
        assert rollback_manager.verify_backup("nonexistent") is False
    
    def test_clear_database(self, rollback_manager, mock_driver):
        """Test clearing database."""
        session = mock_driver.session.return_value.__enter__.return_value
        
        # Mock SHOW INDEXES and SHOW CONSTRAINTS
        indexes_result = MagicMock()
        indexes_result.__iter__.return_value = [{"name": "test_index"}]
        
        constraints_result = MagicMock()
        constraints_result.__iter__.return_value = [{"name": "test_constraint"}]
        
        def run_side_effect(query):
            if "SHOW INDEXES" in query:
                return indexes_result
            elif "SHOW CONSTRAINTS" in query:
                return constraints_result
            return MagicMock()
        
        session.run.side_effect = run_side_effect
        
        # Clear database
        rollback_manager.clear_database()
        
        # Verify deletion queries were executed
        calls = [str(call) for call in session.run.call_args_list]
        assert any("DELETE r" in str(call) for call in calls)
        assert any("DELETE n" in str(call) for call in calls)
    
    def test_restore_from_cypher(self, rollback_manager, mock_driver, temp_backup_dir):
        """Test restoring from Cypher file."""
        session = mock_driver.session.return_value.__enter__.return_value
        
        cypher_file = Path(temp_backup_dir) / "test_backup_export.cypher"
        
        # Restore from Cypher
        rollback_manager.restore_from_cypher(cypher_file)
        
        # Verify Cypher statements were executed
        assert session.run.called
    
    def test_rollback_dry_run(self, rollback_manager):
        """Test rollback in dry run mode."""
        result = rollback_manager.rollback(
            backup_name="test_backup",
            dry_run=True
        )
        
        assert result["dry_run"] is True
        assert result["verification_passed"] is True
        assert result["backup_name"] == "test_backup"
    
    def test_rollback_latest(self, rollback_manager, mock_driver):
        """Test rollback to latest backup."""
        session = mock_driver.session.return_value.__enter__.return_value
        
        # Mock database stats
        labels_result = MagicMock()
        labels_result.__iter__.return_value = [{"label": "Paper"}]
        
        count_result = MagicMock()
        count_result.single.return_value = {"count": 10}
        
        rel_types_result = MagicMock()
        rel_types_result.__iter__.return_value = [{"relationshipType": "HAS_TAXON"}]
        
        indexes_result = MagicMock()
        indexes_result.__iter__.return_value = []
        
        constraints_result = MagicMock()
        constraints_result.__iter__.return_value = []
        
        def run_side_effect(query):
            if "db.labels()" in query:
                return labels_result
            elif "count(n)" in query or "count(r)" in query:
                return count_result
            elif "db.relationshipTypes()" in query:
                return rel_types_result
            elif "SHOW INDEXES" in query:
                return indexes_result
            elif "SHOW CONSTRAINTS" in query:
                return constraints_result
            return MagicMock()
        
        session.run.side_effect = run_side_effect
        
        # Rollback to latest
        result = rollback_manager.rollback(backup_name=None, skip_verification=False)
        
        assert result["backup_name"] == "test_backup"
        assert "rollback_timestamp" in result
        assert "previous_state" in result
        assert "restored_state" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
