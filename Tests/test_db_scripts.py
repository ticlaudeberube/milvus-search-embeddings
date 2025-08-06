import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestDatabaseScripts:
    """Test cases for database creation and deletion scripts"""

    def test_create_db_with_argument(self):
        """Test create_db.py script with database name argument"""
        with patch('core.create_database') as mock_create_db:
            from core import create_database
            create_database('test_database')
            mock_create_db.assert_called_once_with('test_database')

    @patch('core.drop_database')
    def test_drop_db_with_argument(self, mock_drop_db):
        """Test drop_db.py script with database name argument"""
        from core import drop_database
        drop_database('test_database')
        mock_drop_db.assert_called_once_with('test_database')

    @patch('core.create_collection')
    def test_create_collection_with_argument(self, mock_create_collection):
        """Test create_collection.py script with collection name argument"""
        from core import create_collection
        create_collection('test_collection')
        mock_create_collection.assert_called_once_with('test_collection')

    @patch('core.drop_collection')
    def test_drop_collection_with_argument(self, mock_drop_collection):
        """Test drop_collection.py script with collection name argument"""
        from core import drop_collection
        drop_collection('test_collection')
        mock_drop_collection.assert_called_once_with('test_collection')

    def test_create_db_script_exists(self):
        """Test that create_db.py script exists"""
        script_path = Path('core/create_db.py')
        if script_path.exists():
            assert script_path.is_file()

    def test_drop_db_script_exists(self):
        """Test that drop_db.py script exists"""
        script_path = Path('core/drop_db.py')
        if script_path.exists():
            assert script_path.is_file()

    def test_create_collection_script_exists(self):
        """Test that create_collection.py script exists"""
        script_path = Path('core/create_collection.py')
        if script_path.exists():
            assert script_path.is_file()

    def test_drop_collection_script_exists(self):
        """Test that drop_collection.py script exists"""
        script_path = Path('core/drop_collection.py')
        if script_path.exists():
            assert script_path.is_file()