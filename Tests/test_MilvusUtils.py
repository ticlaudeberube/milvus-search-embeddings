
from pymilvus import MilvusException
from unittest.mock import patch, MagicMock
from MilvusUtils import MilvusUtils
db_name='test_db'
# Test cases
def test_create_database_new():
    with patch('pymilvus.db.list_database') as mock_list_db:
        with patch('pymilvus.db.create_database') as mock_create_db:
            mock_list_db.return_value = []
            MilvusUtils.create_database(db_name)
            mock_create_db.assert_called_once_with(db_name)

def test_create_database_existing():
    with patch('pymilvus.db.list_database') as mock_list_db:
        with patch('pymilvus.db.using_database') as mock_using_db:
            with patch('pymilvus.utility.list_collections') as mock_list_collections:
                with patch('pymilvus.db.drop_database') as mock_drop_db:
                    mock_list_db.return_value = [db_name]
                    mock_list_collections.return_value = ['collection1']
                    
                    collection_mock = MagicMock()
                    with patch('pymilvus.Collection', return_value=collection_mock):
                        MilvusUtils.create_database(db_name)
                        
                        mock_using_db.assert_called_once_with(db_name)
                        #collection_mock.drop.assert_called_once()
                        #mock_drop_db.assert_called_once_with(db_name)

def test_create_database_exception():
    with patch('pymilvus.db.list_database') as mock_list_db:
        mock_list_db.side_effect = MilvusException('Test error')
        MilvusUtils.create_database(db_name)

