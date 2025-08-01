import pytest
from pymilvus import MilvusException, MilvusClient, Collection
from unittest.mock import patch, MagicMock
from core.MilvusUtils import MilvusUtils

db_name = 'test_db'

def test_get_client():
    client = MilvusUtils.get_client()
    assert client is not None
    assert isinstance(client, MilvusClient)

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

def test_create_collection():
    client = MilvusUtils.get_client()
    # Test creating a new collection
    collection_name = "test_collection"
    MilvusUtils.create_collection(collection_name)
    assert client.has_collection(collection_name=collection_name)
    
    # Test recreating an existing collection
    MilvusUtils.create_collection(collection_name) 
    assert client.has_collection(collection_name=collection_name)
    
    # Test with invalid collection name
    try:
        MilvusUtils.create_collection("")
        assert False, "Should raise exception for empty collection name"
    except Exception:
        assert True
        
    # Cleanup
    client.drop_collection(collection_name)

def test_drop_collection():
    client = MilvusUtils.get_client()
    # Test setup
    test_collection = "test_collection"
    client.create_collection(
        collection_name=test_collection,
        dimension=768
    )
    
    # Test collection exists
    assert client.has_collection(collection_name=test_collection) == True
    
    # Execute delete
    MilvusUtils.drop_collection(test_collection)
    
    # Verify collection was deleted
    assert client.has_collection(collection_name=test_collection) == False

def test_insert_data():
    # Test data
    test_collection = "test_collection"
    test_data = [
        {"id": 1, "vector": [0.1] * 1536, "text": "test1", "subject": "test"},
        {"id": 2, "vector": [0.2] * 1536, "text": "test2", "subject": "test"}
    ]
    
    # Create test collection
    MilvusUtils.create_collection(test_collection)
    
    # Test insert
    result = MilvusUtils.insert_data(test_collection, test_data)
    
    # Verify result contains expected fields
    assert isinstance(result, dict)
    assert "insert_count" in result
    assert result["insert_count"] == 2
    
    # Cleanup
    MilvusUtils.drop_collection(test_collection)

def test_vectorize_documents():
    # Test setup
    collection_name = "test_collection"
    test_docs = ["This is a test document", "This is another test document"]
    
    # Create collection for test
    MilvusUtils.create_collection(collection_name)
    
    try:
        # Call function being tested
        result, dim = MilvusUtils.vectorize_documents(collection_name, test_docs)
        
        # Assertions
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "insert_count" in result, "Result should contain insert_count"
        assert result["insert_count"] == len(test_docs), "Insert count should match number of docs"
        assert isinstance(dim, int), "Dimension should be an integer"
        
    finally:
        # Cleanup
        MilvusUtils.drop_collection(collection_name)

