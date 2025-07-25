
import sys
from pymilvus import MilvusException, MilvusClient as PyMilvusClient, Collection
from unittest.mock import patch, MagicMock

from core.utils import MilvusClient

db_name='test_db'
# Test cases

def test_get_client():
    client = MilvusClient.get_client()
    assert client is not None
    assert isinstance(client, PyMilvusClient)

def test_create_database_new():
    with patch('pymilvus.db.list_database') as mock_list_db:
        with patch('pymilvus.db.create_database') as mock_create_db:
            mock_list_db.return_value = []
            MilvusClient.create_database(db_name)
            mock_create_db.assert_called_once_with(db_name)

def test_create_database_existing():
    with patch('pymilvus.db.list_database') as mock_list_db:
        mock_list_db.return_value = [db_name]
        MilvusClient.create_database(db_name)
        # Just verify the method runs without error when database exists

def test_create_database_exception():
    with patch('pymilvus.db.list_database') as mock_list_db:
        mock_list_db.side_effect = MilvusException('Test error')
        MilvusClient.create_database(db_name)

def test_create_collection():
    client = MilvusClient.get_client()
    # Test creating a new collection
    collection_name = "test_collection"
    MilvusClient.create_collection(collection_name)
    assert client.has_collection(collection_name=collection_name)
    
    # Test recreating an existing collection
    MilvusClient.create_collection(collection_name) 
    assert client.has_collection(collection_name=collection_name)
    
    # Test with invalid collection name
    try:
        MilvusClient.create_collection("")
        assert False, "Should raise exception for empty collection name"
    except Exception:
        assert True
        
    # Cleanup
    client.drop_collection(collection_name)

def test_drop_collection():
    client = MilvusClient.get_client()
    # Test setup
    test_collection = "test_collection"
    client.create_collection(
        collection_name=test_collection,
        dimension=768
    )
    
    # Test collection exists
    assert client.has_collection(collection_name=test_collection) == True
    
    # Execute delete
    MilvusClient.drop_collection(test_collection)
    
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
    MilvusClient.create_collection(test_collection)
    
    # Test insert
    result = MilvusClient.insert_data(test_collection, test_data)
    
    # Verify result contains expected fields
    assert isinstance(result, dict)
    assert "insert_count" in result
    assert result["insert_count"] == 2
    
    # Cleanup
    MilvusClient.drop_collection(test_collection)

def test_vectorize_documents():
    # Test setup
    collection_name = "test_collection"
    test_docs = ["This is a test document", "This is another test document"]
    
    # Delete collection if it exists to ensure clean state
    if MilvusClient.has_collection(collection_name):
        MilvusClient.drop_collection(collection_name)
    
    result = None
    try:
        # Call function being tested - this will create the collection with correct dimensions
        result = MilvusClient.vectorize_documents(collection_name, test_docs)
        
        # Assertions
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "insert_count" in result, "Result should contain insert_count"
        assert result["insert_count"] == len(test_docs), "Insert count should match number of docs"
        
    finally:
        # Cleanup
        if MilvusClient.has_collection(collection_name):
            MilvusClient.drop_collection(collection_name)
    
    pass

