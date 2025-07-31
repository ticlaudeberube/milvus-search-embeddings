import os, sys
import pytest
from pymilvus import MilvusException, MilvusClient
from unittest.mock import patch, MagicMock

from core import get_client, create_database, create_collection, drop_database, EmbeddingProvider, vectorize_documents, drop_collection, has_collection, insert_data
from core.exceptions import DatabaseError, EmbeddingError

db_name='test_db'
# Test cases

def test_get_client():
    client = get_client()
    assert client is not None
    assert isinstance(client, MilvusClient)

def test_create_database_new():
    with patch.object(get_client(), 'list_databases') as mock_list_db:
        with patch.object(get_client(), 'create_database') as mock_create_db:
            mock_list_db.return_value = []
            create_database(db_name)
            mock_create_db.assert_called_once_with(db_name)

def test_create_database_existing():
    with patch.object(get_client(), 'list_databases') as mock_list_db:
        with patch.object(get_client(), 'using_database') as mock_using_db:
            with patch.object(get_client(), 'drop_database') as mock_drop_db:
                with patch.object(get_client(), 'create_database') as mock_create_db:
                    with patch('pymilvus.utility.list_collections') as mock_list_collections:
                        mock_list_db.return_value = [db_name]
                        mock_list_collections.return_value = []
                        create_database(db_name)
                        mock_using_db.assert_called_once_with(db_name)
                        mock_drop_db.assert_called_once_with(db_name)
                        mock_create_db.assert_called_once_with(db_name)

def test_create_database_exception():
    with patch.object(get_client(), 'list_databases') as mock_list_db:
        mock_list_db.side_effect = MilvusException(400, 'Test error', 0)
        with pytest.raises(DatabaseError):
            create_database(db_name)

def test_create_collection():
    client = get_client()
    # Test creating a new collection
    collection_name = "test_collection"
    create_collection(collection_name)
    assert client.has_collection(collection_name=collection_name)
    
    # Test recreating an existing collection
    create_collection(collection_name) 
    assert client.has_collection(collection_name=collection_name)
    
    # Test with invalid collection name
    try:
        create_collection("")
        assert False, "Should raise exception for empty collection name"
    except Exception:
        assert True
        
    # Cleanup
    client.drop_collection(collection_name)

def test_drop_collection():
    client = get_client()
    # Test setup
    test_collection = "test_collection"
    client.create_collection(
        collection_name=test_collection,
        dimension=768
    )
    
    # Test collection exists
    assert client.has_collection(collection_name=test_collection) == True
    
    # Execute delete
    drop_collection(test_collection)
    
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
    create_collection(test_collection)
    
    # Test insert
    result = insert_data(test_collection, test_data)
    
    # Verify result contains expected fields
    assert isinstance(result, dict)
    assert "insert_count" in result
    assert result["insert_count"] == 2
    
    # Cleanup
    drop_collection(test_collection)

def test_vectorize_documents():
    # Test setup
    collection_name = "test_collection"
    test_docs = ["This is a test document", "This is another test document"]
    
    # Delete collection if it exists to ensure clean state
    client = get_client()
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    try:
        # Call function being tested - this will create the collection with correct dimensions
        result, dimension = vectorize_documents(collection_name, test_docs)
        
        # Assertions
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "insert_count" in result, "Result should contain insert_count"
        assert result["insert_count"] == len(test_docs), "Insert count should match number of docs"
        assert isinstance(dimension, int), "Dimension should be an integer"
        
    finally:
        # Cleanup
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)


def test_drop_database():
    with patch.object(get_client(), 'drop_database') as mock_drop_db:
        drop_database('test_db')
        mock_drop_db.assert_called_once_with(db_name='test_db')

def test_drop_database_empty_name():
    with pytest.raises(DatabaseError, match="db_name is required"):
        drop_database('')

def test_has_collection():
    with patch.object(get_client(), 'has_collection') as mock_has_collection:
        mock_has_collection.return_value = True
        result = has_collection('test_collection')
        assert result is True
        mock_has_collection.assert_called_once_with(collection_name='test_collection')

def test_embed_text_huggingface():
    with patch.dict(os.environ, {'HF_EMBEDDING_MODEL': 'test-model'}):
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_encoder = mock_st.return_value
            mock_embedding = MagicMock()
            mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
            mock_encoder.encode.return_value = [mock_embedding]
            
            result = EmbeddingProvider.embed_text('test text', provider='huggingface')
            assert result == [0.1, 0.2, 0.3]
            mock_st.assert_called_once_with('test-model')

def test_embed_text_ollama():
    with patch.dict(os.environ, {'OLLAMA_EMBEDDING_MODEL': 'test-model'}):
        with patch('core.embeddings.ollama.embeddings') as mock_ollama:
            mock_ollama.return_value = {'embedding': [0.1, 0.2, 0.3]}
            
            result = EmbeddingProvider.embed_text('test text', provider='ollama')
            assert result == [0.1, 0.2, 0.3]
            mock_ollama.assert_called_once_with(model='test-model', prompt='test text')

def test_embed_text_invalid_provider():
    with pytest.raises(EmbeddingError, match="Unsupported embedding provider: invalid"):
        EmbeddingProvider.embed_text('test', provider='invalid')

def test_get_device():
    with patch('torch.backends.mps.is_available') as mock_mps:
        mock_mps.return_value = True
        device = EmbeddingProvider.get_device()
        assert str(device) == 'mps'
        
        mock_mps.return_value = False
        device = EmbeddingProvider.get_device()
        assert str(device) == 'cpu'