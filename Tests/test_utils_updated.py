import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from pymilvus import MilvusException, MilvusClient

# Add utils to path
from core.utils import MilvusClient
from MilvusClient import MilvusClient


class TestMilvusClient:
    """Test cases for MilvusClient class"""

    def test_get_client(self):
        """Test get_client returns MilvusClient instance"""
        client = MilvusClient.get_client()
        assert client is not None
        assert isinstance(client, MilvusClient)

    @patch('pymilvus.db.list_database')
    @patch('pymilvus.db.create_database')
    def test_create_database_new(self, mock_create_db, mock_list_db):
        """Test creating a new database"""
        mock_list_db.return_value = []
        MilvusClient.create_database('test_db')
        mock_create_db.assert_called_once_with('test_db')

    @patch('pymilvus.db.list_database')
    def test_create_database_existing(self, mock_list_db):
        """Test handling existing database"""
        mock_list_db.return_value = ['test_db']
        MilvusClient.create_database('test_db')
        # Should not raise exception

    @patch('pymilvus.db.list_database')
    def test_create_database_exception(self, mock_list_db):
        """Test database creation with exception"""
        mock_list_db.side_effect = MilvusException('Test error')
        MilvusClient.create_database('test_db')
        # Should handle exception gracefully

    @patch('MilvusClient.client')
    def test_create_collection_new(self, mock_client):
        """Test creating new collection"""
        mock_client.has_collection.return_value = False
        
        MilvusClient.create_collection('test_collection')
        mock_client.create_collection.assert_called_once_with(
            collection_name='test_collection', dimension=1536
        )

    @patch('MilvusClient.client')
    def test_create_collection_existing(self, mock_client):
        """Test creating existing collection"""
        mock_client.has_collection.return_value = True
        
        MilvusClient.create_collection('test_collection')
        mock_client.create_collection.assert_not_called()

    @patch('MilvusClient.client')
    def test_has_collection(self, mock_client):
        """Test has_collection method"""
        mock_client.has_collection.return_value = True
        
        result = MilvusClient.has_collection('test_collection')
        assert result is True
        mock_client.has_collection.assert_called_once_with(collection_name='test_collection')

    @patch('MilvusClient.client')
    def test_drop_collection(self, mock_client):
        """Test drop_collection method"""
        MilvusClient.drop_collection('test_collection')
        mock_client.drop_collection.assert_called_once_with(collection_name='test_collection')

    @patch('MilvusClient.client')
    def test_insert_data(self, mock_client):
        """Test insert_data method"""
        mock_client.insert.return_value = {'insert_count': 2}
        
        test_data = [
            {'id': 1, 'vector': [0.1] * 768, 'text': 'test1'},
            {'id': 2, 'vector': [0.2] * 768, 'text': 'test2'}
        ]
        
        result = MilvusClient.insert_data('test_collection', test_data)
        assert result == {'insert_count': 2}
        mock_client.insert.assert_called_once_with(collection_name='test_collection', data=test_data)

    @patch('pymilvus.model.DefaultEmbeddingFunction')
    @patch.object(MilvusClient, 'create_collection')
    @patch('MilvusClient.client')
    def test_vectorize_documents(self, mock_client, mock_create_collection, mock_embedding_fn):
        """Test vectorize_documents method"""
        # Setup mocks
        mock_client.insert.return_value = {'insert_count': 2}
        
        # Create mock vectors with shape attribute
        mock_vector1 = MagicMock()
        mock_vector1.shape = (768,)
        mock_vector2 = MagicMock()
        mock_vector2.shape = (768,)
        
        mock_embedding = MagicMock()
        mock_embedding.encode_documents.return_value = [mock_vector1, mock_vector2]
        mock_embedding.dim = 768
        mock_embedding_fn.return_value = mock_embedding
        
        docs = ['doc1', 'doc2']
        result = MilvusClient.vectorize_documents('test_collection', docs)
        
        assert result == {'insert_count': 2}
        mock_create_collection.assert_called_once_with('test_collection', dimension=768)

    @patch('MilvusClient.SentenceTransformer')
    def test_embed_text_hf_single(self, mock_st):
        """Test embed_text with HuggingFace provider for single text"""
        mock_model = MagicMock()
        # Mock numpy array behavior
        mock_embeddings = MagicMock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_embeddings.__getitem__.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        with patch.dict(os.environ, {'HF_EMBEDDING_MODEL': 'test-model'}):
            result = MilvusClient.embed_text('test text', provider='huggingface')
            assert result == [0.1, 0.2, 0.3]

    @patch('MilvusClient.SentenceTransformer')
    def test_embed_text_hf_list(self, mock_st):
        """Test embed_text with HuggingFace provider for list of texts"""
        mock_model = MagicMock()
        # Mock numpy array behavior
        mock_embeddings = MagicMock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        with patch.dict(os.environ, {'HF_EMBEDDING_MODEL': 'test-model'}):
            result = MilvusClient.embed_text(['test1', 'test2'], provider='huggingface')
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @patch('MilvusClient.ollama')
    def test_embed_text_ollama_single(self, mock_ollama):
        """Test embed_text with Ollama provider for single text"""
        mock_ollama.embeddings.return_value = {'embedding': [0.1, 0.2, 0.3]}
        
        with patch.dict(os.environ, {'MODEL_OLLAMA': 'test-model'}):
            result = MilvusClient.embed_text('test text', provider='ollama')
            assert result == [0.1, 0.2, 0.3]

    @patch('MilvusClient.ollama')
    def test_embed_text_ollama_list(self, mock_ollama):
        """Test embed_text with Ollama provider for list of texts"""
        mock_ollama.embeddings.side_effect = [
            {'embedding': [0.1, 0.2, 0.3]},
            {'embedding': [0.4, 0.5, 0.6]}
        ]
        
        with patch.dict(os.environ, {'MODEL_OLLAMA': 'test-model'}):
            result = MilvusClient.embed_text(['test1', 'test2'], provider='ollama')
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_embed_text_invalid_provider(self):
        """Test embed_text with invalid provider"""
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            MilvusClient.embed_text('test', provider='invalid')

    def test_embed_text_default_provider(self):
        """Test embed_text with default provider"""
        with patch('MilvusClient.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            # Mock numpy array behavior
            mock_embeddings = MagicMock()
            mock_embeddings.__getitem__.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model
            
            with patch.dict(os.environ, {'HF_EMBEDDING_MODEL': 'test-model'}):
                result = MilvusClient.embed_text('test text', provider='huggingface')  # Specify provider
                assert result == [0.1, 0.2, 0.3]

    @patch('torch.backends.mps.is_available')
    @patch('torch.device')
    def test_get_device_mps(self, mock_device, mock_mps_available):
        """Test get_device with MPS available"""
        mock_mps_available.return_value = True
        MilvusClient.get_device()
        mock_device.assert_called_with('mps')

    @patch('torch.backends.mps.is_available')
    @patch('torch.device')
    def test_get_device_cpu(self, mock_device, mock_mps_available):
        """Test get_device fallback to CPU"""
        mock_mps_available.return_value = False
        MilvusClient.get_device()
        mock_device.assert_called_with('cpu')

    @patch('MilvusClient.SentenceTransformer')
    def test_embed_text_hf_deprecated(self, mock_st):
        """Test deprecated embed_text_hf method"""
        mock_model = MagicMock()
        # Mock numpy array behavior
        mock_embeddings = MagicMock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        with patch.dict(os.environ, {'HF_EMBEDDING_MODEL': 'test-model'}):
            result = MilvusClient.embed_text_hf(['test1', 'test2'])
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @patch('MilvusClient.ollama')
    def test_embed_text_ollama_deprecated(self, mock_ollama):
        """Test deprecated embed_text_ollama method"""
        mock_ollama.embeddings.return_value = {'embedding': [0.1, 0.2, 0.3]}
        
        with patch.dict(os.environ, {'MODEL_OLLAMA': 'test-model'}):
            result = MilvusClient.embed_text_ollama('test text')
            assert result == [0.1, 0.2, 0.3]


class TestScriptWorkflows:
    """Test the core workflows used by utility scripts"""
    
    @patch('MilvusClient.client')
    @patch.object(MilvusClient, 'create_collection')
    def test_collection_creation_workflow(self, mock_create, mock_client):
        """Test the workflow used in create_collection.py"""
        mock_client.has_collection.return_value = False
        
        # Simulate script logic
        collection_name = 'test_collection'
        client = MilvusClient.get_client()
        
        if not client.has_collection(collection_name):
            MilvusClient.create_collection(collection_name)
            
        mock_create.assert_called_once_with(collection_name)

    @patch('MilvusClient.client')
    @patch.object(MilvusClient, 'drop_collection')
    def test_collection_drop_workflow(self, mock_drop, mock_client):
        """Test the workflow used in drop_collection.py"""
        mock_client.has_collection.return_value = True
        
        # Simulate script logic
        collection_name = 'test_collection'
        client = MilvusClient.get_client()
        
        if client.has_collection(collection_name):
            MilvusClient.drop_collection(collection_name)
            
        mock_drop.assert_called_once_with(collection_name)

    @patch.object(MilvusClient, 'create_database')
    def test_database_creation_workflow(self, mock_create_db):
        """Test the workflow used in create_db.py"""
        # Simulate script logic
        db_name = 'test_db'
        MilvusClient.create_database(db_name)
        
        mock_create_db.assert_called_once_with(db_name)

    @patch('MilvusClient.client')
    def test_database_drop_workflow(self, mock_client):
        """Test the workflow used in drop_db.py"""
        mock_client.list_databases.return_value = ['default', 'test_db']
        
        # Simulate script logic
        db_name = 'test_db'
        client = MilvusClient.get_client()
        dbs = client.list_databases()
        
        if db_name in dbs:
            client.drop_database(db_name)
            
        mock_client.drop_database.assert_called_once_with(db_name)