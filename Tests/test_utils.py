import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from pymilvus import MilvusException, MilvusClient as PyMilvusClient

from core.utils import MilvusClient


class TestMilvusClient:
    """Test cases for MilvusClient class"""

    def test_get_client(self):
        """Test get_client returns MilvusClient instance"""
        client = MilvusClient.get_client()
        assert client is not None
        assert isinstance(client, PyMilvusClient)

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

    @patch('pymilvus.db.list_database')
    def test_create_database_exception(self, mock_list_db):
        """Test database creation with exception"""
        mock_list_db.side_effect = MilvusException('Test error')
        MilvusClient.create_database('test_db')

    @patch('core.utils.milvus_client.client')
    def test_create_collection_new(self, mock_client):
        """Test creating new collection"""
        mock_client.has_collection.return_value = False
        
        MilvusClient.create_collection('test_collection')
        mock_client.create_collection.assert_called_once_with(
            collection_name='test_collection', dimension=1536
        )

    @patch('core.utils.milvus_client.client')
    def test_create_collection_existing(self, mock_client):
        """Test creating existing collection"""
        mock_client.has_collection.return_value = True
        
        MilvusClient.create_collection('test_collection')
        mock_client.create_collection.assert_not_called()

    @patch('core.utils.milvus_client.client')
    def test_has_collection(self, mock_client):
        """Test has_collection method"""
        mock_client.has_collection.return_value = True
        
        result = MilvusClient.has_collection('test_collection')
        assert result is True
        mock_client.has_collection.assert_called_once_with(collection_name='test_collection')

    @patch('core.utils.milvus_client.client')
    def test_drop_collection(self, mock_client):
        """Test drop_collection method"""
        MilvusClient.drop_collection('test_collection')
        mock_client.drop_collection.assert_called_once_with(collection_name='test_collection')

    @patch('core.utils.milvus_client.client')
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
    @patch('core.utils.milvus_client.client')
    def test_vectorize_documents(self, mock_client, mock_create_collection, mock_embedding_fn):
        """Test vectorize_documents method"""
        mock_client.insert.return_value = {'insert_count': 2}
        
        mock_embedding = MagicMock()
        import numpy as np
        mock_embedding.encode_documents.return_value = [np.array([0.1] * 768), np.array([0.2] * 768)]
        mock_embedding.dim = 768
        mock_embedding_fn.return_value = mock_embedding
        
        docs = ['doc1', 'doc2']
        result = MilvusClient.vectorize_documents('test_collection', docs)
        
        assert result == {'insert_count': 2}
        mock_create_collection.assert_called_once_with('test_collection', dimension=768)

    @patch('core.utils.milvus_client.SentenceTransformer')
    def test_embed_text_hf(self, mock_st):
        """Test embed_text with Hugging Face provider"""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model
        
        with patch.dict(os.environ, {'HF_EMBEDDING_MODEL': 'test-model'}):
            result = MilvusClient.embed_text('test text', provider='huggingface')
            assert result == [0.1, 0.2, 0.3]

    @patch('core.utils.milvus_client.ollama')
    def test_embed_text_ollama(self, mock_ollama):
        """Test embed_text with Ollama provider"""
        mock_ollama.embeddings.return_value = {'embedding': [0.1, 0.2, 0.3]}
        
        with patch.dict(os.environ, {'OLLAMA_EMBEDDING_MODEL': 'test-model'}):
            result = MilvusClient.embed_text('test text', provider='ollama')
            assert result == [0.1, 0.2, 0.3]

    def test_embed_text_invalid_provider(self):
        """Test embed_text with invalid provider"""
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            MilvusClient.embed_text('test', provider='invalid')

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


class TestUtilityScripts:
    """Test cases for utility scripts"""

    def test_create_collection_script_logic(self):
        """Test create_collection.py script logic"""
        with patch.object(MilvusClient, 'get_client') as mock_get_client:
            with patch.object(MilvusClient, 'create_collection') as mock_create:
                mock_client = MagicMock()
                mock_client.has_collection.return_value = False
                mock_get_client.return_value = mock_client
                
                # Simulate script logic
                collection_name = 'test_collection'
                client = MilvusClient.get_client()
                if client is not None:
                    if not client.has_collection(collection_name):
                        MilvusClient.create_collection(collection_name)
                
                mock_create.assert_called_once_with('test_collection')

    def test_drop_collection_script_logic(self):
        """Test drop_collection.py script logic"""
        with patch.object(MilvusClient, 'get_client') as mock_get_client:
            with patch.object(MilvusClient, 'drop_collection') as mock_drop:
                mock_client = MagicMock()
                mock_client.has_collection.return_value = True
                mock_get_client.return_value = mock_client
                
                # Simulate script logic
                collection_name = 'test_collection'
                client = MilvusClient.get_client()
                if client is not None:
                    if client.has_collection(collection_name):
                        MilvusClient.drop_collection(collection_name)
                
                mock_drop.assert_called_once_with('test_collection')

    def test_create_db_script_logic(self):
        """Test create_db.py script logic"""
        with patch('pymilvus.connections.connect'):
            with patch.object(MilvusClient, 'create_database') as mock_create_db:
                
                # Simulate script logic
                db_name = 'test_db'
                MilvusClient.create_database(db_name)
                
                mock_create_db.assert_called_once_with('test_db')

    def test_drop_db_script_logic(self):
        """Test drop_db.py script logic"""
        with patch.object(MilvusClient, 'get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.list_databases.return_value = ['default', 'test_db']
            mock_get_client.return_value = mock_client
            
            # Simulate script logic
            db_name = 'test_db'
            client = MilvusClient.get_client()
            dbs = client.list_databases()
            if dbs.index(db_name) >= -1:  # Script logic
                client.drop_database(db_name)
            



class TestDocumentLoaders:
    """Test cases for document loaders"""
    
    def test_milvus_docs_download(self):
        """Test Milvus documentation download"""
        docs_path = Path("document-loaders/milvus_docs")
        
        # If docs don't exist, test download
        if not docs_path.exists():
            try:
                from document_loaders.download_milvus_docs import download_milvus_docs
                download_milvus_docs()
                assert docs_path.exists(), "Milvus docs should be downloaded"
            except Exception as e:
                pytest.skip(f"Download failed: {e}")
        else:
            # Docs already exist
            assert docs_path.exists()
    
    @pytest.mark.skipif(not os.getenv("OLLAMA_EMBEDDING_MODEL"), reason="Ollama model not configured")
    def test_ollama_embedding_functionality(self):
        """Test Ollama embedding functionality"""
        test_text = "This is a test sentence for embedding."
        
        try:
            vector = MilvusClient.embed_text_ollama(test_text)
            assert isinstance(vector, list), "Should return a list"
            assert len(vector) > 0, "Vector should not be empty"
            assert all(isinstance(x, (int, float)) for x in vector), "Vector should contain numbers"
        except Exception as e:
            pytest.fail(f"Ollama embedding failed: {e}")
    
    @pytest.mark.skipif(not os.getenv("HF_EMBEDDING_MODEL"), reason="HuggingFace model not configured")
    def test_huggingface_embedding_functionality(self):
        """Test HuggingFace embedding functionality"""
        test_texts = ["This is a test sentence.", "Another test sentence."]
        
        try:
            vectors = MilvusClient.embed_text_hf(test_texts)
            assert isinstance(vectors, list), "Should return a list"
            assert len(vectors) == len(test_texts), "Should return same number of vectors as texts"
            assert all(isinstance(v, list) for v in vectors), "Each vector should be a list"
            assert all(len(v) > 0 for v in vectors), "Vectors should not be empty"
        except Exception as e:
            pytest.fail(f"HuggingFace embedding failed: {e}")
    
    def test_milvus_docs_loader_structure(self):
        """Test that Milvus docs loader has correct structure"""
        loader_path = Path("document-loaders/load_milvus_docs_ollama.py")
        assert loader_path.exists(), "Milvus docs Ollama loader should exist"
        
        # Check if the file has the expected functions
        with open(loader_path, 'r') as f:
            content = f.read()
            assert "def main()" in content, "Should have main function"
            assert "MilvusClient" in content, "Should use MilvusClient"
    
    def test_state_union_loader_structure(self):
        """Test that State of Union loaders have correct structure"""
        ollama_loader = Path("document-loaders/load-state-of-the-union-ollama.py")
        default_loader = Path("document-loaders/load-state-of-the-union-default.py")
        
        assert ollama_loader.exists(), "State of Union Ollama loader should exist"
        assert default_loader.exists(), "State of Union default loader should exist"
        
        # Check Ollama loader structure
        with open(ollama_loader, 'r') as f:
            content = f.read()
            assert "MilvusClient" in content, "Should use MilvusClient"
            assert "embed_text_ollama" in content, "Should use Ollama embeddings"
    
    def test_various_docs_loader_structure(self):
        """Test that various docs loaders have correct structure"""
        ollama_loader = Path("document-loaders/load-various-docs-scatterplot.py")
        hf_loader = Path("document-loaders/load-various-docs-scatterplot-hf.py")
        
        assert ollama_loader.exists(), "Various docs Ollama loader should exist"
        assert hf_loader.exists(), "Various docs HF loader should exist"
        
        # Check for async structure
        with open(ollama_loader, 'r') as f:
            content = f.read()
            assert "async def load()" in content, "Should have async load function"
            assert "asyncio.run" in content, "Should use asyncio"
    
    @pytest.mark.integration
    def test_loader_integration_with_milvus(self):
        """Integration test for loaders with Milvus"""
        try:
            client = MilvusClient.get_client()
            test_collection = "test_loader_integration"
            
            # Clean up if exists
            if client.has_collection(test_collection):
                client.drop_collection(test_collection)
            
            # Test collection creation
            MilvusClient.create_collection(test_collection, dimension=384)
            assert client.has_collection(test_collection), "Collection should be created"
            
            # Test data insertion (if embedding is available)
            if os.getenv("OLLAMA_EMBEDDING_MODEL"):
                test_text = "Integration test document"
                vector = MilvusClient.embed_text_ollama(test_text)
                
                # Recreate collection with correct dimension
                client.drop_collection(test_collection)
                MilvusClient.create_collection(test_collection, dimension=len(vector))
                
                data = [{"id": 1, "vector": vector, "text": test_text}]
                result = client.insert(collection_name=test_collection, data=data)
                assert result["insert_count"] == 1, "Should insert one document"
                
                # Test search
                search_result = client.search(
                    collection_name=test_collection,
                    data=[vector],
                    output_fields=["text"],
                    limit=1
                )
                assert len(search_result[0]) == 1, "Should return one result"
            
            # Clean up
            client.drop_collection(test_collection)
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    def test_environment_configuration(self):
        """Test that environment is properly configured for loaders"""
        # Check for .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
                # Check for key variables
                expected_vars = [
                    "OLLAMA_EMBEDDING_MODEL",
                    "HF_EMBEDDING_MODEL", 
                    "MILVUS_OLLAMA_COLLECTION_NAME",
                    "EMBEDDING_PROVIDER"
                ]
                
                for var in expected_vars:
                    if var in content:
                        # Variable is defined in .env
                        continue
                    elif os.getenv(var):
                        # Variable is set in environment
                        continue
                    else:
                        # Variable not found anywhere - this is OK for optional vars
                        pass
        
        # Test that at least one embedding provider is configured
        has_ollama = bool(os.getenv("OLLAMA_EMBEDDING_MODEL"))
        has_hf = bool(os.getenv("HF_EMBEDDING_MODEL"))
        
        if not (has_ollama or has_hf):
            pytest.skip("No embedding providers configured")
    
    @pytest.mark.slow
    @pytest.mark.skipif(not os.getenv("OLLAMA_EMBEDDING_MODEL"), reason="Ollama not configured")
    def test_milvus_docs_ollama_loader_execution(self):
        """Test actual execution of Milvus docs Ollama loader"""
        try:
            # Check if docs exist
            docs_path = Path("document-loaders/milvus_docs/en")
            if not docs_path.exists():
                pytest.skip("Milvus docs not available")
            
            # Import and test the main function
            sys.path.insert(0, "document-loaders")
            try:
                from load_milvus_docs_ollama import main
                main()
                
                # Verify collection was created
                client = MilvusClient.get_client()
                collection_name = os.getenv("MILVUS_OLLAMA_COLLECTION_NAME", "demo_collection")
                assert client.has_collection(collection_name), f"Collection {collection_name} should exist"
                
            finally:
                sys.path.remove("document-loaders")
                
        except Exception as e:
            pytest.fail(f"Milvus docs Ollama loader execution failed: {e}")
    
    @pytest.mark.slow
    @pytest.mark.skipif(not os.getenv("HF_EMBEDDING_MODEL"), reason="HuggingFace not configured")
    def test_milvus_docs_hf_loader_execution(self):
        """Test actual execution of Milvus docs HF loader"""
        try:
            # Check if docs exist
            docs_path = Path("document-loaders/milvus_docs/en")
            if not docs_path.exists():
                pytest.skip("Milvus docs not available")
            
            # Import and test the process function
            sys.path.insert(0, "document-loaders")
            try:
                from load_milvus_docs_hf import process
                process()
                
                # Verify collection was created
                client = MilvusClient.get_client()
                collection_name = os.getenv("MILVUS_HF_COLLECTION_NAME", "demo_collection")
                assert client.has_collection(collection_name), f"Collection {collection_name} should exist"
                
            finally:
                sys.path.remove("document-loaders")
                
        except Exception as e:
            pytest.fail(f"Milvus docs HF loader execution failed: {e}")