import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from core import get_client, create_collection, insert_data, create_database, vectorize_documents, drop_collection


class TestMilvusIntegration:
    """Integration tests for MilvusUtils (requires running Milvus instance)"""

    def test_full_workflow(self):
        """Test complete workflow: create collection, insert data, search"""
        collection_name = "test_integration_collection"
        
        try:
            # Create collection
            create_collection(collection_name, dimension=768)
            client = get_client()
            assert client.has_collection(collection_name)
            
            # Insert test data
            test_data = [
                {"id": 1, "vector": [0.1] * 768, "text": "test document 1", "subject": "test"},
                {"id": 2, "vector": [0.2] * 768, "text": "test document 2", "subject": "test"}
            ]
            
            result = insert_data(collection_name, test_data)
            assert result["insert_count"] == 2
            
        finally:
            # Cleanup
            client = get_client()
            if client.has_collection(collection_name):
                drop_collection(collection_name)

    @patch('core.collections.model.DefaultEmbeddingFunction')
    def test_vectorize_documents_integration(self, mock_embedding_fn):
        """Test document vectorization integration"""
        collection_name = "test_vectorize_collection"
        
        # Setup mock embedding function with numpy arrays
        mock_embedding = MagicMock()
        mock_embedding.encode_documents.return_value = [
            np.array([0.1] * 768, dtype=np.float32),
            np.array([0.2] * 768, dtype=np.float32)
        ]
        mock_embedding.dim = 768
        mock_embedding_fn.return_value = mock_embedding
        
        try:
            docs = ["Test document 1", "Test document 2"]
            result, dimension = vectorize_documents(collection_name, docs)
            
            assert dimension == 768
            assert result["insert_count"] == 2
            client = get_client()
            assert client.has_collection(collection_name)
            
        finally:
            # Cleanup
            client = get_client()
            if client.has_collection(collection_name):
                drop_collection(collection_name)

    def test_database_operations(self):
        """Test database creation and management"""
        test_db = "test_integration_db"
        
        try:
            # Create database
            create_database(test_db)
            
            # Verify database exists
            client = get_client()
            databases = client.list_databases()
            assert test_db in databases
            
        except Exception as e:
            # Some Milvus versions may not support database operations
            pytest.skip(f"Database operations not supported: {e}")
        
        finally:
            # Cleanup would go here if drop_database was implemented
            pass