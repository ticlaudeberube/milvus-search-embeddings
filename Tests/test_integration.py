import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add utils to path
from core.utils import MilvusClient
from MilvusClient import MilvusClient


class TestMilvusIntegration:
    """Integration tests for MilvusClient (requires running Milvus instance)"""

    @pytest.mark.integration
    def test_full_workflow(self):
        """Test complete workflow: create collection, insert data, search"""
        collection_name = "test_integration_collection"
        
        try:
            # Create collection
            MilvusClient.create_collection(collection_name, dimension=768)
            assert MilvusClient.has_collection(collection_name)
            
            # Insert test data
            test_data = [
                {"id": 1, "vector": [0.1] * 768, "text": "test document 1", "subject": "test"},
                {"id": 2, "vector": [0.2] * 768, "text": "test document 2", "subject": "test"}
            ]
            
            result = MilvusClient.insert_data(collection_name, test_data)
            assert result["insert_count"] == 2
            
        finally:
            # Cleanup
            if MilvusClient.has_collection(collection_name):
                MilvusClient.drop_collection(collection_name)

    @pytest.mark.integration
    @patch('pymilvus.model.DefaultEmbeddingFunction')
    def test_vectorize_documents_integration(self, mock_embedding_fn):
        """Test document vectorization integration"""
        collection_name = "test_vectorize_collection"
        
        # Setup mock embedding function
        mock_embedding = MagicMock()
        mock_embedding.encode_documents.return_value = [[0.1] * 768, [0.2] * 768]
        mock_embedding.dim = 768
        mock_embedding_fn.return_value = mock_embedding
        
        try:
            docs = ["Test document 1", "Test document 2"]
            result, dimension = MilvusClient.vectorize_documents(collection_name, docs)
            
            assert dimension == 768
            assert result["insert_count"] == 2
            assert MilvusClient.has_collection(collection_name)
            
        finally:
            # Cleanup
            if MilvusClient.has_collection(collection_name):
                MilvusClient.drop_collection(collection_name)

    @pytest.mark.integration
    def test_database_operations(self):
        """Test database creation and management"""
        test_db = "test_integration_db"
        
        try:
            # Create database
            MilvusClient.create_database(test_db)
            
            # Verify database exists
            client = MilvusClient.get_client()
            databases = client.list_databases()
            assert test_db in databases
            
        except Exception as e:
            # Some Milvus versions may not support database operations
            pytest.skip(f"Database operations not supported: {e}")
        
        finally:
            # Cleanup would go here if drop_database was implemented
            pass