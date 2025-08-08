import pytest
from unittest.mock import Mock, patch
from rag_core import RAGCore, optimized_rag_query

def test_needs_retrieval():
    """Test classification logic"""
    mock_llm = Mock()
    rag_core = RAGCore(mock_llm, "test_collection")
    
    # Mock the chain directly
    rag_core.classification_chain = Mock()
    
    # Test Milvus question
    rag_core.classification_chain.invoke.return_value = "YES"
    assert rag_core.needs_retrieval("How to create Milvus collection?", []) == True
    
    # Test greeting
    rag_core.classification_chain.invoke.return_value = "NO"
    assert rag_core.needs_retrieval("Hello", []) == False

@patch('rag_core.get_client')
@patch('rag_core.EmbeddingProvider.embed_text')
def test_rag_with_retrieval(mock_embed, mock_get_client):
    """Test RAG with document retrieval"""
    mock_llm = Mock()
    rag_core = RAGCore(mock_llm, "test_collection")
    
    # Mock the chain directly
    rag_core.rag_chain = Mock()
    rag_core.rag_chain.invoke.return_value = "To create a collection in Milvus..."
    
    # Mock client and search results
    mock_client = Mock()
    mock_search_results = [
        {"entity": {"text": "Milvus collection creation guide"}},
        {"entity": {"text": "Vector database concepts"}}
    ]
    mock_client.search.return_value = [mock_search_results]
    mock_get_client.return_value = mock_client
    mock_embed.return_value = [0.1, 0.2]
    response, doc_count = rag_core.rag_query_with_retrieval("How to create collection?", [])
    
    assert doc_count == 2
    assert "To create a collection in Milvus..." in response

def test_direct_response():
    """Test direct response without retrieval"""
    mock_llm = Mock()
    rag_core = RAGCore(mock_llm, "test_collection")
    
    # Mock the chain directly
    rag_core.direct_chain = Mock()
    rag_core.direct_chain.invoke.return_value = "Hello! How can I help you?"
    
    response = rag_core.direct_response("Hi there", [])
    
    assert response == "Hello! How can I help you?"

def test_optimized_rag_query():
    """Test the full two-stage pipeline"""
    mock_client = Mock()
    mock_llm = Mock()
    
    # Test direct response path
    with patch('rag_core.RAGCore') as mock_rag_core:
        mock_instance = Mock()
        mock_instance.query.return_value = ("Direct answer", 0)
        mock_rag_core.return_value = mock_instance
        
        response, doc_count = optimized_rag_query(
            mock_client, mock_llm, "test_collection", "Hello", []
        )
        
        assert response == "Direct answer"
        assert doc_count == 0
    
    # Test retrieval path
    with patch('rag_core.RAGCore') as mock_rag_core:
        mock_instance = Mock()
        mock_instance.query.return_value = ("RAG answer", 3)
        mock_rag_core.return_value = mock_instance
        
        response, doc_count = optimized_rag_query(
            mock_client, mock_llm, "test_collection", "What is Milvus?", []
        )
        
        assert response == "RAG answer"
        assert doc_count == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])