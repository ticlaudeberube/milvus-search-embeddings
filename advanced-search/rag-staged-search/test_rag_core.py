import pytest
from unittest.mock import Mock, patch
from rag_core import needs_retrieval, rag_query_with_retrieval, direct_response, optimized_rag_query

def test_needs_retrieval():
    """Test classification logic"""
    mock_llm = Mock()
    
    # Test Milvus question
    mock_llm.invoke.return_value = "YES"
    assert needs_retrieval(mock_llm, "How to create Milvus collection?") == True
    
    # Test greeting
    mock_llm.invoke.return_value = "NO"
    assert needs_retrieval(mock_llm, "Hello") == False

def test_rag_with_retrieval():
    """Test RAG with document retrieval"""
    mock_client = Mock()
    mock_llm = Mock()
    
    # Mock search results
    mock_search_results = [
        {"entity": {"text": "Milvus collection creation guide"}},
        {"entity": {"text": "Vector database concepts"}}
    ]
    mock_client.search.return_value = [mock_search_results]
    mock_llm.invoke.return_value = "To create a collection in Milvus..."
    
    with patch('rag_core.MilvusUtils.embed_text_ollama', return_value=[0.1, 0.2]):
        response, doc_count = rag_query_with_retrieval(
            mock_client, mock_llm, "test_collection", "How to create collection?", []
        )
    
    assert doc_count == 2
    assert "To create a collection in Milvus..." in response
    mock_client.search.assert_called_once()

def test_direct_response():
    """Test direct response without retrieval"""
    mock_llm = Mock()
    mock_llm.invoke.return_value = "Hello! How can I help you?"
    
    response = direct_response(mock_llm, "Hi there", [])
    
    assert response == "Hello! How can I help you?"
    mock_llm.invoke.assert_called_once()

def test_optimized_rag_query():
    """Test the full two-stage pipeline"""
    mock_client = Mock()
    mock_llm = Mock()
    
    # Test direct response path
    with patch('rag_core.needs_retrieval', return_value=False), \
         patch('rag_core.direct_response', return_value="Direct answer"):
        
        response, doc_count = optimized_rag_query(
            mock_client, mock_llm, "test_collection", "Hello", []
        )
        
        assert response == "Direct answer"
        assert doc_count == 0
    
    # Test retrieval path
    with patch('rag_core.needs_retrieval', return_value=True), \
         patch('rag_core.rag_query_with_retrieval', return_value=("RAG answer", 3)):
        
        response, doc_count = optimized_rag_query(
            mock_client, mock_llm, "test_collection", "What is Milvus?", []
        )
        
        assert response == "RAG answer"
        assert doc_count == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])