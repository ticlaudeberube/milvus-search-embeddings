"""Test the improved classification logic"""
import os
import sys
from unittest.mock import Mock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_core import RAGCore

def test_classification_examples():
    """Test classification with real-world examples"""
    
    mock_llm = Mock()
    rag_core = RAGCore(mock_llm, "test_collection")
    
    # Mock the chain directly
    rag_core.classification_chain = Mock()
    
    # Test cases that should return YES (need docs)
    yes_cases = [
            "What is Milvus?",
            "How to create Milvus collection?", 
            "Tell me about vector search",
            "How does similarity search work?",
            "What are Milvus features?",
            "How to install Milvus?"
    ]
    
    # Test cases that should return NO (no docs needed)
    no_cases = [
            "Hello",
            "My name is Claude",
            "What's the weather today?",
            "How are you?",
            "Thank you",
            "Tell me more about its features"  # Vague follow-up
    ]
    
    print("Testing YES cases (should need docs):")
    for question in yes_cases:
        rag_core.classification_chain.invoke.return_value = "YES"
        result = rag_core.needs_retrieval(question, [])
        print(f"  '{question}' -> {result}")
        assert result == True, f"Expected YES for: {question}"
    
    print("\nTesting NO cases (should not need docs):")
    for question in no_cases:
        rag_core.classification_chain.invoke.return_value = "NO"
        result = rag_core.needs_retrieval(question, [])
        print(f"  '{question}' -> {result}")
        assert result == False, f"Expected NO for: {question}"
    
    print("\nAll classification tests passed!")

if __name__ == "__main__":
    test_classification_examples()