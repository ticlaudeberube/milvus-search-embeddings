#!/usr/bin/env python3
"""Test script for manual testing scenarios"""

import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import get_client
from rag_core import RAGCore
from langchain_ollama.llms import OllamaLLM

def setup_rag_system():
    """Initialize RAG system for testing"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        print("Error: OLLAMA_COLLECTION_NAME not found")
        return None
    
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b"),
        temperature=0.0,
        num_predict=500,  # Allow longer responses for detailed summaries
        top_k=5,
        top_p=0.8,
        num_thread=8
    )
    
    client = get_client()
    client.load_collection(collection_name)
    return RAGCore(llm, collection_name)

def test_scenarios():
    """Test all manual scenarios"""
    rag_core = setup_rag_system()
    if not rag_core:
        return
    
    # Test scenarios with expected results
    scenarios = [
        ("How is data stored in milvus?", "YES", "First technical question"),
        ("My name is Claude. From now on always include my name in the answer.", "NO", "Personal instruction"),
        ("What is Milvus?", "YES", "Technical question"),
        ("How is data stored in milvus?", "YES", "Repeat question - should still get docs"),
        ("What is the weather today?", "NO", "Off-topic question"),
        ("How vectors are used to retrieve context data?", "YES", "Technical question"),
        ("Tell me more about its features?", "NO", "Vague follow-up"),
        ("Hello there", "NO", "Greeting"),
        ("How does Milvus indexing work?", "YES", "Technical question"),
        ("Can you resume our conversation?", "NO", "Conversation management")
    ]
    
    chat_history = []
    passed = 0
    failed = 0
    
    print("=" * 80)
    print("TESTING MANUAL SCENARIOS")
    print("=" * 80)
    
    for i, (question, expected, description) in enumerate(scenarios, 1):
        print(f"\n{i}. {description}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        
        scenario_start = time.time()
        
        try:
            # Test classification
            needs_docs = rag_core.needs_retrieval(question, chat_history)
            classification = "YES" if needs_docs else "NO"
            
            # Get response
            response, doc_count = rag_core.query(question, chat_history)
            
            # Check result
            if classification == expected:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1
            
            scenario_time = time.time() - scenario_start
            print(f"Actual: {classification} ({doc_count} docs) - {status} (took {scenario_time:.2f}s)")
            print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            print(f"History length: {len(chat_history)} items")
            
            # Add to history like GUI does (exclude only off-topic redirects)
            is_redirect = "I'm specialized in" in response and "I don't have information" in response
            if not is_redirect:
                chat_history.append({"question": question, "answer": response})
            
            # Special handling for resume test
            if "resume" in question.lower() and len(chat_history) > 1:
                print(f"[INFO] Resume request with {len(chat_history)-1} previous exchanges")
            
        except Exception as e:
            scenario_time = time.time() - scenario_start
            print(f"ERROR: {str(e)} (took {scenario_time:.2f}s)")
            failed += 1
            # Continue with next scenario even if this one fails
            continue
        
        print("-" * 40)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED out of {len(scenarios)} scenarios")
    print(f"Total conversation history: {len(chat_history)} exchanges")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        test_scenarios()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()