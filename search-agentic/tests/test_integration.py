#!/usr/bin/env python3
"""Comprehensive integration tests for the agentic RAG pipeline"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from orchestrator.rag_orchestrator import RAGOrchestrator

load_dotenv()

def setup_orchestrator():
    """Initialize RAG orchestrator for testing"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        print("Error: OLLAMA_COLLECTION_NAME not found")
        return None
    
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b"),
        temperature=0.0,
        num_predict=500,
        top_k=5,
        top_p=0.8,
        num_thread=8
    )
    
    return RAGOrchestrator(llm, collection_name)

def test_basic_pipeline():
    """Test basic pipeline functionality"""
    orchestrator = setup_orchestrator()
    if not orchestrator:
        return False
    
    test_cases = [
        ("What is Milvus?", "Technical - should retrieve docs"),
        ("Hello there", "Greeting - should be direct"),
        ("How does vector indexing work?", "Technical - should retrieve docs")
    ]
    
    chat_history = []
    
    print("=" * 60)
    print("TESTING BASIC PIPELINE")
    print("=" * 60)
    
    for i, (question, expected) in enumerate(test_cases, 1):
        print(f"\n{i}. {expected}")
        print(f"Q: {question}")
        
        try:
            response, doc_count = orchestrator.process_query(question, chat_history)
            
            classification = "RETRIEVAL" if doc_count > 0 else "DIRECT"
            print(f"Classification: {classification} ({doc_count} docs)")
            print(f"Response: {response[:100]}...")
            
            chat_history.append({"question": question, "answer": response})
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return False
        
        print("-" * 40)
    
    print(f"Total interactions: {len(chat_history)}")
    return True

def test_comprehensive_scenarios():
    """Test comprehensive scenarios with timing"""
    orchestrator = setup_orchestrator()
    if not orchestrator:
        return False
    
    scenarios = [
        ("How is data stored in milvus?", "YES", "Technical question"),
        ("My name is Claude. From now on always include my name in the answer.", "NO", "Personal instruction"),
        ("What is Milvus?", "YES", "Technical question"),
        ("What is the weather today?", "NO", "Off-topic question"),
        ("How vectors are used to retrieve context data?", "YES", "Technical question"),
        ("Hello there", "NO", "Greeting"),
        ("Can you resume our conversation?", "NO", "Conversation management")
    ]
    
    chat_history = []
    passed = 0
    failed = 0
    
    print("=" * 80)
    print("TESTING COMPREHENSIVE SCENARIOS")
    print("=" * 80)
    
    for i, (question, expected, description) in enumerate(scenarios, 1):
        print(f"\n{i}. {description}")
        print(f"Q: {question}")
        
        scenario_start = time.time()
        
        try:
            needs_docs = orchestrator.classification_agent.classify(question, chat_history)
            classification = "YES" if needs_docs else "NO"
            
            response, doc_count = orchestrator.process_query(question, chat_history)
            
            if classification == expected:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1
            
            scenario_time = time.time() - scenario_start
            print(f"Classification: {classification} ({doc_count} docs) - {status}")
            print(f"Time: {scenario_time:.2f}s")
            print(f"Response: {response[:100]}...")
            
            # Add to history (exclude redirects)
            is_redirect = "I'm specialized in" in response and "I don't have information" in response
            if not is_redirect:
                chat_history.append({"question": question, "answer": response})
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            failed += 1
        
        print("-" * 40)
    
    print(f"\nRESULTS: {passed} PASSED, {failed} FAILED out of {len(scenarios)}")
    print(f"Total conversation history: {len(chat_history)} exchanges")
    return failed == 0

def test_integration():
    """Run all integration tests"""
    print("=" * 80)
    print("AGENTIC RAG INTEGRATION TESTS")
    print("=" * 80)
    
    basic_passed = test_basic_pipeline()
    comprehensive_passed = test_comprehensive_scenarios()
    
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: Basic={basic_passed}, Comprehensive={comprehensive_passed}")
    print("=" * 80)
    
    return basic_passed and comprehensive_passed

if __name__ == "__main__":
    try:
        test_integration()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")