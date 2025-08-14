#!/usr/bin/env python3
"""Test script for classification agent with mock and real LLM tests"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from agents.classification_agent import ClassificationAgent
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import Runnable
from typing import Any, Dict, List, Optional

load_dotenv()

class MockLLM(Runnable):
    """Mock LLM for testing"""
    
    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> str:
        if isinstance(input, dict):
            question = input.get('question', '').lower()
        else:
            question = str(input).lower()
        
        if any(word in question for word in ['what is', 'tell me', 'features', 'milvus']):
            return "YES"
        return "NO"

def test_mock_classification():
    """Test classification with mock LLM"""
    llm = MockLLM()
    agent = ClassificationAgent(llm)
    
    test_cases = [
        ("What is Milvus?", True),
        ("Tell me more about its features", True),
        ("Hello", False),
        ("Thanks", False),
        ("Please tell me something about another subject", False),
        ("How does Milvus work?", True),
        ("Explain Milvus capabilities", True),
        ("Hi there", False),
        ("Can you suggest some keywords in mIlvus docs I could use for pattern matching?", True),
    ]
    
    print("Testing Classification Agent (Mock LLM):")
    print("-" * 50)
    
    passed = 0
    for question, expected in test_cases:
        result = agent.classify(question, [])
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} | '{question}' -> {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"Mock tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_real_classification():
    """Test classification with real LLM (specific failing case)"""
    try:
        llm = OllamaLLM(model="llama3.2:1b", temperature=0.0)
        agent = ClassificationAgent(llm)
        
        # Test the specific failing case
        question = "Please tell me something about another subject"
        result = agent.classify(question, [])
        
        print("\nTesting Classification Agent (Real LLM):")
        print("-" * 50)
        print(f"Question: {question}")
        print(f"Classification: {result}")
        print(f"Expected: False (should not retrieve docs)")
        
        if not result:
            print("✅ PASS: Correctly classified as non-retrieval")
            return True
        else:
            print("❌ FAIL: Incorrectly classified as retrieval")
            return False
            
    except Exception as e:
        print(f"Real LLM test failed: {e}")
        return False

def test_classification():
    """Run all classification tests"""
    print("=" * 60)
    print("CLASSIFICATION AGENT TESTS")
    print("=" * 60)
    
    mock_passed = test_mock_classification()
    real_passed = test_real_classification()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: Mock={mock_passed}, Real={real_passed}")
    print("=" * 60)

if __name__ == "__main__":
    test_classification()