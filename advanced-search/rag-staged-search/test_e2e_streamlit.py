#!/usr/bin/env python3
"""E2E test simulating Streamlit app usage"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.append('./advanced-search/rag-staged-search')

from core import MilvusUtils
from langchain_ollama.llms import OllamaLLM
from rag_core import optimized_rag_query

def simulate_streamlit_session():
    """Simulate a real Streamlit session with session state"""
    
    # Initialize like Streamlit app
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME", "milvus_ollama_collection")
    client = MilvusUtils.get_client()
    llm = OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"))
    
    # Simulate session_state.chat_history
    session_chat_history = []
    
    # E2E conversation flow
    conversation = [
        "My name is Claude. From now on always include my name in the answer.",
        "What is Milvus?", 
        "Tell me more about its features?",
        "How do I install it?"
    ]
    
    print("E2E Streamlit Simulation Test")
    print("=" * 50)
    
    for i, question in enumerate(conversation, 1):
        print(f"\n[Step {i}] User asks: {question}")
        
        # Simulate Streamlit form submission
        if question.strip():
            try:
                # This is exactly what the Streamlit app does
                response, doc_count = optimized_rag_query(
                    client, llm, collection_name, question, session_chat_history
                )
                
                # Show results like Streamlit UI
                if doc_count > 0:
                    print(f"[INFO] Retrieved {doc_count} documents from knowledge base")
                else:
                    print("[INFO] Answered directly without document retrieval")
                
                # Add to session history like Streamlit
                session_chat_history.append({"question": question, "answer": response})
                
                # Show response preview
                print(f"[RESPONSE] {response[:150]}...")
                
                # Validation
                expected_retrieval = question in ["What is Milvus?", "Tell me more about its features?", "How do I install it?"]
                actual_retrieval = doc_count > 0
                
                status = "PASS" if expected_retrieval == actual_retrieval else "FAIL"
                print(f"[{status}] Expected retrieval: {expected_retrieval}, Got: {actual_retrieval}")
                
            except Exception as e:
                print(f"[ERROR] {e}")
        
        print("-" * 30)
    
    print(f"\nFinal session history: {len(session_chat_history)} items")

if __name__ == "__main__":
    simulate_streamlit_session()