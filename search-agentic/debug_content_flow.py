#!/usr/bin/env python3
"""Debug the complete content flow from search to response"""

import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from orchestrator.rag_orchestrator import RAGOrchestrator

load_dotenv()

def debug_content_flow():
    """Debug the complete content flow"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        print("Error: OLLAMA_COLLECTION_NAME not found")
        return
    
    query = "What is Milvus?"
    
    print("=" * 60)
    print("DEBUGGING COMPLETE CONTENT FLOW")
    print("=" * 60)
    
    # Initialize components
    llm = OllamaLLM(model="llama3.2:1b", temperature=0.0)
    orchestrator = RAGOrchestrator(llm, collection_name)
    
    print(f"\nQuery: {query}")
    
    # Step 1: Classification
    print("\n1. CLASSIFICATION:")
    needs_retrieval = orchestrator.classification_agent.classify(query, [])
    print(f"   Needs retrieval: {needs_retrieval}")
    
    if needs_retrieval:
        # Step 2: Retrieval
        print("\n2. RETRIEVAL:")
        context, doc_count = orchestrator.retrieval_agent.retrieve(query)
        print(f"   Doc count: {doc_count}")
        print(f"   Context length: {len(context)}")
        print(f"   Context preview: {context[:300]}...")
        
        # Step 3: Response generation
        print("\n3. RESPONSE GENERATION:")
        
        # Test direct method first
        print("\n   3a. Direct RAG method:")
        try:
            direct_response = orchestrator.response_agent._generate_rag_answer(f"{context}|||{query}")
            print(f"   Direct response: {direct_response[:200]}...")
        except Exception as e:
            print(f"   Direct method error: {e}")
        
        # Test agent method
        print("\n   3b. Agent RAG method:")
        try:
            agent_response = orchestrator.response_agent.generate_rag_response(query, context)
            print(f"   Agent response: {agent_response[:200]}...")
        except Exception as e:
            print(f"   Agent method error: {e}")
        
        # Step 4: Full pipeline
        print("\n4. FULL PIPELINE:")
        try:
            final_response, final_count = orchestrator.process_query(query, [])
            print(f"   Final doc count: {final_count}")
            print(f"   Final response: {final_response[:300]}...")
        except Exception as e:
            print(f"   Pipeline error: {e}")
    
    else:
        print("\n2. DIRECT RESPONSE (no retrieval needed)")

if __name__ == "__main__":
    debug_content_flow()