#!/usr/bin/env python3
"""Debug script to examine search result content"""

import os
from dotenv import load_dotenv
from tools.milvus_tool import MilvusTool
from core import get_client

load_dotenv()

def debug_search_content():
    """Debug what's actually in the search results"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        print("Error: OLLAMA_COLLECTION_NAME not found")
        return
    
    query = "What is Milvus?"
    
    print("=" * 60)
    print("DEBUGGING SEARCH CONTENT")
    print("=" * 60)
    
    # Test 1: Raw client search
    print("\n1. RAW CLIENT SEARCH:")
    try:
        from core import EmbeddingProvider
        client = get_client()
        embedding = EmbeddingProvider.embed_text(query, provider="ollama")
        
        raw_results = client.search(
            collection_name=collection_name,
            data=[embedding],
            limit=3,
            search_params={"metric_type": "COSINE", "params": {"ef": 32}},
            output_fields=["text"]
        )
        
        print(f"Raw results type: {type(raw_results)}")
        print(f"Raw results length: {len(raw_results) if raw_results else 0}")
        
        if raw_results and raw_results[0]:
            for i, res in enumerate(raw_results[0][:2]):
                print(f"\nResult {i+1}:")
                print(f"  Type: {type(res)}")
                print(f"  Keys: {res.keys() if hasattr(res, 'keys') else 'No keys'}")
                print(f"  Entity keys: {res.get('entity', {}).keys() if 'entity' in res else 'No entity'}")
                
                text_content = res.get("entity", {}).get("text", "NO TEXT FOUND")
                print(f"  Text length: {len(text_content)}")
                print(f"  Text preview: {text_content[:200]}...")
                print(f"  Score: {res.get('distance', 'NO SCORE')}")
        
    except Exception as e:
        print(f"Raw search error: {e}")
    
    # Test 2: MilvusTool search
    print("\n2. MILVUS TOOL SEARCH:")
    try:
        milvus_tool = MilvusTool(collection_name)
        tool_results = milvus_tool.search(query, limit=3)
        
        print(f"Tool results type: {type(tool_results)}")
        print(f"Tool results length: {len(tool_results)}")
        
        for i, res in enumerate(tool_results[:2]):
            print(f"\nTool Result {i+1}:")
            print(f"  Keys: {res.keys()}")
            text_content = res.get("text", "NO TEXT")
            print(f"  Text length: {len(text_content)}")
            print(f"  Text preview: {text_content[:200]}...")
            print(f"  Score: {res.get('score', 'NO SCORE')}")
            
    except Exception as e:
        print(f"Tool search error: {e}")
    
    # Test 3: Collection info
    print("\n3. COLLECTION INFO:")
    try:
        client = get_client()
        client.load_collection(collection_name)
        
        # Get a few entities to see what's stored
        results = client.query(
            collection_name=collection_name,
            expr="",
            limit=2,
            output_fields=["text"]
        )
        
        print(f"Query results: {len(results)} entities")
        for i, entity in enumerate(results[:2]):
            text_content = entity.get("text", "NO TEXT")
            print(f"Entity {i+1} text length: {len(text_content)}")
            print(f"Entity {i+1} preview: {text_content[:200]}...")
            
    except Exception as e:
        print(f"Collection query error: {e}")

if __name__ == "__main__":
    debug_search_content()