#!/usr/bin/env python3
"""Test State of Union loader specifically."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_state_union_ollama():
    """Test State of Union with Ollama embeddings."""
    print("Testing State of Union Loader (Ollama)")
    print("=" * 50)
    
    try:
        from core.utils import MilvusClient
        import wget
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from tqdm import tqdm
        
        client = MilvusClient.get_client()
        collection = "test_state_of_the_union_ollama"
        
        # Download file if needed
        filename = 'state_of_the_union.txt'
        filepath = Path("document-loaders") / filename
        
        if not filepath.exists():
            print("Downloading State of Union document...")
            url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'
            os.makedirs("document-loaders", exist_ok=True)
            wget.download(url, out=str(filepath))
            print(f"\nDownloaded {filename}")
        
        # Load and split document
        print("Loading and splitting document...")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_text(content)
        print(f"Split into {len(docs)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        data = []
        for i, doc in enumerate(tqdm(docs[:50], desc="Creating embeddings")):  # Test with first 50 chunks
            vector = MilvusClient.embed_text_ollama(doc)
            data.append({"id": i, "vector": vector, "text": doc})
        
        # Create collection
        dimension = len(data[0]['vector'])
        if client.has_collection(collection):
            client.drop_collection(collection)
        
        MilvusClient.create_collection(collection_name=collection, dimension=dimension)
        print(f"Created collection '{collection}' with dimension {dimension}")
        
        # Insert data
        result = client.insert(collection_name=collection, data=data)
        print(f"Inserted {result['insert_count']} documents")
        
        # Test search
        query = "What did the president say about Ketanji Brown Jackson?"
        print(f"\nTesting search with query: '{query}'")
        
        query_vector = MilvusClient.embed_text_ollama(query)
        search_result = client.search(
            collection_name=collection,
            data=[query_vector],
            output_fields=["text"],
            limit=3,
            search_params={"metric_type": "COSINE"}
        )
        
        print(f"Search returned {len(search_result[0])} results:")
        for i, result in enumerate(search_result[0]):
            print(f"  {i+1}. Score: {result['distance']:.4f}")
            print(f"     Text: {result['entity']['text'][:100]}...")
        
        # Clean up
        client.drop_collection(collection)
        print(f"\nCleaned up collection '{collection}'")
        
        print("\n[SUCCESS] State of Union loader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_state_union_ollama()
    if success:
        print("\nState of Union loader is working correctly!")
    else:
        print("\nState of Union loader test failed.")

if __name__ == "__main__":
    main()