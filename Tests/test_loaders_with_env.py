#!/usr/bin/env python3
"""Test document loaders with proper environment setup."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

from core import get_client, EmbeddingProvider, create_collection

def test_individual_loaders():
    """Test each loader individually."""
    print("Testing Individual Document Loaders")
    print("=" * 50)
    
    client = get_client()
    
    # Test 1: Milvus Docs with Ollama
    print("\n1. Testing Milvus Docs (Ollama)")
    try:
        # Check if docs exist
        docs_path = Path("document-loaders/milvus_docs/en")
        if not docs_path.exists():
            print("   Downloading Milvus docs...")
            from document_loaders import download_milvus_docs
            download_milvus_docs()
        
        # Test embedding
        test_text = "Milvus is a vector database"
        vector = EmbeddingProvider._embed_ollama(test_text)
        print(f"   [OK] Ollama embedding works (dim: {len(vector)})")
        
        # Test collection creation
        collection_name = "test_milvus_ollama"
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        
        create_collection(collection_name, dimension=len(vector))
        print(f"   [OK] Collection '{collection_name}' created")
        
        # Test data insertion
        data = [{"id": 1, "vector": vector, "text": test_text}]
        result = client.insert(collection_name=collection_name, data=data)
        print(f"   [OK] Data inserted: {result['insert_count']} records")
        
        # Clean up
        client.drop_collection(collection_name)
        print("   [OK] Cleanup completed")
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test 2: State of Union with Ollama
    print("\n2. Testing State of Union (Ollama)")
    try:
        # Test the load function
        collection_name = "test_state_union_ollama"
        
        # Download file if needed
        filename = 'state_of_the_union.txt'
        if not os.path.isfile(f"document-loaders/{filename}"):
            import wget
            url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'
            os.chdir("document-loaders")
            wget.download(url, out=filename)
            os.chdir("..")
            print(f"   Downloaded {filename}")
        
        # Test text splitting and embedding
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        os.chdir("document-loaders")
        loader = TextLoader(filename)
        documents = loader.load()
        
        with open(filename) as f:
            content = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, length_function=len
        )
        docs = text_splitter.split_text(content)
        os.chdir("..")
        
        print(f"   [OK] Split into {len(docs)} chunks")
        
        # Test embedding a few chunks
        sample_docs = docs[:3]
        vectors = []
        for doc in sample_docs:
            vector = EmbeddingProvider._embed_ollama(doc)
            vectors.append(vector)
        
        print(f"   [OK] Created {len(vectors)} embeddings")
        
        # Test collection and insertion
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        
        create_collection(collection_name, dimension=len(vectors[0]))
        
        data = []
        for i, (doc, vector) in enumerate(zip(sample_docs, vectors)):
            data.append({"id": i, "vector": vector, "text": doc})
        
        result = client.insert(collection_name=collection_name, data=data)
        print(f"   [OK] Inserted {result['insert_count']} records")
        
        # Test search
        query = "president"
        query_vector = EmbeddingProvider._embed_ollama(query)
        search_result = client.search(
            collection_name=collection_name,
            data=[query_vector],
            output_fields=["text"],
            limit=2
        )
        print(f"   [OK] Search returned {len(search_result[0])} results")
        
        # Clean up
        client.drop_collection(collection_name)
        print("   [OK] Cleanup completed")
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test 3: HuggingFace embeddings (if configured)
    print("\n3. Testing HuggingFace Embeddings")
    try:
        if not os.getenv("HF_EMBEDDING_MODEL"):
            print("   [SKIP] HF_EMBEDDING_MODEL not configured")
            return
        
        test_texts = ["This is a test", "Another test sentence"]
        vectors = EmbeddingProvider._embed_huggingface(test_texts)
        print(f"   [OK] HF embedding works (dim: {len(vectors[0])})")
        
        # Test collection
        collection_name = "test_hf_collection"
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        
        create_collection(collection_name, dimension=len(vectors[0]))
        
        data = []
        for i, (text, vector) in enumerate(zip(test_texts, vectors)):
            data.append({"id": i, "vector": vector, "text": text})
        
        result = client.insert(collection_name=collection_name, data=data)
        print(f"   [OK] Inserted {result['insert_count']} records")
        
        # Clean up
        client.drop_collection(collection_name)
        print("   [OK] Cleanup completed")
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")

def main():
    """Main test function."""
    print("Document Loader Test Suite")
    print("=" * 50)
    
    # Check environment
    print("Environment Variables:")
    env_vars = ["OLLAMA_EMBEDDING_MODEL", "HF_EMBEDDING_MODEL", "EMBEDDING_PROVIDER"]
    for var in env_vars:
        value = os.getenv(var, "Not set")
        print(f"  {var}: {value}")
    
    # Check Milvus connection
    try:
        client = get_client()
        print("\n[OK] Milvus connection successful")
    except Exception as e:
        print(f"\n[FAIL] Milvus connection failed: {e}")
        return
    
    # Run tests
    test_individual_loaders()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    main()