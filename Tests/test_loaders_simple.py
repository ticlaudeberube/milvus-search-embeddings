#!/usr/bin/env python3
"""Simple test script for document loaders."""

import os
from pathlib import Path
from core import create_collection

def test_basic_functionality():
    """Test basic functionality of loaders."""
    print("Testing Document Loaders")
    print("=" * 40)
    
    # Test 1: Check if core utils work
    try:
        from core import get_client
        client = get_client()
        print("[OK] Milvus client connection successful")
    except Exception as e:
        print(f"[FAIL] Milvus connection failed: {e}")
        return False
    
    # Test 2: Check if docs exist
    docs_path = Path("document-loaders/milvus_docs/en")
    if docs_path.exists():
        print("[OK] Milvus docs found")
    else:
        print("[INFO] Milvus docs not found - will download")
        try:
            from document_loaders import download_milvus_docs
            download_milvus_docs()
            print("[OK] Milvus docs downloaded")
        except Exception as e:
            print(f"[FAIL] Failed to download docs: {e}")
    
    # Test 3: Test embedding functions
    test_text = "This is a test sentence."
    
    # Test Ollama embedding (if available)
    try:
        if os.getenv("OLLAMA_EMBEDDING_MODEL"):
            from core import EmbeddingProvider
            vector = EmbeddingProvider._embed_ollama(test_text)
            print(f"[OK] Ollama embedding works (dim: {len(vector)})")
        else:
            print("[SKIP] Ollama embedding - no model configured")
    except Exception as e:
        print(f"[FAIL] Ollama embedding failed: {e}")
    
    # Test HuggingFace embedding (if available)
    try:
        if os.getenv("HF_EMBEDDING_MODEL"):
            from core import EmbeddingProvider
            vector = EmbeddingProvider._embed_huggingface([test_text])
            print(f"[OK] HuggingFace embedding works (dim: {len(vector[0])})")
        else:
            print("[SKIP] HuggingFace embedding - no model configured")
    except Exception as e:
        print(f"[FAIL] HuggingFace embedding failed: {e}")
    
    # Test 4: Test collection operations
    try:
        test_collection = "test_loader_collection"
        
        # Clean up if exists
        if client.has_collection(test_collection):
            client.drop_collection(test_collection)
        
        # Create test collection
        create_collection(test_collection, dimension=384)
        print("[OK] Collection creation works")
        
        # Clean up
        client.drop_collection(test_collection)
        print("[OK] Collection cleanup works")
        
    except Exception as e:
        print(f"[FAIL] Collection operations failed: {e}")
    
    print("\nTest completed!")
    return True

def check_environment():
    """Check environment configuration."""
    print("\nEnvironment Check")
    print("=" * 40)
    
    env_vars = [
        "OLLAMA_EMBEDDING_MODEL",
        "HF_EMBEDDING_MODEL", 
        "HF_TOKEN",
        "MILVUS_OLLAMA_COLLECTION_NAME",
        "EMBEDDING_PROVIDER"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Hide sensitive tokens
            if "TOKEN" in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"[SET] {var} = {display_value}")
        else:
            print(f"[NOT SET] {var}")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("[OK] .env file found")
    else:
        print("[INFO] .env file not found - using system environment")

def main():
    """Main test function."""
    check_environment()
    test_basic_functionality()

if __name__ == "__main__":
    main()