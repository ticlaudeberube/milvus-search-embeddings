#!/usr/bin/env python3
"""Test script for environment variables configuration."""
import os
from core import MilvusUtils

def test_env_vars():
    """Test environment variable configuration."""
    print("Testing Environment Variables Configuration\n")
    
    # Test basic env vars
    vars_to_test = [
        ("OLLAMA_LLM_MODEL", "llama3.2"),
        ("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5"),
        ("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        ("EMBEDDING_PROVIDER", "ollama"),
        ("MILVUS_OLLAMA_COLLECTION_NAME", "demo_collection"),
        ("MY_DATABASE", "default")
    ]
    
    for var_name, default_val in vars_to_test:
        value = os.getenv(var_name, default_val)
        status = "[SET]" if os.getenv(var_name) else "[DEFAULT]"
        print(f"{status} {var_name}: {value}")
    
    print("\nTesting MilvusClient embedding methods:")
    
    # Test embedding with different providers
    test_text = "Hello world"
    
    try:
        # Test Ollama embedding
        print(f"[OK] Ollama embedding: {len(MilvusUtils.embed_text(test_text, provider='ollama'))} dimensions")
    except Exception as e:
        print(f"[FAIL] Ollama embedding failed: {e}")
    
    try:
        # Test HuggingFace embedding
        print(f"[OK] HuggingFace embedding: {len(MilvusUtils.embed_text(test_text, provider='huggingface'))} dimensions")
    except Exception as e:
        print(f"[FAIL] HuggingFace embedding failed: {e}")

if __name__ == "__main__":
    test_env_vars()