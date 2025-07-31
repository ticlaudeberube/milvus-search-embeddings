#!/usr/bin/env python3
"""Test what happens when environment variables are missing."""
import sys, os

from core import MilvusUtils

# Clear environment variables to test missing scenarios
test_vars = ['HF_EMBEDDING_MODEL', 'OLLAMA_EMBEDDING_MODEL', 'EMBEDDING_PROVIDER']
original_values = {}

for var in test_vars:
    original_values[var] = os.environ.get(var)
    if var in os.environ:
        del os.environ[var]

print("Testing missing environment variables...\n")

try:
    # Test HuggingFace with missing HF_EMBEDDING_MODEL
    print("1. Testing HuggingFace with missing HF_EMBEDDING_MODEL:")
    result = MilvusUtils.embed_text("test", provider='huggingface')
    print(f"   Result: {len(result)} dimensions")
except Exception as e:
    print(f"   ERROR: {e}")

try:
    # Test Ollama with missing OLLAMA_EMBEDDING_MODEL
    print("2. Testing Ollama with missing OLLAMA_EMBEDDING_MODEL:")
    result = MilvusUtils.embed_text("test", provider='ollama')
    print(f"   Result: {len(result)} dimensions")
except Exception as e:
    print(f"   ERROR: {e}")

try:
    # Test deprecated method with missing MODEL_OLLAMA
    print("3. Testing deprecated embed_text_ollama with missing MODEL_OLLAMA:")
    result = MilvusUtils.embed_text_ollama("test")
    print(f"   Result: {len(result)} dimensions")
except Exception as e:
    print(f"   ERROR: {e}")

# Restore original values
for var, value in original_values.items():
    if value is not None:
        os.environ[var] = value