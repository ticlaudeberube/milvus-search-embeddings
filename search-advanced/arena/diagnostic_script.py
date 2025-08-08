#!/usr/bin/env python3
"""Diagnostic script for configuration provider issues"""

import os
from config import ModelProvider

def diagnose_config():
    """Diagnose configuration issues"""
    print("=== Configuration Diagnosis ===\n")
    
    # Check environment variables
    print("1. Environment Variables:")
    env_vars = [
        "HF_LLM_MODEL", "HF_EMBEDDING_MODEL", "HF_TOKEN",
        "OLLAMA_LLM_MODEL", "OLLAMA_EMBEDDING_MODEL", 
        "MILVUS_HOST", "MILVUS_PORT",
        "MAX_TOKENS_HF", "MAX_TOKENS_OLLAMA"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "NOT SET")
        print(f"   {var}: {value}")
    
    print("\n2. Provider Registry:")
    print(model_registry.debug_info())
    
    print("\n3. Provider Configs:")
    for provider in ModelProvider:
        try:
            config = model_registry.get_provider(provider.value)
            print(f"   {provider.value}:")
            print(f"     Model: {config.default_model}")
            print(f"     Embedding: {config.embedding_model}")
            print(f"     Collection: {config.collection_name}")
        except Exception as e:
            print(f"   {provider.value}: ERROR - {e}")
    
    print("\n4. Issues Found:")
    issues = []
    
    # Check for missing models
    for provider in ModelProvider:
        try:
            config = model_registry.get_provider(provider.value)
            if not config.default_model:
                issues.append(f"{provider.value}: Missing default_model")
            if not config.embedding_model:
                issues.append(f"{provider.value}: Missing embedding_model")
        except Exception as e:
            issues.append(f"{provider.value}: Config error - {e}")
    
    if not issues:
        print("   No critical issues found!")
    else:
        for issue in issues:
            print(f"   ❌ {issue}")

if __name__ == "__main__":
    diagnose_config()