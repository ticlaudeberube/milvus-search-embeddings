#!/usr/bin/env python3
"""Test script to verify configuration works correctly"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class QAConfig:
    """Configuration for the QA system"""
    model_provider: str = "ollama"
    model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text:v1.5"
    collection_name: str = "milvus_ollama_collection"
    max_tokens: int = 512
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        from config import get_embedding_provider, get_ollama_llm_model, get_ollama_embedding_model, get_ollama_collection_name, get_hf_llm_model, get_hf_embedding_model, get_hf_collection_name
        
        self.model_provider = get_embedding_provider()
        
        if self.model_provider == "huggingface":
            self.model = get_hf_llm_model()
            self.embedding_model = get_hf_embedding_model()
            self.collection_name = get_hf_collection_name()
        else:  # ollama
            self.model = get_ollama_llm_model()
            self.embedding_model = get_ollama_embedding_model()
            self.collection_name = get_ollama_collection_name()

def main():
    print("Testing QAConfig...")
    
    # Test default config
    config = QAConfig()
    
    print(f"Provider: {config.model_provider}")
    print(f"Model: {config.model}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Collection: {config.collection_name}")
    print(f"Max Tokens: {config.max_tokens}")
    
    print("\nConfiguration test completed successfully!")

if __name__ == "__main__":
    main()