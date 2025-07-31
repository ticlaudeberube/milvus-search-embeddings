"""Configuration module for model names and settings."""
import os
from typing import Dict, Any
from enum import Enum

class ModelProvider(Enum):
    """Supported model providers"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"

# Default model configurations
DEFAULT_MODELS: Dict[str, Any] = {
    "ollama": {
        "llm": "llama3.2",
        "embedding": "nomic-embed-text:v1.5",
        "collection": "milvus_ollama_collection",
        "alternatives": {
            "llm": ["llama2", "mistral", "orca-mini", "neural-chat"],
            "embedding": ["mxbai-embed-large", "nomic-embed-text"]
        }
    },
    "huggingface": {
        "llm": "mistralai/Mixtral-8x7B-v0.1",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "collection": "milvus_hf_collection",
        "alternatives": [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-albert-small-v2"
        ]
    }
}

def get_ollama_llm_model() -> str:
    """Get Ollama LLM model from environment or default."""
    return os.getenv("OLLAMA_LLM_MODEL", DEFAULT_MODELS["ollama"]["llm"])

def get_ollama_embedding_model() -> str:
    """Get Ollama embedding model from environment or default."""
    return os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_MODELS["ollama"]["embedding"])

def get_ollama_collection_name() -> str:
    """Get Ollama collection name from environment or default."""
    return os.getenv("OLLAMA_COLLECTION_NAME", DEFAULT_MODELS["ollama"]["collection"])

def get_hf_llm_model() -> str:
    """Get HuggingFace LLM model from environment or default."""
    return os.getenv("HF_LLM_MODEL", DEFAULT_MODELS["huggingface"]["llm"])

def get_hf_embedding_model() -> str:
    """Get HuggingFace embedding model from environment or default."""
    return os.getenv("HF_EMBEDDING_MODEL", DEFAULT_MODELS["huggingface"]["embedding"])

def get_hf_collection_name() -> str:
    """Get HuggingFace collection name from environment or default."""
    return os.getenv("HF_COLLECTION_NAME", DEFAULT_MODELS["huggingface"]["collection"])

def get_embedding_provider() -> str:
    """Get embedding provider from environment or default."""
    return os.getenv("EMBEDDING_PROVIDER", "ollama")

def get_milvus_host() -> str:
    """Get Milvus host from environment or default."""
    return os.getenv("MILVUS_HOST", "localhost")

def get_milvus_port() -> str:
    """Get Milvus port from environment or default."""
    return os.getenv("MILVUS_PORT", "19530")

def get_database_name() -> str:
    """Get database name from environment or default."""
    return os.getenv("MY_DB_NAME", "default")

# Backward compatibility
get_HF_EMBEDDING_MODEL = get_hf_embedding_model