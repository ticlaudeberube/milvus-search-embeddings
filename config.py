"""Configuration module for model names and settings."""
import os
from typing import Dict, Any

# Default model configurations
DEFAULT_MODELS: Dict[str, Any] = {
    "ollama": {
        "llm": "llama3.2",
        "embedding": "nomic-embed-text:v1.5",
        "alternatives": {
            "llm": ["llama2", "mistral", "orca-mini", "neural-chat"],
            "embedding": ["mxbai-embed-large", "nomic-embed-text"]
        }
    },
    "huggingface": {
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
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

def get_HF_EMBEDDING_MODEL() -> str:
    """Get HuggingFace model from environment or default."""
    return os.getenv("HF_EMBEDDING_MODEL", DEFAULT_MODELS["huggingface"]["embedding"])

def get_embedding_provider() -> str:
    """Get embedding provider from environment or default."""
    return os.getenv("EMBEDDING_PROVIDER", "ollama")