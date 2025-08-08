"""Embedding providers for text vectorization."""

import os
import torch
from typing import Optional, Union, List
from sentence_transformers import SentenceTransformer
import ollama
from .config import get_embedding_config
from .exceptions import EmbeddingError

class EmbeddingProvider:
    @staticmethod
    def embed_text(text: Union[str, List[str]], provider: str = 'huggingface', model: Optional[str] = None):
        """Unified embedding method supporting multiple providers."""
        if provider == 'huggingface':
            return EmbeddingProvider._embed_huggingface(text, model)
        elif provider == 'ollama':
            return EmbeddingProvider._embed_ollama(text, model)
        else:
            raise EmbeddingError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def _embed_huggingface(text: Union[str, List[str]], model: Optional[str] = None):
        """Embed text using HuggingFace SentenceTransformers."""
        _model = model or os.getenv('HF_EMBEDDING_MODEL')
        if not _model:
            raise EmbeddingError("HF_EMBEDDING_MODEL environment variable not set")
        
        st = SentenceTransformer(_model)
        text_input = [text] if isinstance(text, str) else text
        embeddings = st.encode(text_input, batch_size=256, show_progress_bar=True)
        return embeddings[0].tolist() if isinstance(text, str) else embeddings.tolist()
    
    @staticmethod
    def _embed_ollama(text: Union[str, List[str]], model: Optional[str] = None):
        """Embed text using Ollama."""
        # Ensure OLLAMA_NUM_THREADS is always set
        if not os.getenv('OLLAMA_NUM_THREADS'):
            os.environ['OLLAMA_NUM_THREADS'] = '4'
        
        _model = model or os.getenv('OLLAMA_EMBEDDING_MODEL')
        if not _model:
            raise EmbeddingError("OLLAMA_EMBEDDING_MODEL environment variable not set")
        
        if isinstance(text, list):
            return [ollama.embeddings(model=_model, prompt=t)["embedding"] for t in text]
        return ollama.embeddings(model=_model, prompt=text)["embedding"]
    
    @staticmethod
    def get_device() -> torch.device:
        """Get optimal device for embeddings."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("WARNING: MPS not available. Falling back to CPU.")
            return torch.device("cpu")