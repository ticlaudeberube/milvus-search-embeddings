"""Core Milvus utilities package."""

# New modular interface
from .client import get_client, reset_client
from .embeddings import EmbeddingProvider
from .collections import create_collection, drop_collection, has_collection, insert_data, vectorize_documents
from .databases import create_database, drop_database, list_databases
from .config import get_milvus_config, get_embedding_config
from .exceptions import MilvusConnectionError, DatabaseError, CollectionError, EmbeddingError

# Backward compatibility - MilvusUtils class
class MilvusUtils:
    """Legacy interface for backward compatibility."""
    
    # Client methods
    get_client = staticmethod(get_client)
    
    # Database methods
    create_database = staticmethod(create_database)
    drop_database = staticmethod(drop_database)
    
    # Collection methods
    create_collection = staticmethod(create_collection)
    drop_collection = staticmethod(drop_collection)
    has_collection = staticmethod(has_collection)
    insert_data = staticmethod(insert_data)
    vectorize_documents = staticmethod(vectorize_documents)
    
    # Embedding methods
    embed_text = staticmethod(EmbeddingProvider.embed_text)
    get_device = staticmethod(EmbeddingProvider.get_device)
    
    # Deprecated methods for compatibility
    @staticmethod
    def embed_text_hf(sentences, model=None):
        """Deprecated: Use embed_text(provider='huggingface') instead"""
        return EmbeddingProvider.embed_text(sentences, provider='huggingface', model=model)
    
    @staticmethod
    def embed_text_ollama(text, model=None):
        """Deprecated: Use embed_text(provider='ollama') instead"""
        return EmbeddingProvider.embed_text(text, provider='ollama', model=model)

# Export everything for easy access
__all__ = [
    # New interface
    'get_client', 'reset_client',
    'EmbeddingProvider',
    'create_collection', 'drop_collection', 'has_collection', 'insert_data', 'vectorize_documents',
    'create_database', 'drop_database', 'list_databases',
    'get_milvus_config', 'get_embedding_config',
    'MilvusConnectionError', 'DatabaseError', 'CollectionError', 'EmbeddingError',
    # Legacy interface
    'MilvusUtils'
]