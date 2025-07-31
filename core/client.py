"""Milvus client connection management."""

from typing import Optional
from pymilvus import MilvusClient
from .config import get_milvus_config
from .exceptions import MilvusConnectionError

# Global client instance - initialized lazily
_client: Optional[MilvusClient] = None

def get_client() -> MilvusClient:
    """Get Milvus client with lazy initialization and connection validation."""
    global _client
    if _client is None:
        try:
            config = get_milvus_config()
            _client = MilvusClient(uri=config.uri, token=config.token)
            # Test connection
            _client.list_databases()
        except Exception as e:
            raise MilvusConnectionError(f"Failed to connect to Milvus: {e}")
    return _client

def reset_client() -> None:
    """Reset client connection (useful for testing)."""
    global _client
    _client = None