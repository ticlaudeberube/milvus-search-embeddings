"""Collection operations for Milvus."""

from typing import Any, Dict, List, Tuple
from pymilvus import MilvusException, model
from .client import get_client
from .exceptions import CollectionError

def create_collection(collection_name: str | None, dimension: int = 1536, 
                     metric_type: str = "COSINE", consistency_level: str = "Session") -> None:
    """Create or recreate a collection."""
    if not collection_name:
        raise CollectionError("collection_name is required")
    
    try:
        client = get_client()
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type,
            consistency_level=consistency_level
        )
        print(f"Collection - {collection_name} - created successfully")
    except MilvusException as e:
        raise CollectionError(f"Failed to create collection '{collection_name}': {e}")

def drop_collection(collection_name: str | None) -> None:
    """Drop a collection."""
    if not collection_name:
        raise CollectionError("collection_name is required")
    
    try:
        client = get_client()
        client.drop_collection(collection_name=collection_name)
        print(f"Collection - {collection_name} - dropped successfully")
    except MilvusException as e:
        raise CollectionError(f"Failed to drop collection '{collection_name}': {e}")

def has_collection(collection_name: str) -> bool:
    """Check if collection exists."""
    client = get_client()
    return client.has_collection(collection_name=collection_name)

def insert_data(collection_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Insert data into collection."""
    client = get_client()
    return client.insert(collection_name=collection_name, data=data)

def vectorize_documents(collection_name: str, docs: List[str]) -> Tuple[Dict[str, Any], int]:
    """Vectorize documents and insert into collection."""
    # Use default embedding model
    embedding_fn = model.DefaultEmbeddingFunction()
    vectors = embedding_fn.encode_documents(docs)
    
    print("Dim:", embedding_fn.dim, vectors[0].shape)
    dimension = embedding_fn.dim
    
    # Create collection with correct dimensions
    create_collection(collection_name, dimension=dimension)
    
    # Prepare data
    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
        for i in range(len(vectors))
    ]
    
    print("Data has", len(data), "entities, each with fields:", data[0].keys())
    print("Vector dim:", len(data[0]["vector"]))
    
    # Insert data
    res = insert_data(collection_name, data)
    return res, dimension