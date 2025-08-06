import pytest
import sys
from unittest.mock import MagicMock

@pytest.fixture
def mock_milvus_client():
    """Mock MilvusClient for testing"""
    client = MagicMock()
    client.has_collection.return_value = False
    client.create_collection.return_value = None
    client.drop_collection.return_value = None
    client.insert.return_value = {'insert_count': 1}
    client.list_databases.return_value = ['default']
    return client

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "This is a test document about machine learning.",
        "Another document discussing vector databases.",
        "A third document about embeddings and search."
    ]

@pytest.fixture
def sample_vectors():
    """Sample vectors for testing"""
    return [
        [0.1] * 768,
        [0.2] * 768,
        [0.3] * 768
    ]

@pytest.fixture
def sample_data():
    """Sample data for insertion testing"""
    return [
        {"id": 1, "vector": [0.1] * 768, "text": "test1", "subject": "test"},
        {"id": 2, "vector": [0.2] * 768, "text": "test2", "subject": "test"},
        {"id": 3, "vector": [0.3] * 768, "text": "test3", "subject": "test"}
    ]

@pytest.fixture(autouse=True)
def reset_modules():
    """Reset imported modules before each test"""
    modules_to_remove = [
        'create_collection', 'create_db', 'drop_collection', 'drop_db'
    ]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
    yield