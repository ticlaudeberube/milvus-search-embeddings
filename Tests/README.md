# Tests for Core Package

This directory contains comprehensive unit tests for the `core` package components.

## Test Files

- `test_utils.py` - Main unit tests for MilvusClient class and utility scripts
- `test_MilvusUtils.py` - Legacy test file (being updated)
- `test_integration.py` - Integration tests (requires running Milvus instance)
- `conftest.py` - Pytest configuration and fixtures

## Running Tests

### Unit Tests Only (Recommended)
```bash
# Run all unit tests
pytest Tests/test_utils.py -v

# Run with coverage
pytest Tests/test_utils.py --cov=core --cov-report=term-missing

# Using the test runner script
python run_tests.py
```

### Integration Tests (Requires Milvus)
```bash
# Start Milvus first
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Run integration tests
pytest Tests/test_integration.py -m integration -v
```

### All Tests
```bash
# Run all tests (unit + integration)
pytest Tests/ -v

# Skip integration tests
pytest Tests/ -m "not integration" -v
```

## Test Coverage

The tests cover:

### MilvusClient Class (from core.utils)
- ✅ Client initialization (`get_client`)
- ✅ Database operations (`create_database`)
- ✅ Collection operations (`create_collection`, `drop_collection`, `has_collection`)
- ✅ Data insertion (`insert_data`)
- ✅ Document vectorization (`vectorize_documents`)
- ✅ Text embedding (`embed_text`, `embed_text_hf`, `embed_text_ollama`)
- ✅ Device detection (`get_device`)

### Utility Scripts
- ✅ create_collection.py
- ✅ create_db.py  
- ✅ drop_collection.py
- ✅ drop_db.py

## Test Markers

- `integration` - Tests requiring running Milvus instance
- `slow` - Tests that take longer to execute

## Import Structure

Tests use the refactored import:
```python
from core.utils import MilvusClient
```

## Dependencies

Tests require:
- pytest
- pytest-cov
- numpy (for array mocking)
- unittest.mock (built-in)

Install with:
```bash
pip install pytest pytest-cov numpy
```