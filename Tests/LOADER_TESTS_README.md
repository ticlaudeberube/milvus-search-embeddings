# Document Loader Tests

This document describes the comprehensive test suite for all document loaders in the project.

## Test Coverage

The test suite in `Tests/test_utils.py` now includes comprehensive testing for all document loaders:

### Core Functionality Tests
- **MilvusClient Tests**: Basic client operations, database management, collection operations
- **Embedding Tests**: Both Ollama and HuggingFace embedding functionality
- **Utility Script Tests**: Create/drop collections and databases

### Document Loader Tests (`TestDocumentLoaders` class)

#### 1. **Download Tests**
- `test_milvus_docs_download`: Tests Milvus documentation download functionality

#### 2. **Embedding Functionality Tests**
- `test_ollama_embedding_functionality`: Tests Ollama embedding with real text
- `test_huggingface_embedding_functionality`: Tests HuggingFace embedding with real text

#### 3. **Loader Structure Tests**
- `test_milvus_docs_loader_structure`: Validates Milvus docs loader file structure
- `test_state_union_loader_structure`: Validates State of Union loader structure
- `test_various_docs_loader_structure`: Validates various docs loader structure

#### 4. **Integration Tests**
- `test_loader_integration_with_milvus`: End-to-end integration test with Milvus
- `test_environment_configuration`: Validates environment setup

#### 5. **Execution Tests** (marked as `@pytest.mark.slow`)
- `test_milvus_docs_ollama_loader_execution`: Tests actual execution of Milvus docs Ollama loader
- `test_milvus_docs_hf_loader_execution`: Tests actual execution of Milvus docs HF loader

## Running the Tests

### Run All Tests
```bash
python -m pytest Tests/test_utils.py -v
```

### Run Only Document Loader Tests
```bash
python -m pytest Tests/test_utils.py::TestDocumentLoaders -v
```

### Run Tests Excluding Slow Tests
```bash
python -m pytest Tests/test_utils.py -v -m "not slow"
```

### Run Tests Excluding Integration Tests
```bash
python -m pytest Tests/test_utils.py -v -m "not integration"
```

### Run with Coverage
```bash
python -m pytest Tests/test_utils.py --cov=core --cov-report=term-missing
```

## Test Requirements

### Environment Variables
The tests automatically detect and skip tests based on available configuration:

- `OLLAMA_EMBEDDING_MODEL`: Required for Ollama-based tests
- `HF_EMBEDDING_MODEL`: Required for HuggingFace-based tests
- `MILVUS_OLLAMA_COLLECTION_NAME`: Collection name for Ollama tests
- `MILVUS_HF_COLLECTION_NAME`: Collection name for HuggingFace tests

### Dependencies
- Milvus server running on localhost:19530
- Internet connection for document downloads
- Configured embedding models (Ollama or HuggingFace)

## Test Results Summary

When properly configured, all tests should pass:

```
============================= test session starts =============================
Tests/test_utils.py::TestDocumentLoaders::test_milvus_docs_download PASSED
Tests/test_utils.py::TestDocumentLoaders::test_ollama_embedding_functionality PASSED
Tests/test_utils.py::TestDocumentLoaders::test_huggingface_embedding_functionality PASSED
Tests/test_utils.py::TestDocumentLoaders::test_milvus_docs_loader_structure PASSED
Tests/test_utils.py::TestDocumentLoaders::test_state_union_loader_structure PASSED
Tests/test_utils.py::TestDocumentLoaders::test_various_docs_loader_structure PASSED
Tests/test_utils.py::TestDocumentLoaders::test_loader_integration_with_milvus PASSED
Tests/test_utils.py::TestDocumentLoaders::test_environment_configuration PASSED
Tests/test_utils.py::TestDocumentLoaders::test_milvus_docs_ollama_loader_execution PASSED
Tests/test_utils.py::TestDocumentLoaders::test_milvus_docs_hf_loader_execution PASSED
================= 10 passed ==================
```

## Tested Document Loaders

The following document loaders are tested:

1. **download_milvus_docs.py** - Downloads Milvus documentation
2. **load_milvus_docs_ollama.py** - Loads Milvus docs with Ollama embeddings
3. **load_milvus_docs_hf.py** - Loads Milvus docs with HuggingFace embeddings
4. **load-state-of-the-union-ollama.py** - Loads State of Union with Ollama
5. **load-state-of-the-union-default.py** - Loads State of Union with default embeddings
6. **load-various-docs-scatterplot.py** - Loads various docs with Ollama (with visualization)
7. **load-various-docs-scatterplot-hf.py** - Loads various docs with HuggingFace (with visualization)

## Notes

- Tests marked with `@pytest.mark.slow` may take several minutes to complete
- Tests marked with `@pytest.mark.integration` require a running Milvus instance
- Some tests are automatically skipped if required environment variables are not set
- All tests preserve existing behavior and validate current functionality