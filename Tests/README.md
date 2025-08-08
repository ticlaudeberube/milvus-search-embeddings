# Tests Documentation

This directory contains comprehensive tests for the `core` package and related functionality.

## Test Files Status

### ✅ **Useful and Working Tests**

#### `test_milvus_utils.py` - **Core Unit Tests** (15 tests)
- **Status**: ✅ All 15 tests passing
- **Coverage**: Complete MilvusUtils functionality
- **Tests**: Client, database ops, collections, embeddings, device detection
- **Run**: `pytest tests/test_milvus_utils.py -v`

#### `test_db_scripts.py` - **Database Script Tests** (8 tests)
- **Status**: ✅ All 8 tests passing
- **Coverage**: Database and collection script validation
- **Tests**: create_db, drop_db, create_collection, drop_collection scripts
- **Run**: `pytest tests/test_db_scripts.py -v`

#### `test_integration.py` - **Integration Tests** (3 tests)
- **Status**: ✅ 2 passing, 1 skipped (expected)
- **Coverage**: End-to-end workflows with Milvus
- **Requires**: Running Milvus instance
- **Run**: `pytest tests/test_integration.py -v`

#### `conftest.py` - **Test Configuration**
- **Status**: ✅ Working
- **Purpose**: Pytest fixtures and configuration
- **Contains**: Mock clients, sample data, module reset

### ⚠️ **Diagnostic/Manual Tests**

#### `test_env_vars.py` - **Environment Variable Test**
- **Status**: ⚠️ Manual test script (not pytest)
- **Purpose**: Validate environment configuration
- **Run**: `python tests/test_env_vars.py`
- **Expected**: Shows env vars and embedding test results

#### `test_missing_env.py` - **Missing Environment Test**
- **Status**: ⚠️ Manual test script
- **Purpose**: Test error handling for missing env vars
- **Run**: `python tests/test_missing_env.py`
- **Expected**: Shows proper error messages

#### `test_all_loaders.py` - **Document Loader Integration**
- **Status**: ⚠️ Complex integration test
- **Purpose**: Test all document loading workflows
- **Requires**: Milvus + external dependencies
- **Run**: `python tests/test_all_loaders.py`

### 📄 **Support Files**

#### `rag_test_data.py` - **Test Data**
- **Status**: ✅ Support file
- **Purpose**: Sample data for RAG testing

#### `README.md` - **This Documentation**
- **Status**: ✅ Documentation

## Quick Test Commands

### Run Core Tests (Recommended)
```bash
# Essential core functionality tests
pytest tests/test_milvus_utils.py tests/test_db_scripts.py -v

# With coverage report
pytest tests/test_milvus_utils.py --cov=core --cov-report=term-missing
```

### Run Integration Tests
```bash
# Requires running Milvus instance
pytest tests/test_integration.py -v
```

### Run All Automated Tests
```bash
# All pytest-compatible tests
pytest tests/ -v
```

### Manual Diagnostic Tests
```bash
# Environment validation
python tests/test_env_vars.py
python tests/test_missing_env.py

# Document loader integration (slow)
python tests/test_all_loaders.py
```

## Test Coverage Summary

- **✅ 23 automated tests** (15 core + 8 scripts)
- **✅ 3 integration tests** (requires Milvus)
- **✅ 3 manual diagnostic tests**
- **✅ Complete core functionality coverage**
- **✅ Proper mocking** (no external API calls in unit tests)
- **✅ Error handling validation**

## Prerequisites

```bash
# Install package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov numpy
```

## Test Categories

- **Unit Tests**: Core functionality with mocking
- **Integration Tests**: End-to-end with real Milvus
- **Script Tests**: Utility script validation
- **Diagnostic Tests**: Environment and configuration validation

## Recommendations

1. **Always run**: `test_milvus_utils.py` and `test_db_scripts.py`
2. **Before deployment**: Run integration tests with Milvus
3. **Environment issues**: Use diagnostic tests
4. **CI/CD**: Focus on automated pytest tests