# Milvus Search Embeddings

This is a Python POC for Milvus embedings, search and Milvus Docs RAG

## Package structure.

- [Install Milvus Docker Container](https://milvus.io/docs/install_standalone-docker.md)
- Launch Milvus Docker container 
- Access web UI: http://127.0.0.1:9091/webui/

### Document folder
Contains scripts for downloading and embed documents

### Search folder
Contains basic embedding Alan Touring search 

### Core folder
Contains MilvusUtils custom static class which is used by other scripts

Script can be parameterized by adding a collection_name or db_name on creation or deletion
 
```$ python ./core/create_collection.py  my_collection```

## Test Coverage

### Quick Test Commands
```bash
# Run all core tests
pytest Tests/test_milvus_utils.py -v

# Run with coverage report
pytest Tests/test_milvus_utils.py --cov=core --cov-report=term-missing

# Test specific functionality
pytest Tests/test_milvus_utils.py::test_embed_text_huggingface -v
```

### Test Results
- **17 comprehensive tests** covering all MilvusUtils methods
- **Database operations**: create, drop, exception handling
- **Embedding providers**: HuggingFace, Ollama with proper mocking
- **Collection management**: create, drop, existence validation
- **Data operations**: insertion, vectorization, search preparation

## [Milvuv_cli](https://milvus.io/docs/cli_commands.md)
- milvus_cli 
- connect -uri http://localhost:19530
- list databases
- list collections
- create database --db_name test
- use database --db_name test
- use database --db_name default
- delete database --db_name test

## Environment Configuration

### Quick Setup (.env file - Recommended)
```cmd
# Copy template and customize
copy .env.example .env
# Edit .env with your values
```

### Alternative: Use environment scripts
```cmd
# Windows batch files
environments\setup-all.bat

# PowerShell
.\environments\setup-all.ps1

# Linux/macOS
source environments/set-envs.sh
```

### Python Environment Loader
```python
# Use in your scripts
from environments.load_env import setup_environment, get_config
setup_environment()
config = get_config()
```

## Installation

### Windows Setup with Virtual Environment

1. **Check Python version** (requires Python 3.12+):
```cmd
python --version
```

2. **Create virtual environment**:
```cmd
python -m venv venv
```

3. **Activate virtual environment**:
```cmd
venv\Scripts\activate
```

4. **Install in development mode**:
```cmd
pip install -e .
```

5. **Verify installation**:
```cmd
python -c "from core import get_client; print('Installation successful!')"
```

### Deactivate Environment
```cmd
deactivate
```

**Note:** Always activate the virtual environment before running scripts to ensure the correct Python version and dependencies.

## Project Structure

```
milvus-search-embeddings/
├── core/                    # Core package (installable)
│   └── milvus_utils.py      # MilvusUtils class
├── advanced-search/         # RAG and search scripts
├── document-loaders/        # Document processing scripts
├── benchmark/               # Performance testing scripts
└── Tests/                   # Unit and integration tests
```

## Usage

### Import the Core Package
```python
from core import MilvusUtils 

# Get client
client = MilvusUtils.get_client()

# Create collection
MilvusUtils.create_collection("my_collection")

# Embed text
embeddings = MilvusUtils.embed_text("Hello world", provider="ollama")
```

### Run Utility Scripts
```bash
# Create collection
python core/create_collection.py my_collection

# Create database
python core/create_db.py my_database

# Load documents
python document-loaders/load_milvus_docs_ollama.py

# Search with RAG
python advanced-search/search-ollama-chat.py
```

## Milvus Database

Start Milvus container:
```bash
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

- Web UI: http://127.0.0.1:9091/webui/
- [Full Installation Guide](https://milvus.io/docs/install_standalone-docker.md)

## Testing

### Comprehensive Test Suite
The project includes extensive test coverage with **17 comprehensive tests** for MilvusUtils:

**Core Functionality Tests:**
- ✅ Database operations (create, drop, exception handling)
- ✅ Collection management (create, drop, existence checks)
- ✅ Data insertion and vectorization
- ✅ Embedding providers (HuggingFace, Ollama)
- ✅ Device detection and utility functions
- ✅ Deprecated method compatibility

**Document Loader Tests:**
- ✅ Ollama embedding functionality
- ✅ HuggingFace embedding functionality
- ✅ Integration tests with Milvus
- ✅ Environment configuration validation

### Run Tests
```bash
# Core MilvusUtils tests (17 tests)
pytest Tests/test_milvus_utils.py -v

# Document loader and integration tests
pytest Tests/test_utils.py::TestDocumentLoaders -v

# Database script tests
pytest Tests/test_db_scripts.py -v

# All tests with coverage
pytest Tests/ --cov=core --cov-report=term-missing -v

# Quick core functionality test
pytest Tests/test_milvus_utils.py::test_get_client Tests/test_milvus_utils.py::test_create_database_new -v
```

### Test Categories
- **Unit Tests**: Core MilvusUtils functionality
- **Integration Tests**: End-to-end workflows with Milvus
- **Script Tests**: Utility script validation
- **Environment Tests**: Configuration and setup validation

## Milvus CLI

```bash
milvus_cli
connect -uri http://localhost:19530
list databases
list collections
use database --db_name test
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MY_DATABASE` | Database name | default | No |
| `MILVUS_OLLAMA_COLLECTION_NAME` | Ollama collection | demo_collection | No |
| `EMBEDDING_PROVIDER` | Provider (ollama/huggingface) | ollama | No |
| `OLLAMA_LLM_MODEL` | Ollama LLM model | llama3.2 | No |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model | nomic-embed-text:v1.5 | No |
| `HF_EMBEDDING_MODEL` | HuggingFace model | sentence-transformers/all-MiniLM-L6-v2 | No |
| `HF_TOKEN` | HuggingFace API token | - | Yes* |
| `OLLAMA_NUM_THREADS` | Ollama threads | 4 | No |

*Required for HuggingFace API access

## Features

- ✅ **Clean package structure** with `core.MilvusUtils`
- ✅ **Global imports** - no path manipulation needed
- ✅ **Multiple embedding providers** (HuggingFace, Ollama)
- ✅ **Environment management** with `.env` files and cross-platform scripts
- ✅ **RAG implementations** with Streamlit and Gradio
- ✅ **Comprehensive test suite** - 17 tests with full coverage
- ✅ **Proper mocking** - No external API calls in tests
- ✅ **Database management** - Create, drop, and manage databases
- ✅ **Cross-platform** Windows/Linux/macOS support
- ✅ **Type hints** and PEP 8 compliance
- ✅ **Error handling** with proper exception management