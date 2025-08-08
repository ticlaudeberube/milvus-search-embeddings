# Milvus Search Embeddings

This is a Python POC for Milvus embedings, search and Milvus Docs RAG

## Package structure.

- [Install Milvus Docker Container](https://milvus.io/docs/install_standalone-docker.md)
- Launch Milvus Docker container 
- Access web UI: http://127.0.0.1:9091/webui/

### Core Package
Installable Python package with MilvusUtils class and utility scripts

### Search Modules
- **search-advanced/**: RAG implementations with Streamlit and Gradio UIs
- **search-agentic/**: Agentic RAG with classification and response agents
- **search-filtered/**: Filtered RAG with document classification
- **search-hello-world-milvus/**: Basic vector search examples

### Document Loaders
Scripts for downloading and embedding documents from various sources

Core utilities can be parameterized with collection_name or db_name:
 
```bash
# Database operations
python core/utils/create_db.py my_database
python core/utils/drop_db.py my_database

# Collection operations  
python core/utils/create_collection.py my_collection
python core/utils/drop_collection.py my_collection
```

## Test Coverage

### Quick Test Commands
```bash
# Run all core tests
pytest tests/test_milvus_utils.py -v

# Run with coverage report
pytest tests/test_milvus_utils.py --cov=core --cov-report=term-missing

# Test specific functionality
pytest tests/test_milvus_utils.py::test_embed_text_huggingface -v
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
│   ├── utils/               # Utility scripts (create_db, create_collection)
├── search-advanced/         # RAG implementations with Streamlit/Gradio
├── search-filtered/         # Filtered RAG with classification
├── search-hello-world-milvus/ # Basic vector search examples
├── document-loaders/        # Document processing and embedding
├── benchmark/               # Performance testing and optimization
├── environments/            # Environment setup scripts
└── tests/                   # Comprehensive test suite
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
# Database and collection management
python core/utils/create_db.py my_database
python core/utils/create_collection.py my_collection

# Load documents
python document-loaders/download_milvus_docs.py
python document-loaders/load_milvus_docs_ollama.py

# Search implementations
python search-advanced/search_ollama_chat.py
python search-agentic/agentic_rag_app.py
python search-filtered/streamlit_filtered_rag.py

# Web interfaces
streamlit run search-advanced/search_ollama_streamlit_rag.py
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
pytest tests/test_milvus_utils.py -v

# Document loader and integration tests
pytest tests/test_utils.py::TestDocumentLoaders -v

# Database script tests
pytest tests/test_db_scripts.py -v

# All tests with coverage
pytest tests/ --cov=core --cov-report=term-missing -v

# Quick core functionality test
pytest tests/test_milvus_utils.py::test_get_client tests/test_milvus_utils.py::test_create_database_new -v
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
| `OLLAMA_LLM_MODEL` | Ollama LLM model | llama3.2:1b | No |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model | nomic-embed-text:v1.5 | No |
| `HF_EMBEDDING_MODEL` | HuggingFace model | sentence-transformers/all-MiniLM-L6-v2 | No |
| `HF_TOKEN` | HuggingFace API token | - | Yes* |
| `OLLAMA_NUM_THREADS` | Ollama threads | **Auto-set to 4** | No |

*Required for HuggingFace API access

### Ollama Threading

`OLLAMA_NUM_THREADS` is **automatically set** when using Ollama embeddings:

```python
from core import EmbeddingProvider
# Auto-sets OLLAMA_NUM_THREADS=4 if not already set
embeddings = EmbeddingProvider.embed_text("text", provider='ollama')
```

Manual control:
```python
from core import ensure_threads
ensure_threads(8)  # Set custom thread count
```

Find optimal threads:
```bash
python benchmark/ollama-threads-check.py
```

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