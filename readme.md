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

### Utils folder
Contains MilvusUitls custom satic class which is used by other scripts

Script can be parameterized by adding a colelction_name or db_name on creation or deletion
 
```$ python ./utils/create-collection.py  my_collection```

## Test
- Run pytest with coverage

    ``` $ coverage run -m pytest tests/test_MilvusUtils.py ```

    ``` $ coverage report -m ```

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
python -c "from core import MilvusUtils; print('Installation successful!')"
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
│   └── MilvusUtils.py      # MilvusUtils class
├── utils/                   # Database & collection scripts
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
python utils/create_collection.py my_collection

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

Run tests:
```bash
# Unit tests
pytest Tests/test_utils.py -v

# With coverage
pytest Tests/test_utils.py --cov=core --cov-report=term-missing

# All tests
pytest Tests/ -v
```

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
- ✅ **Comprehensive tests** with pytest
- ✅ **Cross-platform** Windows/Linux/macOS support