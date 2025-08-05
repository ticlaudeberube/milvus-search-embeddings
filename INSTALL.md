# Installation Instructions

## Development Setup

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 2. Install Package
```bash
# Install in development mode (recommended)
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev]"

# Or install from requirements (all dependencies)
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Copy environment template
copy .env.example .env

# Edit .env with your configuration
# Key variables:
# - MILVUS_URI=http://localhost:19530
# - MILVUS_TOKEN=root:Milvus
# - MY_DATABASE=default
# - HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# - OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5
# - EMBEDDING_PROVIDER=ollama
```

## Usage

### New Modular Interface (Recommended)
```python
# Import specific functions
from core import get_client, create_collection, EmbeddingProvider
from core import create_database, drop_database, list_databases
from core.exceptions import DatabaseError, CollectionError, EmbeddingError

# Use modular functions
client = get_client()
create_collection("my_collection", dimension=768)
embeddings = EmbeddingProvider.embed_text("Hello world", provider="ollama")
create_database("my_database")
```

### Legacy Interface (Backward Compatible)
```python
# Import legacy class
from core import MilvusUtils

# Use legacy methods
client = MilvusUtils.get_client()
MilvusUtils.create_collection("my_collection")
embeddings = MilvusUtils.embed_text("Hello world", provider="ollama")
```

### Utility Scripts
```bash
# Create collection
python core/utils/create_collection.py my_collection

# Create database
python core/utils/create_db.py my_database

# Drop collection
python core/utils/drop_collection.py my_collection

# Drop database
python core/utils/drop_db.py my_database

# Create index
python core/utils/create_index.py my_collection

# Load documents
python document-loaders/load_milvus_docs_ollama.py

# Run RAG search
streamlit run advanced-search/search_ollama_streamlit_rag.py
```

## Package Structure

```
core/
├── __init__.py              # Unified interface + backward compatibility
├── client.py               # Connection management
├── embeddings.py           # Embedding providers (HuggingFace, Ollama)
├── collections.py          # Collection operations
├── databases.py            # Database operations
├── config.py              # Configuration management
├── exceptions.py          # Custom exceptions
├── milvus_utils_legacy.py  # Legacy MilvusUtils implementation
├── readme.md              # Core package documentation
└── utils/                 # Utility scripts
    ├── create_collection.py
    ├── create_db.py
    ├── create_index.py
    ├── drop_collection.py
    └── drop_db.py
```

## Testing

### Run All Tests
```bash
pytest Tests/ -v
```

### Run Specific Tests
```bash
# Core functionality
pytest Tests/test_milvus_utils.py -v

# With coverage
pytest Tests/ --cov=core --cov-report=term-missing
```

### Test Categories
- **Unit Tests**: Core functionality with mocking
- **Integration Tests**: End-to-end workflows
- **Script Tests**: Utility script validation

## Error Handling

The refactored package includes proper exception handling:

```python
from core import create_database
from core.exceptions import DatabaseError

try:
    create_database("my_db")
except DatabaseError as e:
    print(f"Database operation failed: {e}")
```

## Migration Guide

### From Legacy to New Interface

**Old way:**
```python
from core import MilvusUtils
client = MilvusUtils.get_client()
embeddings = MilvusUtils.embed_text_ollama("text")
MilvusUtils.create_database("my_db")
```

**New way:**
```python
from core import get_client, EmbeddingProvider, create_database
client = get_client()
embeddings = EmbeddingProvider.embed_text("text", provider="ollama")
create_database("my_db")
```

### Benefits of New Interface
- **Modular imports** - Import only what you need
- **Better error handling** - Specific exception types
- **Cleaner code** - Focused, single-responsibility modules
- **Easier testing** - Mock individual components
- **Type safety** - Better IDE support and type hints

## Troubleshooting

### Connection Issues
```python
from core.exceptions import MilvusConnectionError

try:
    client = get_client()
except MilvusConnectionError as e:
    print(f"Failed to connect: {e}")
    # Check MILVUS_URI and MILVUS_TOKEN environment variables
```

### Environment Variables
Ensure these are set in your `.env` file:
- `MILVUS_URI` - Milvus server URI (default: http://localhost:19530)
- `MILVUS_TOKEN` - Authentication token (default: root:Milvus)
- `MY_DATABASE` - Default database name (default: default)
- `EMBEDDING_PROVIDER` - Provider to use (ollama/huggingface)
- `HF_EMBEDDING_MODEL` - HuggingFace model name
- `OLLAMA_EMBEDDING_MODEL` - Ollama model name
- `HF_TOKEN` - HuggingFace API token (required for HF API access)