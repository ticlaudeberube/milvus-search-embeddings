# About
This is a Python POC for Milvus embeddings, search and Milvus Docs RAG

## Milvus Db

- [Install Milvus Docker Container](https://milvus.io/docs/install_standalone-docker.md)
- Launch Milvus Docker container 
- Access web UI: http://127.0.0.1:9091/webui/

## Project Structure

### document-loaders/
Contains scripts for downloading and embedding documents:
- `get_milvus_docs.py` - Downloads Milvus documentation
- `load_milvus_docs_hf.py` - Loads docs using HuggingFace embeddings
- `load_milvus_docs_ollama.py` - Loads docs using Ollama embeddings
- Various other document loading scripts

### hello-world-milvus-search/
Contains basic Milvus search examples:
- `vectorize-search.py` - Basic vector search implementation
- `create-index.py` - Index creation examples
- `range-search.py` - Range search examples

### advanced-search/
Contains advanced search implementations with different frameworks:
- Gradio-based chat interfaces
- Streamlit-based RAG applications
- Ollama and HuggingFace integrations

### benchmark/
Contains benchmarking scripts for performance testing

### core/
Contains core utilities for Milvus operations:
- `MilvusUtils.py` - Main utilities class with database and collection management
- `create_collection.py` - CLI script for creating collections
- `drop_collection.py` - CLI script for dropping collections
- `create_db.py` - CLI script for creating databases
- `drop_db.py` - CLI script for dropping databases

### Usage

**Programmatic usage:**
```python
from core.MilvusUtils import MilvusUtils

# Create database
MilvusUtils.create_database("my_database")

# Create collection
MilvusUtils.create_collection("my_collection", dimension=768)

# Drop collection
MilvusUtils.drop_collection("my_collection")
```

**CLI usage:**
```bash
# Collections
python core/create_collection.py my_collection
python core/drop_collection.py my_collection

# Databases  
python core/create_db.py my_database
python core/drop_db.py my_database
```

## Tests
- Run pytest with coverage:

    ```bash
    coverage run -m pytest Tests/test_MilvusUtils.py
    coverage report -m
    ```

## [Milvus CLI](https://milvus.io/docs/cli_commands.md)
- `milvus_cli`
- `connect -uri http://localhost:19530`
- `list databases`
- `list collections`
- `create database --db_name test`
- `use database --db_name test`
- `use database --db_name default`
- `delete database --db_name test`