# Hello World Milvus Search Demo

## Quick Start

1. **Set environment variables**:
   ```bash
   # Linux/macOS
   source ./search-hello-world-milvus/search-envs.sh
   
   # Windows Command Prompt
   .\search-hello-world-milvus\search-envs.bat
   
   # Windows PowerShell
   .\search-hello-world-milvus\search-envs.ps1
   ```

2. **Create database**:
   ```bash
   python utils/create_db.py my_db
   ```

3. **Create collection**:
   ```bash
   python search-hello-world-milvus/create-index.py
   ```

4. **Vectorize documents**:
   ```bash
   python search-hello-world-milvus/vectorize-search.py
   ```

5. **Search**:
   ```bash
   python search-hello-world-milvus/range-search.py
   ```

## Scripts

- `create-index.py` - Creates collection with proper schema and index
- `vectorize-search.py` - Inserts sample documents with embeddings
- `range-search.py` - Performs range-based similarity search