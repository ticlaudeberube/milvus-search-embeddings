# Hello World Milvus Search Demo

## Quick Start

1. **Set environment variables**:
   ```bash
   # Linux/macOS
   source ./hello-world-milvus-search/search-envs.sh
   
   # Windows Command Prompt
   .\hello-world-milvus-search\search-envs.bat
   
   # Windows PowerShell
   .\hello-world-milvus-search\search-envs.ps1
   ```

2. **Create database**:
   ```bash
   python utils/create_db.py my_db
   ```

3. **Create collection**:
   ```bash
   python hello-world-milvus-search/create-index.py
   ```

4. **Vectorize documents**:
   ```bash
   python hello-world-milvus-search/vectorize-search.py
   ```

5. **Search**:
   ```bash
   python hello-world-milvus-search/range-search.py
   ```

## Scripts

- `create-index.py` - Creates collection with proper schema and index
- `vectorize-search.py` - Inserts sample documents with embeddings
- `range-search.py` - Performs range-based similarity search