### Embeddings 

Tested Ollama and HuggingFace embeddings on Mac Intel AMD GPU
- AMD GPU and MPS not supported by Ollama
- Hugging Face provides improved performance for intel cpu
    - user can set batch_size 
    - MPS is supported
    - See benchmark details

### Collection Management

Both loaders now detect existing collections and prompt for user confirmation:
- **(d)rop**: Drop existing collection and recreate
- **(a)bort**: Cancel the process

```bash
python ./document-loaders/load_milvus_docs_ollama.py
```

### Sync Functionality

Sync embeddings from JSON file with checksum-based deduplication and deletion:

```bash
python ./document-loaders/Sync_from_json.py
```

**Features:**
- Loads embeddings from `./data/embeddings.json`
- Creates collection if it doesn't exist
- Compares checksums to identify changed content
- Removes records from collection that are missing from source data
- Only updates records with different checksums
- Logs count of new and updated documents

**Output:**
```
Upserted 5 new, 3 updated documents
```
