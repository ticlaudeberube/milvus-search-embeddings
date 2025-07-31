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
python load_milvus_docs_ollama.py
python load_milvus_docs_hf.py
```

### Env variables are to be set from search-rag folder
```
. ./environments/set-envs.sh
```
