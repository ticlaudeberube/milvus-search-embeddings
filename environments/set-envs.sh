export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/utils"

export MY_DB_NAME="default"

export HF_COLLECTION_NAME="milvus_hf_collection"
export HF_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export HF_LLM_MODEL="sentence-transformers/all-mpnet-base-v2"

export OLLAMA_COLLECTION_NAME="milvus_ollama_collection"
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
export OLLAMA_LLM_MODEL="llama3.2"
