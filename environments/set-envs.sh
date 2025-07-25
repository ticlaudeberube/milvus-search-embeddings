
export MY_DB_NAME="my_database"
export MILVUS_OLLAMA_COLLECTION_NAME="milvus_ollama_collection"
export MILVUS_HF_COLLECTION_NAME="milvus_hf_collection"
export MODEL_HF="all-MiniLM-L6-v2"
export MODEL_OLLAMA="nomic-embed-text"
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/utils"
export EMBEDDING_MODEL_OLLAMA="llama3.2"
export EMBEDDING_MODEL_HF="sentence-transformers/all-mpnet-base-v2"