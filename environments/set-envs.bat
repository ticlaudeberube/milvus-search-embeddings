@echo off
REM Updated environment variables for Milvus Search Embeddings
set MY_DB_NAME=my_database
set MILVUS_OLLAMA_COLLECTION_NAME=milvus_ollama_collection
set MILVUS_HF_COLLECTION_NAME=milvus_hf_collection
set EMBEDDING_PROVIDER=ollama
set OLLAMA_LLM_MODEL=llama3.2:1b
set OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5
set HF_EMBEDDING_MODEL=all-MiniLM-L6-v2
set HF_LLM_MODEL=sentence-transformers/all-mpnet-base-v2
set OLLAMA_NUM_THREADS=8
set TOKENIZERS_PARALLELISM=false
set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
set USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36
set MILVUS_HOST=localhost
set MILVUS_PORT=19530
REM set HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

echo Environment variables set successfully!
echo OLLAMA_LLM_MODEL=%OLLAMA_LLM_MODEL%
echo OLLAMA_EMBEDDING_MODEL=%OLLAMA_EMBEDDING_MODEL%
echo HF_EMBEDDING_MODEL=%HF_EMBEDDING_MODEL%
echo HF_LLM_MODEL=%HF_LLM_MODEL%