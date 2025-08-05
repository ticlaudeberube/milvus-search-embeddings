# PowerShell version - Updated environment variables for Milvus Search Embeddings
$env:MY_DB_NAME = "my_database"
$env:MILVUS_OLLAMA_COLLECTION_NAME = "milvus_ollama_collection"
$env:MILVUS_HF_COLLECTION_NAME = "milvus_hf_collection"
$env:EMBEDDING_PROVIDER = "ollama"
$env:OLLAMA_LLM_MODEL = "llama3.2:1b"
$env:OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
$env:HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
$env:HF_LLM_MODEL = "sentence-transformers/all-mpnet-base-v2"
$env:OLLAMA_NUM_THREADS = "8"
# Set environment variables for better performance
$env:OLLAMA_NUM_PARALLEL=4
$env:OLLAMA_MAX_LOADED_MODELS=2
$env:OLLAMA_FLASH_ATTENTION=1
# Memory optimization
$env:OLLAMA_KEEP_ALIVE="5m"  # Unload models after 5 minutes
$env:OLLAMA_MAX_QUEUE=512     # Limit request queue
$env:OLLAMA_RUNNERS=2         # Limit concurrent model runners

$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_MPS_HIGH_WATERMARK_RATIO = "0.0"
$env:USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
$env:MILVUS_HOST = "localhost"
$env:MILVUS_PORT = "19530"
$env:HUGGINGFACEHUB_API_TOKEN = "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"

Write-Host "Environment variables set successfully!" -ForegroundColor Green
Write-Host "OLLAMA_LLM_MODEL: $env:OLLAMA_LLM_MODEL"
Write-Host "OLLAMA_EMBEDDING_MODEL: $env:OLLAMA_EMBEDDING_MODEL"
Write-Host "HF_EMBEDDING_MODEL $env:HF_EMBEDDING_MODEL"
Write-Host "HF_LLM_MODEL: $env:HF_LLM_MODEL"