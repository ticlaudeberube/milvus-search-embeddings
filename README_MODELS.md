# Model Configuration Guide

## Environment Variables

Copy `.env.example` to `.env` and configure your models:

```bash
cp .env.example .env
```

### Key Variables:

- `OLLAMA_LLM_MODEL`: LLM model for chat/generation (default: `llama3.2`)
- `OLLAMA_EMBEDDING_MODEL`: Embedding model (default: `nomic-embed-text:v1.5`)
- `HF_EMBEDDING_MODEL`: HuggingFace model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_PROVIDER`: Provider choice (`ollama` or `huggingface`)

## Recommended Models

### Ollama LLM Models:
- **Fast**: `orca-mini` (1.9GB)
- **Balanced**: `llama3.2` (2GB)
- **Accurate**: `mistral` (4.1GB)

### Ollama Embedding Models:
- **Fast**: `nomic-embed-text:v1.5` (274MB)
- **Accurate**: `mxbai-embed-large` (669MB)

### HuggingFace Models:
- **Fast**: `all-MiniLM-L6-v2` (91MB)
- **Accurate**: `all-mpnet-base-v2` (438MB)

## Usage

All scripts now automatically use environment variables. No code changes needed.