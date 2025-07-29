#!/usr/bin/env python3
"""
Cross-platform environment loader for Milvus Search Embeddings
Loads environment variables from .env file or system environment
"""

import os
from pathlib import Path
from typing import Dict, Optional

def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load environment variables from .env file"""
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

def setup_environment() -> None:
    """Setup all required environment variables"""
    # Load from .env file
    env_vars = load_env_file()
    
    # Set environment variables (don't override existing ones)
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Add utils to Python path if not already there
    utils_path = str(Path(__file__).parent.parent / "utils")
    python_path = os.environ.get("PYTHONPATH", "")
    if utils_path not in python_path:
        os.environ["PYTHONPATH"] = f"{python_path};{utils_path}" if python_path else utils_path

def get_config() -> Dict[str, str]:
    """Get current configuration"""
    setup_environment()
    
    config_keys = [
        "MY_DB_NAME", "MILVUS_OLLAMA_COLLECTION_NAME", "EMBEDDING_PROVIDER",
        "HF_EMBEDDING_MODEL", "HF_LLM_MODEL", "OLLAMA_LLM_MODEL", "OLLAMA_EMBEDDING_MODEL",
        "OLLAMA_NUM_THREADS", "HUGGINGFACEHUB_API_TOKEN"
    ]
    
    return {key: os.environ.get(key, "") for key in config_keys}

if __name__ == "__main__":
    setup_environment()
    config = get_config()
    
    print("Current Environment Configuration:")
    print("-" * 40)
    for key, value in config.items():
        if value:
            print(f"{key}: {value}")
        else:
            print(f"{key}: NOT SET")