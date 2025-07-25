"""Load Milvus documentation and create embeddings using Ollama."""
import os
from glob import glob
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from core.utils import MilvusClient

COLLECTION_NAME = os.getenv("MILVUS_OLLAMA_COLLECTION_NAME", "demo_collection")
DOCS_PATH = "./document-loaders/milvus_docs/en/**/*.md"
MIN_TEXT_LENGTH = 10


def create_collection_if_needed(client, collection_name: str, embedding_dim: int) -> None:
    """Create collection if it doesn't exist."""
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            metric_type="IP",
            consistency_level="Strong"
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")


def load_and_split_documents() -> List[str]:
    """Load markdown files and split by headers."""
    text_chunks = []
    
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Searching for files at: {DOCS_PATH}")
        file_paths = glob(DOCS_PATH, recursive=True)
        print(f"Found {len(file_paths)} files")
        if not file_paths:
            raise FileNotFoundError(f"No markdown files found at {DOCS_PATH}")
            
        for file_path in tqdm(file_paths, desc="Reading files"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    chunks = [chunk.strip() for chunk in content.split("# ") 
                             if chunk.strip() and len(chunk.strip()) >= MIN_TEXT_LENGTH]
                    text_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []
        
    return text_chunks


def create_embeddings_data(text_chunks: List[str]) -> List[Dict[str, Any]]:
    """Create embedding data from text chunks."""
    data = []
    
    for i, text in enumerate(tqdm(text_chunks, desc="Creating embeddings")):
        if not text or not isinstance(text, str) or not text.strip():
            continue
        try:
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
            vector = MilvusClient.embed_text_ollama(text, model=embedding_model)
            if vector:
                data.append({"id": i, "vector": vector, "text": text})
        except Exception as e:
            print(f"Error creating embedding for chunk {i}: {e}")
            continue
            
    return data


def main() -> None:
    """Main processing function."""
    try:
        client = MilvusClient.get_client()
        
        # Load and process documents
        text_chunks = load_and_split_documents()
        if not text_chunks:
            print("No valid text chunks found.")
            return
            
        print(f"Found {len(text_chunks)} text chunks")
        
        # Create embeddings
        data = create_embeddings_data(text_chunks)
        if not data:
            print("No embeddings created.")
            return
            
        # Create collection and insert data
        dimension = len(data[0]['vector'])
        create_collection_if_needed(client, COLLECTION_NAME, dimension)
        
        client.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"Successfully inserted {len(data)} documents into '{COLLECTION_NAME}'")
        
    except Exception as e:
        print(f"Error in main process: {e}")


if __name__ == "__main__":
    main()