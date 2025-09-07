"""Shared pipeline functions for both Airflow and standalone execution"""

import urllib.request
import zipfile
import tempfile
import os
from glob import glob
from tqdm import tqdm
import ollama
from pymilvus import MilvusClient

def download_and_extract():
    """Download and extract Milvus docs to temp directory"""
    docs_url = "https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip"
    
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "milvus_docs.zip")
    extract_path = os.path.join(temp_dir, "extracted")
    
    print(f"\U0001F4E6 Downloading Milvus docs...")
    urllib.request.urlretrieve(docs_url, zip_path)
    
    print(f"\U0001F4E6 Extracting to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"\u2705 Extraction complete!")
    return extract_path

def load_files(extract_path):
    """Load markdown files and split into chunks"""
    text_lines = []
    file_paths = glob(f"{extract_path}/**/en/**/*.md", recursive=True)
    
    print(f"Found {len(file_paths)} markdown files")
    
    for file_path in tqdm(file_paths, desc="Reading files"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_text = file.read()
            text_lines += file_text.split("# ")
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue
    
    print(f"\U0001F4DA Created {len(text_lines)} text chunks")
    return text_lines

def process_input(text_lines):
    """Process and filter text chunks"""
    print(f"\n[INFO] Filtering {len(text_lines)} text chunks...")
    
    processed_chunks = []
    for line in text_lines:
        stripped = line.strip()
        if stripped and len(stripped) >= 10:
            processed_chunks.append(stripped)
    
    filtered_out = len(text_lines) - len(processed_chunks)
    print(f"[OK] Kept {len(processed_chunks)} valid chunks, filtered out {filtered_out} short/empty chunks")
    return processed_chunks

def produce_embeddings(chunks):
    """Generate embeddings for text chunks"""
    data = []
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
    
    for i, chunk in enumerate(tqdm(chunks, desc="Creating embeddings")):
        try:
            response = ollama.embeddings(model=model, prompt=chunk)
            vector = response['embedding']
            if vector:
                data.append({"id": len(data), "vector": vector, "text": chunk})
        except Exception as e:
            print(f"Failed to embed chunk {i}: {e}")
            continue
    
    return data

def load_to_milvus(data):
    """Load embeddings to local Milvus instance"""
    if not data:
        print("No data to load")
        return
    
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME", "milvus_ollama_collection")
    client = MilvusClient(uri="http://localhost:19530")
    
    # Drop existing collection if exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")
    
    # Create new collection
    dimension = len(data[0]['vector'])
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        metric_type="COSINE",
        consistency_level="Session",
    )
    print(f"Created collection: {collection_name}")
    
    # Insert data
    client.insert(collection_name=collection_name, data=data)
    print(f"Inserted {len(data)} documents")