# We load all markdown files from the folder milvus_docs/en/faq. For each document, we just simply use "# "
# to separate the content in the file, which can roughly separate the content of each main part of the markdown file.
import os
from glob import glob
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from core import get_client, has_collection, EmbeddingProvider


collection_name = os.getenv("OLLAMA_COLLECTION_NAME") or "milvus_ollama_collection"

client = get_client()

def check_collection_and_confirm():
    """Check if collection exists and get user confirmation"""
    if has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        choice = input("Do you want to (d)rop and recreate, or (a)bort? [d/a]: ").lower().strip()
        
        if choice == 'a':
            print("Process aborted by user.")
            return False
        elif choice == 'd':
            print(f"Dropping collection '{collection_name}'...")
            return True
        else:
            print("Invalid choice. Aborting.")
            return False
    return True

def create_collection(embedding_dim=1024):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"Collection '{collection_name}' dropped successfully.")

    client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="COSINE",  # Inner product distance
        consistency_level="Session",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
    )
    print(f"Collection '{collection_name}' created successfully.")


def process():
    # Check collection and get user confirmation
    if not check_collection_and_confirm():
        return
    
    text_lines = []

    for file_path in tqdm(glob("./document-loaders/milvus_docs/en/**/*.md", recursive=True), desc="Reading files"):
        with open(file_path, "r", encoding="utf-8") as file:
            file_text = file.read()
        text_lines += file_text.split("# ")

    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        if not line.strip() or len(line.strip()) < 10:
            continue
            
        try:
            vector = EmbeddingProvider.embed_text(line, provider='ollama')
            if vector:
                data.append({"id": len(data), "vector": vector, "text": line})
        except Exception as e:
            print(f"Failed to embed text chunk {i}: {e}")
            continue
    
    if len(data) == 0:
        return
        
    dimension = len(data[0]['vector'])
    create_collection(dimension)
    client.insert(collection_name=collection_name, data=data)

if __name__ == "__main__":
    process()