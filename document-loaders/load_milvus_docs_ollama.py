# We load all markdown files from the folder milvus_docs/en/faq. For each document, we just simply use "# "
# to separate the content in the file, which can roughly separate the content of each main part of the markdown file.
import os, sys
from glob import glob
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.MilvusUtils import MilvusUtils


collection_name = os.getenv("OLLAMA_COLLECTION_NAME") or "milvus_ollama_collection"

client = MilvusUtils.get_client()

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
            vector = MilvusUtils.embed_text_ollama(line)
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

process()