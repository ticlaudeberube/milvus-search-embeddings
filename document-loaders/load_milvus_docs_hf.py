# We load all markdown files from the folder milvus_docs/en/faq. For each document, we just simply use "# "
# to separate the content in the file, which can roughly separate the content of each main part of the markdown file.
import torch, time, os, sys
from glob import glob
from tqdm import tqdm

from core.utils import MilvusClient


collection_name = os.getenv("MILVUS_HF_COLLECTION_NAME") or "demo_collection"

client = MilvusClient.get_client()

def create_collection(embedding_dim=1024):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"Collection '{collection_name}' dropped successfully.")

    client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
    )
    print(f"Collection '{collection_name}' created successfully.")
    
def process():
    start = time.time()
    text_lines = []
    for file_path in tqdm(glob("./document-loaders/milvus_docs/en/**/*.md", recursive=True), desc="Reading files"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            file_text = file.read()

        text_lines += file_text.split("# ")

    vectors =  MilvusClient.embed_text_hf(text_lines)
    if len(vectors) == 0:
        print("No vectors generated. Exiting.")
        return
    create_collection(embedding_dim=len(vectors[0]))
    data = []
    for i in range(len(vectors)):
        data.append({"id": i, "vector": vectors[i], "text": text_lines[i]})
    # print(data)
    if len(data) == 0:
        return 
    client.insert(collection_name=collection_name, data=data)
    end = time.time()
    print(f"{device} time: {end - start:.2f} seconds")

device = MilvusClient.get_device()
process()