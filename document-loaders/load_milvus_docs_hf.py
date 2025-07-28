# We load all markdown files from the folder milvus_docs/en/faq. For each document, we just simply use "# "
# to separate the content in the file, which can roughly separate the content of each main part of the markdown file.
import time
import os
import sys
from glob import glob
from typing import List, Dict, Any
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.MilvusUtils import MilvusUtils


collection_name: str = os.getenv("HF_COLLECTION_NAME") or "demo_collection"

client = MilvusUtils.get_client()

def create_collection(embedding_dim: int = 1024) -> None:
    if MilvusUtils.has_collection(collection_name):
        MilvusUtils.drop_collection(collection_name)
        print(f"Collection '{collection_name}' dropped successfully.")

    MilvusUtils.create_collection(collection_name, dimension=embedding_dim)
    print(f"Collection '{collection_name}' created successfully.")
    
def process() -> None:
    start = time.time()
    text_lines: List[str] = []
    for file_path in tqdm(glob("./document-loaders/milvus_docs/en/**/*.md", recursive=True), desc="Reading files"):
        with open(file_path, "r", encoding="utf-8") as file:
            file_text: str = file.read()

        text_lines += file_text.split("# ")

    vectors: List[List[float]] = MilvusUtils.embed_text_hf(text_lines)
    create_collection(embedding_dim=len(vectors[0]))
    data: List[Dict[str, Any]] = []
    for i in range(len(vectors)):
        data.append({"id": i, "vector": vectors[i], "text": text_lines[i]})
    # print(data)
    if len(data) == 0:
        return 
    client.insert(collection_name=collection_name, data=data)
    end = time.time()
    print(f"{device} time: {end - start:.2f} seconds")

device: Any = MilvusUtils.get_device()
process()