# We load all markdown files from the folder milvus_docs/en/faq. For each document, we just simply use "# "
# to separate the content in the file, which can roughly separate the content of each main part of the markdown file.
import ollama, os, sys
from glob import glob
from tqdm import tqdm
from pymilvus import MilvusClient

sys.path.insert(1, './utils')
from MilvusUtils import MilvusUtils


collection_name = os.getenv("MY_COLLECTION_NAME") or "demo_collection"

client = MilvusUtils.get_client()

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

def emb_text(text):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    return response["embedding"]

def process():
    text_lines = []

    for file_path in tqdm(glob("./documents/milvus_docs/en/**/*.md", recursive=True), desc="Reading files"):
        with open(file_path, "r") as file:
            file_text = file.read()

        text_lines += file_text.split("# ")

    create_collection()
    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(line), "text": line})
    #print(data)
    client.insert(collection_name=collection_name, data=data)

process()