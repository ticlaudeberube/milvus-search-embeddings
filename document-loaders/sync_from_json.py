import json
import os
from dotenv import load_dotenv
load_dotenv()

from core import get_client

collection_name = os.getenv("OLLAMA_COLLECTION_NAME") or "milvus_ollama_collection"
client = get_client()

def sync_embeddings():
    """Sync embeddings from JSON file - adds, updates, and removes vectors"""
    with open("./data/embeddings.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not data:
        print("No data to sync")
        return
    
    if not client.has_collection(collection_name):
        dimension = len(data[0]['vector'])
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type="COSINE",
            consistency_level="Session",
        )
        print(f"Created collection: {collection_name}")
        existing_checksums = {}
    else:
        # Get existing data
        existing = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id", "checksum"],
            limit=16384
        )
        existing_checksums = {item["id"]: item["checksum"] for item in existing}
    
    # Prepare data for upsert
    json_ids = {item["id"] for item in data}
    existing_ids = set(existing_checksums.keys())
    
    to_upsert = []
    new_count = 0
    updated_count = 0
    
    for item in data:
        item_id = item["id"]
        if item_id not in existing_checksums:
            new_count += 1
            to_upsert.append(item)
        elif existing_checksums[item_id] != item["checksum"]:
            updated_count += 1
            to_upsert.append(item)
    
    # Delete vectors not in JSON
    to_delete = existing_ids - json_ids
    if to_delete:
        client.delete(collection_name=collection_name, filter=f"id in {list(to_delete)}")
        print(f"Deleted {len(to_delete)} vectors")
    
    # Upsert new/changed vectors
    if to_upsert:
        client.upsert(collection_name=collection_name, data=to_upsert)
        print(f"Upserted {new_count} new, {updated_count} updated documents")
    else:
        print("No documents need updating")

if __name__ == "__main__":
    sync_embeddings()