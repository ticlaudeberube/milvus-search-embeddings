import os, sys
from pymilvus import client
from core.utils import MilvusClient

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_name = sys.argv[1] 
    else:
        collection_name = "demo_collection"

    client = MilvusClient.get_client()

    if client is not None:
        if client.has_collection(collection_name):
            print(f"Collection: {collection_name} already exists.")
        else:
            MilvusClient.create_collection(collection_name)
            print(f"Collection: {collection_name} created successfully.")
    else:
        print("Milvus client is not initialized.")