import os
from pymilvus import MilvusClient, client
from MilvusUtils import MilvusUtils
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_name = sys.argv[1] 
    else:
        collection_name = "demo_collection"

    client = MilvusUtils.get_client()

    if client is not None: 
        if client.has_collection(collection_name):
            MilvusUtils.delete_collection(collection_name)
            print(f"Collection: {collection_name} deleted successfully.")
        else:
            print(f"Collection: {collection_name} does not exist.")
    else:
        print("Milvus client is not initialized.")