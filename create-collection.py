import os, sys
from pymilvus import MilvusClient, client
from MilvusUtils import MilvusUtils

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_name = sys.argv[1] 
    else:
        collection_name = "demo_collection"

    # db_name = os.getenv("MY_DB_NAME")
    # print(f"{db_name}.db")
    # print(collection_name)

    client = MilvusClient( 
        uri="http://localhost:19530",
        token="root:Milvus"
    )

    if client is not None:
        if client.has_collection(collection_name):
            print(f"Collection: {collection_name} already exists.")
        else:
            MilvusUtils.create_collection(collection_name, client)
            print(f"Collection: {collection_name} created successfully.")
    else:
        print("Milvus client is not initialized.")