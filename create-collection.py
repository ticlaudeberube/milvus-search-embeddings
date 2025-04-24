import os
from pymilvus import MilvusClient, client
from MilvusUtils import MilvusUtils

collection_name = os.getenv("MY_COLLECTION_NAME") or "demo_collection"
db_name = os.getenv("MY_DB_NAME")
# print(f"{db_name}.db")
# print(default_collection)

client = MilvusClient( 
    uri="http://localhost:19530",
    token="root:Milvus"
)

if client is not None:   
    MilvusUtils.create_collection(collection_name, client)
    print(f"Collection: {collection_name} created successfully.")
else:
    print("Milvus client is not initialized.")