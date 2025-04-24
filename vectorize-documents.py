import os

from pymilvus import MilvusClient
from MilvusUtils import MilvusUtils
client = MilvusClient( 
    uri="http://localhost:19530",
    token="root:Milvus"
)
collection_name = os.getenv("MY_COLLECTION_NAME") or "demo_collection"

documents = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

MilvusUtils.vectorize_documents(collection_name, documents, client)