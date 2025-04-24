import os
from pymilvus import MilvusClient, model

embedding_fn = model.DefaultEmbeddingFunction()
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

collection_name = os.getenv("MY_COLLECTION_NAME") or "demo_collection"
client = MilvusClient( 
    uri="http://localhost:19530",
    token="root:Milvus"
)
def search():
    res = client.search(
        collection_name=collection_name,  # target collection
        data=query_vectors,  # query vectors
        limit=2,  # number of returned entities
        output_fields=["text", "subject"],  # specifies fields to be returned
    )
    response = ''
    for r in res[0]:
        response += r["entity"]["text"]
    
    return response

search = search()
print(search)