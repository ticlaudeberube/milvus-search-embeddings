import os, sys
from pymilvus import model
from termcolor import colored, cprint

sys.path.insert(1, './utils')
from MilvusUtils import MilvusUtils
# https://milvus.io/docs/range-search.md

embedding_fn = model.DefaultEmbeddingFunction()
notFoundQuery = "Who is Victor Hugo?"
query = "Who is Alan Turing?"
livedIn = "Did Alan Turing lived near Turing city"
query_vectors = embedding_fn.encode_queries([query])

collection_name = os.getenv("MY_COLLECTION_NAME") or "demo_collection"

client = MilvusUtils.get_client()
# print(query_vectors[0])
#start searching
def search():
    cprint('\nSearching..\n', 'green', attrs=['blink'])
    res = client.search(
        collection_name=collection_name,  # target collection
        data=query_vectors,  # query vectors
        limit=3,  # number of returned entities
        anns_field="vector",
        search_params={"metric_type": "COSINE", "params": {"radius": 0.4, "range_filter": 0.6}},
        output_fields=["text", "subject"],  # specifies fields to be returned
    )
    #print(res)
    response = ''
    for r in res[0]:
        response += f"{r["entity"]["text"]} "

    print(f"Found ({len(res[0])}) results: {response}")
    cprint('\nSearch Complete.\n', 'green', attrs=['blink'])

search()
