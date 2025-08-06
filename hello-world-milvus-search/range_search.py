import os
from typing import List
from pymilvus import model, MilvusClient
from termcolor import cprint
from dotenv import load_dotenv

from core import get_client

load_dotenv()

# Constants
QUERY: str = "Who is Alan Turing?"
SEARCH_LIMIT: int = 3
RADIUS: float = 0.4
RANGE_FILTER: float = 0.6
OUTPUT_FIELDS: List[str] = ["text", "subject"]

embedding_fn = model.DefaultEmbeddingFunction()
query_vectors = embedding_fn.encode_queries([QUERY])
collection_name: str = os.getenv("MY_COLLECTION_NAME") or "hello_world_collection"
client: MilvusClient = get_client()

def search() -> None:
    """Perform range search on Milvus collection."""
    cprint('\nSearching...\n', 'green', attrs=['blink'])
    
    try:
        res = client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=SEARCH_LIMIT,
            anns_field="vector",
            search_params={
                "metric_type": "COSINE", 
                "params": {"radius": RADIUS, "range_filter": RANGE_FILTER}
            },
            output_fields=OUTPUT_FIELDS,
        )
        
        response: str = ' '.join(r["entity"]["text"] for r in res[0])
        print(f"Found ({len(res[0])}) results: {response}")
        
    except Exception as e:
        print(f"Search error: {e}")
    finally:
        cprint('\nSearch Complete.\n', 'green', attrs=['blink'])

if __name__ == "__main__":
    search()