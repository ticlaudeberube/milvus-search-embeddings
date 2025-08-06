"""
This script vectorizes a list of documents and inserts them into a Milvus collection.
"""

from typing import List
from core import vectorize_documents

# Constants
COLLECTION_NAME: str = "hello_world_collection"

DOCUMENTS: List[str] = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
    "Turing city is located in the state of New York",
    "He never lived in Turing city",
]

def main() -> None:
    """Vectorize documents and insert into Milvus collection."""
    try:
        vectorize_documents(COLLECTION_NAME, DOCUMENTS)
        print(f"Successfully vectorized {len(DOCUMENTS)} documents into {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error vectorizing documents: {e}")

if __name__ == "__main__":
    main()