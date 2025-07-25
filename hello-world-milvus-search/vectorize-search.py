"""
This script vectorizes a list of documents and inserts them into a Milvus collection.

Usage:
    Set the environment variable MY_COLLECTION_NAME to specify the collection name, or it defaults to 'demo_collection'.
    The script uses MilvusClient from the utils directory.
"""

import os
import sys

from core.utils import MilvusClient

collection_name = os.getenv("MY_COLLECTION_NAME") or "demo_collection"

documents = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
    "Turing city is located in the state of New York",
    "He never lived in Turing city",
]

MilvusClient.vectorize_documents(collection_name, documents)