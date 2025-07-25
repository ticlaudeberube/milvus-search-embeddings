#!/usr/bin/env python3
from core.collection_manager import create_collection
import sys

if __name__ == "__main__":
    collection_name: str = sys.argv[1] if len(sys.argv) > 1 else "demo_collection"
    create_collection(collection_name)