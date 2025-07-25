#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.MilvusUtils import MilvusUtils

if __name__ == "__main__":
    collection_name: str = sys.argv[1] if len(sys.argv) > 1 else "demo_collection"
    MilvusUtils.drop_collection(collection_name)