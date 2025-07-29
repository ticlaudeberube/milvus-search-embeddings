#!/usr/bin/env python3
"""Setup script for development installation"""
import subprocess
import sys

def main():
    """Install package in development mode"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ Package installed in development mode")
        print("You can now import: from milvus_search_embeddings.utils import MilvusUtils")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()