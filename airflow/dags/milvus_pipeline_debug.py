#!/usr/bin/env python3
"""Standalone Milvus docs pipeline - no Airflow dependencies"""

from milvus_pipeline_functions import (
    download_and_extract,
    load_files,
    process_input,
    produce_embeddings,
    load_to_milvus
)

def main():
    """Run the complete pipeline"""
    print("🚀 Starting Milvus docs pipeline...")
    
    extract_path = download_and_extract()
    text_lines = load_files(extract_path)
    chunks = process_input(text_lines)
    embeddings = produce_embeddings(chunks)
    load_to_milvus(embeddings)
    
    print("\u2705 Pipeline completed successfully!")

if __name__ == "__main__":
    main()