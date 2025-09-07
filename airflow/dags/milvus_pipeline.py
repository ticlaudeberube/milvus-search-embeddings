from datetime import datetime
from airflow.decorators import dag, task
from milvus_pipeline_functions import (
    download_and_extract as _download_and_extract,
    load_files as _load_files,
    process_input as _process_input,
    produce_embeddings as _produce_embeddings,
    load_to_milvus as _load_to_milvus
)

@dag(
    dag_id='milvus_docs_pipeline_dry',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=['milvus', 'embeddings', 'docs'],
    description='DRY Milvus docs pipeline using shared functions',
)
def milvus_docs_pipeline():
    
    @task
    def download_and_extract():
        return _download_and_extract()
    
    @task
    def load_files(extract_path):
        return _load_files(extract_path)

    @task
    def process_input(text_lines):
        return _process_input(text_lines)

    @task
    def produce_embeddings(chunks):
        return _produce_embeddings(chunks)

    @task
    def load_to_milvus(data):
        return _load_to_milvus(data)
    
    # Define task flow
    extract_path = download_and_extract()
    text_lines = load_files(extract_path)
    chunks = process_input(text_lines)
    embeddings = produce_embeddings(chunks)
    load_to_milvus(embeddings)

dag_instance = milvus_docs_pipeline()

if __name__ == "__main__":
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Set AIRFLOW_HOME
    airflow_home = Path(__file__).parent.parent
    os.environ["AIRFLOW_HOME"] = str(airflow_home)
    
    # Initialize DB if needed
    try:
        subprocess.run([sys.executable, "-m", "airflow", "db", "migrate"], 
                      check=True, capture_output=True)
        print("\u2705 Airflow DB initialized")
    except subprocess.CalledProcessError:
        print("\u26a0\ufe0f DB already initialized or error occurred")
    
    # Test the DAG
    print("\ud83d\ude80 Testing DAG...")
    dag_instance.test()