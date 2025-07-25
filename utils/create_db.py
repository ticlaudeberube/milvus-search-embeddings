from core.utils import MilvusClient
import os, sys
from pymilvus import connections

# Establish connection first
connections.connect("default", host="localhost", port="19530")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_name = sys.argv[1] 
    else:
        db_name = os.getenv("MY_DB_NAME") or "demo_db"

    MilvusClient.create_database(db_name)