import os,sys
from core.utils import MilvusClient

client = MilvusClient.get_client()
# TODO: Create a function to drop a database in MilvusClient similar 
# to create_database
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        db_name = sys.argv[1]
    else:
        db_name = os.getenv("MY_DB_NAME") or "demo_db"

    dbs = client.list_databases()
    print(f"List of databases: {dbs}")

    if dbs.index(db_name) >= -1:
        client.drop_database(db_name)
        print(f"Database {db_name} dropped.")
    else:
        print(f"Database {db_name} does not exist.")