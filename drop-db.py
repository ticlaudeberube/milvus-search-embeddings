from pymilvus import connections, db, MilvusClient
import os,sys

client = MilvusClient( 
    uri="http://localhost:19530",
    token="root:Milvus"
)

if __name__ == "__main__":
    # conn = connections.connect(host="localhost", port=19530)
    
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