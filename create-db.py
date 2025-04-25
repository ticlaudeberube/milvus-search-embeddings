from MilvusUtils import MilvusUtils
import os,sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_name = sys.argv[1] 
    else:
        db_name = os.getenv("MY_DB_NAME") or "demo_db"

MilvusUtils.create_database(db_name)