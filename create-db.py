from MilvusUtils import MilvusUtils
import os
db_name = os.getenv("MY_DB_NAME") or "test_db"

MilvusUtils.create_database(db_name)