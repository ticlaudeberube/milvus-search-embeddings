#!/usr/bin/env python3
from core.database_manager import drop_database
import sys
import os

if __name__ == "__main__":
    db_name: str = sys.argv[1] if len(sys.argv) > 1 else (os.getenv("MY_DB_NAME") or "demo_db")
    drop_database(db_name)