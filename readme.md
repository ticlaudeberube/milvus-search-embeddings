## Milvus Db

- Install Milvus Docker Container: https://milvus.io/docs/install_standalone-docker.md
- Launch Milvus Docker container 
- Access web UI: http://127.0.0.1:9091/webui/

## Demo
- Set db-name, collection_name in envs file and run $ . ./envs.py
- Create db: run create-milvus-db 
- Create collection: run create-collection.py
- Run vectorize-document.py
- Run search.py