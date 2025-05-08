# About
This is a Python POC for Milvus embedings, search and Milvus Docs RAG

## Milvus Db

- [Install Milvus Docker Container](https://milvus.io/docs/install_standalone-docker.md)
- Launch Milvus Docker container 
- Access web UI: http://127.0.0.1:9091/webui/

## Document folder
Contains scripts for downloading and embed documents

## Search folder
Contains basic embedding Alan Touring search 

## Utils folder
Contains MilvusUitls custom satic class which is used by other scripts

Script can be parameterized by adding a colelction_name or db_name on creation or deletion
 
```$ python ./utils/create-collection.py  my_collection```

## Test
- Run pytest with coverage

    ``` $ coverage run -m pytest tests/test_MilvusUtils.py ```

    ``` $ coverage report -m ```

## [Milvuv_cli](https://milvus.io/docs/cli_commands.md)
- milvus_cli 
- connect -uri http://localhost:19530
- list databases
- list collections
- create database --db_name test
- use database --db_name test
- use database --db_name default
- delete database --db_name test