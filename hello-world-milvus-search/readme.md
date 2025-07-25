
## Search Demo 
- Set db_name, collection_name in envs file and run:

    ```$ ./hello-world-milvus-search/search-envs.sh```
- Create db
    
    ```python utils/create_db.py my_db```
- Create collection: 

    ```python create-collection.py my_colletion```
- Vectorize:

    ```python hello-world-milvus-search/vectorize-search.py```
- Search: 

    ```python hello-world-milvus-search/range-search.py```