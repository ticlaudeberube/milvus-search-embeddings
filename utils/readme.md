
## Search Demo 
- Set db_name, collection_name in envs in respective folder:

    ```$ ./search/search-envs.sh```
- Create db
    
    ```python utils/create-db my_db```
- Create collection: 

    ```python utils/create-collection.py my_collection```

- Delete collection:
    ```python utils/drop-collection.py my_collection```

- Drop db
    ```python utils/drop-db my_db```