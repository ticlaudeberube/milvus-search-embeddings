
## Milvus RAG Search Demo 
This ddmeo will search Milvus Documentation.
- Set db_name, collection_name in envs file and run:

    ```$ . ./search/search-envs.sh```

- Load Documents:

    ```
        $ python ./documents/get_milvus_docs.py
    ```

    ```
        $ python ./documents/load_milvus_docs.py
    ```

- Run RAG:
    
    ```
        $ Python ./search-rag/search-ollama-rag.py
    ```