
## Milvus RAG Search Demo 
This demo will search Milvus Documentation.
- Set db_name, collection_name in envs file and run:

    ```$ . ./environments/set-envs.sh```

- Load Documents:

    ```
        $ python ./document-loaders/download_milvus_docs.py
    ```

    ```
        $ python ./document-loaders/load_milvus_docs_[hf|ollama].py
    ```

- Run:
     ```
        $ streamlit run search-advanced/search_ollama_streamlit_rag.py
    ```
    ```
        $ streamlit run search-advanced/search_hf_streamlit_rag.py
    ```