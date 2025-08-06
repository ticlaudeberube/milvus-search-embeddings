
## Milvus RAG Search Demo 
This demo will search Milvus Documentation.
- Set db_name, collection_name in envs file and run:

    ```$ . ./environments/set-envs.sh```

- Load Documents:

    ```
        $ python ./documents/get_milvus_docs.py
    ```

    ```
        $ python ./documents/load_milvus_docs_[hf|ollama].py
    ```

- Run:
     ```
        $ streamlit run advanced-search/search-rag-streamlit.py
    ```
    ```
        $ streamlit run advanced-search/search-rag-streamlit.py -- --model-provider=huggingface
    ```