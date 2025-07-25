
## Milvus RAG Search Demo 
This ddmeo will search Milvus Documentation.
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
        $ python advanced-search-rag/search-[hr|ollama]-gradio-chat.py
    ```
    
    ```
        $ streamlit run advanced-search-rag/search-[hf|ollama]-streamlit-rag.py
    ```