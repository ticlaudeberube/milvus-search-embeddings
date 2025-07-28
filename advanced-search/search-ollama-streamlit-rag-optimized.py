import os
import sys
from langchain_ollama.llms import OllamaLLM
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.MilvusUtils import MilvusUtils

def initialize_qa_system():
    """Initialize the QA system components"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        st.error('Cannot find OLLAMA_COLLECTION_NAME environment variable')
        return None, None
    
    llm = OllamaLLM(model="llama2")
    client = MilvusUtils.get_client()
    
    return client, llm, collection_name

def is_milvus_related(llm, question):
    """Stage 1: Quick classification to check if question is Milvus-related"""
    classification_prompt = f"""
Human: Is this question related to Milvus database, vector databases, or data storage/retrieval? 
Answer only "YES" or "NO".

Question: {question}