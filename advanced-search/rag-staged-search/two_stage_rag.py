import os
from langchain_ollama.llms import OllamaLLM
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from core import MilvusUtils
from rag_core import needs_retrieval, rag_query_with_retrieval, direct_response, optimized_rag_query

def initialize_qa_system():
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        st.error('Cannot find OLLAMA_COLLECTION_NAME environment variable')
        return None, None
    
    llm = OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"))    
    client = MilvusUtils.get_client()
    
    return client, llm, collection_name



def load_readme_content():
    """Load README.md content"""
    try:
        readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
        with open(readme_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "README.md file not found."

def create_sidebar():
    """Create sidebar with README content"""
    with st.sidebar:
        st.markdown("## 📖 Project Documentation")
        readme_content = load_readme_content()
        st.markdown(readme_content)

def create_streamlit_ui():
    st.set_page_config(
        page_title="AI Question Answering System",
        page_icon="🤖",
        layout="wide"
    )
    
    create_sidebar()
    
    st.title("🤖 AI Question Answering System")
    
    st.markdown("""
    ### Welcome to the Milvus Question Answering System
    Ask questions about Milvus database or give me instructions. I'll use document retrieval when needed.
    """)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    with st.form(key="question_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="Type your question here...",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Get Answer", type="primary")
    
    return question, submit_button

def main():
    result = initialize_qa_system()
    if result is None or result[0] is None:
        return
    
    client, llm, collection_name = result
    
    question, submit_button = create_streamlit_ui()
    
    if submit_button:
        if question.strip():
            with st.spinner("Generating answer..."):
                try:
                    response, doc_count = optimized_rag_query(client, llm, collection_name, question, st.session_state.chat_history)
                    
                    if doc_count > 0:
                        st.info(f"Retrieved {doc_count} documents from knowledge base")
                    else:
                        st.info("Answered directly without document retrieval")
                    
                    st.success("Answer generated successfully!")
                    
                    st.session_state.chat_history.append({"question": question, "answer": response})
                    
                    st.markdown("### Chat History")
                    for chat in reversed(st.session_state.chat_history):
                        st.markdown(f"**Q:** {chat['question']}")
                        st.markdown(f"**A:** {chat['answer']}")
                        st.markdown("---")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question first.")
    
    st.markdown("---")
    st.markdown("*Powered by Ollama and Milvus*")

if __name__ == "__main__":
    main()