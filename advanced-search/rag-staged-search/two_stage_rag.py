import os
from langchain_ollama.llms import OllamaLLM
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from core import get_client
from rag_core import RAGCore

# Set page config FIRST to prevent flash
st.set_page_config(
    page_title="AI Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit UI elements for faster load
if 'ui_hidden' not in st.session_state:
    st.session_state.ui_hidden = True
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def get_rag_system():
    """Cache the RAG system initialization with parallel loading"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        return None
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Initialize LLM and client in parallel
        llm_future = executor.submit(lambda: OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b")))
        client_future = executor.submit(get_client)
        
        llm = llm_future.result()
        client = client_future.result()
        client.load_collection(collection_name)
        
    return RAGCore(llm, collection_name)

def initialize_session_state():
    """Initialize session state efficiently with preloading"""
    if 'rag_core' not in st.session_state:
        st.session_state.rag_core = get_rag_system()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

@st.cache_data
def load_readme_content():
    """Cache README content"""
    try:
        readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
        with open(readme_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "README.md file not found."

def create_sidebar():
    with st.sidebar:
        st.markdown("## 📖 Project Documentation")
        if st.button("Load Documentation"):
            readme_content = load_readme_content()
            st.markdown(readme_content)


def create_streamlit_ui():
    create_sidebar()
    
    st.title("🤖 AI Question Answering System")
    
    # Show status based on system readiness
    if st.session_state.get('rag_core'):
        st.markdown("### Ready to answer your questions!")
        st.markdown("Ask questions about Milvus database or give me instructions.")
    else:
        st.markdown("### Loading system...")
    
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
    initialize_session_state()
    
    question, submit_button = create_streamlit_ui()
    
    if st.session_state.rag_core is None:
        st.error('Cannot find OLLAMA_COLLECTION_NAME environment variable')
        return
    
    if submit_button:
        if question.strip():
            with st.spinner("Generating answer..."):
                try:
                    response, doc_count = st.session_state.rag_core.query(question, st.session_state.chat_history)
                    
                    if doc_count > 0:
                        st.info(f"Retrieved {doc_count} documents from knowledge base")
                    else:
                        st.info("Answered directly without document retrieval")
                    
                    st.success("Answer generated successfully!")
                    
                    # Display the current response
                    st.markdown("### Current Response")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {response}")
                    st.markdown("---")
                    
                    # Display chat history if available (excluding current question)
                    if st.session_state.chat_history:
                        st.markdown("### Previous Conversations")
                        for chat in reversed(st.session_state.chat_history[-5:]):
                            st.markdown(f"**Q:** {chat['question']}")
                            st.markdown(f"**A:** {chat['answer']}")
                            st.markdown("---")
                    
                    # Add to UI history after displaying (if not off-topic)
                    is_redirect = "I'm specialized in" in response and "I don't have information" in response
                    if not is_redirect:
                        st.session_state.chat_history.append({"question": question, "answer": response})
                    else:
                        st.warning("⚠️ Question was off-topic and not saved to conversation history.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question first.")
    
    st.markdown("---")
    st.markdown("*Powered by Ollama and Milvus*")

if __name__ == "__main__":
    main()