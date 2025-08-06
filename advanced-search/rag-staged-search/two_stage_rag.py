import os
from langchain_ollama.llms import OllamaLLM
import streamlit as st

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

# Hide Streamlit UI elements and fix button colors
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

def get_rag_system():
    """Initialize RAG system only when needed"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        return None
    
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b"),
        temperature=0.0,  # Deterministic for faster responses
        num_predict=750,   # Longer responses
        top_k=5,         # Smaller sampling space
        top_p=0.8,       # More focused
        num_thread=8     # Use more CPU threads
    )
    
    client = get_client()
    client.load_collection(collection_name)
    return RAGCore(llm, collection_name)

def initialize_session_state():
    """Initialize session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'docs_loaded' not in st.session_state:
        st.session_state.docs_loaded = False

def create_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 AI Assistant")
        st.markdown("**Status:** Ready")
        st.markdown("**Model:** Ollama + Milvus")
        st.markdown("---")
        
        if st.button("📖 Toggle Documentation", type="secondary"):
            st.session_state.docs_loaded = not st.session_state.docs_loaded
        
        if st.session_state.docs_loaded:
            try:
                readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
                with open(readme_path, "r", encoding="utf-8") as file:
                    st.markdown(file.read())
            except FileNotFoundError:
                st.error("README.md not found")
        
        st.markdown("**Tips:**")
        st.markdown("• Ask about Milvus database")
        st.markdown("• Give me instructions")
        st.markdown("• View conversation history below")


def create_streamlit_ui():
    create_sidebar()
    
    st.title("🤖 AI Question Answering System")
    st.markdown("Ask questions about Milvus database or give me instructions.")
    
    with st.form(key="question_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="Type your question here...",
            key="question_input"
        )
        
        submit_button = st.form_submit_button("Get Answer", type="primary")
    
    return question, submit_button

def main():
    initialize_session_state()
    
    question, submit_button = create_streamlit_ui()
    
    if submit_button:
        if question.strip():
            # Initialize RAG system only when first question is asked
            if 'rag_core' not in st.session_state:
                with st.spinner("Initializing system..."):
                    st.session_state.rag_core = get_rag_system()
                    if st.session_state.rag_core is None:
                        st.error('Cannot find OLLAMA_COLLECTION_NAME environment variable')
                        return
            
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
            # Only show warning if no chat history exists
            if not st.session_state.chat_history:
                st.warning("⚠️ Please enter a question first.")
    
    st.markdown("---")
    st.markdown("*Powered by Ollama and Milvus*")

if __name__ == "__main__":
    main()