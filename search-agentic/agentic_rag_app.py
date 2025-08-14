import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from orchestrator.rag_orchestrator import RAGOrchestrator

load_dotenv()

st.set_page_config(
    page_title="Classification-Driven RAG Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fix emoji display and hide sidebar initially
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
    h1, h2 {
        font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", system-ui, sans-serif !important;
        font-size: 20px !important
    }
    h2, h3 {
        font-size: 18px !important
    }
    button:hover, 
    button:active,
    button:focus          {
        border-color: rgb(0, 104, 201) !important;
        color: rgb(0, 104, 201) !important;
        background-color: #ddd !important;  
    }
    # [class^="stElementContainer element-container st-emotion-cache h2"] {
    #     display: none !important;
    # }
    </style>
    """, unsafe_allow_html=True)

def get_orchestrator():
    """Initialize RAG orchestrator"""
    collection_name = os.getenv("OLLAMA_COLLECTION_NAME")
    if not collection_name:
        return None
    
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b"),
        temperature=0.0,
        num_predict=750,
        top_k=5,
        top_p=0.8,
        num_thread=8
    )
    
    return RAGOrchestrator(llm, collection_name)

def main():
    st.title("🤖 Classification-Driven RAG Pipeline")
    st.markdown("**Agentic Architecture** with intelligent query routing and semantic search")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Input form
    with st.form(key="query_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="Ask about Milvus or have a conversation...",
            key="question_input"
        )
        submit_button = st.form_submit_button("Process Query", type="primary")
    
    if submit_button and question.strip():
        # Initialize orchestrator
        if 'orchestrator' not in st.session_state:
            with st.spinner("Initializing agents..."):
                st.session_state.orchestrator = get_orchestrator()
                if st.session_state.orchestrator is None:
                    st.error('Cannot find OLLAMA_COLLECTION_NAME environment variable')
                    return
        
        # Process query through orchestrator
        with st.spinner("Processing through agent pipeline..."):
            try:
                response, doc_count = st.session_state.orchestrator.process_query(
                    question, st.session_state.chat_history
                )
                
                # Display results
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("### Agent Response")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {response}")
                
                with col2:
                    st.markdown("### Pipeline Info")
                    if doc_count > 0:
                        st.success(f"🔍 Retrieved {doc_count} documents")
                        st.info("🧠 Classification: **RETRIEVAL**")
                    else:
                        st.info("💬 Direct response")
                        st.info("🧠 Classification: **DIRECT**")
                
                # Add to history
                is_redirect = "I'm specialized in" in response and "I don't have information" in response
                if not is_redirect:
                    st.session_state.chat_history.append({
                        "question": question, 
                        "answer": response
                    })
                
                # Show conversation history
                if st.session_state.chat_history:
                    st.markdown("### Conversation History")
                    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
                        with st.expander(f"Exchange {len(st.session_state.chat_history) - i + 1}"):
                            st.markdown(f"**Q:** {chat['question']}")
                            st.markdown(f"**A:** {chat['answer']}")
                
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
    
    # Sidebar with README content
    with st.sidebar:
        if st.button("📊 Show Architecture"):
            st.session_state.show_diagram = not st.session_state.get('show_diagram', False)
        
        if st.session_state.get('show_diagram', False):
            components.html("""
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <div class="mermaid" style="width: 100%, height: auto">
                graph TD
                    A[User Query] --> B[Classification Agent]
                    B --> C{Query Type?}
                    C -->|Retrieval| D[Retrieval Agent]
                    C -->|Direct| E[Response Agent]
                    D --> F[Milvus Vector DB]
                    F --> G[Retrieved Documents]
                    G --> E
                    E --> H[Final Response]
                    M[RAG Orchestrator] --> B
                    M --> D
                    M --> E
                    style A fill:#e1f5fe
                    style B fill:#f3e5f5
                    style D fill:#e8f5e8
                    style E fill:#fff3e0
                    style F fill:#fce4ec
                    style M fill:#f0f4c3
            </div>
            <script>
                mermaid.initialize({startOnLoad:true, theme: 'default'});
            </script>
            """, height=550, scrolling=False)
        
        if 'readme_content' not in st.session_state:
            try:
                with open("./search-agentic/README.md", "r", encoding="utf-8") as f:
                    st.session_state.readme_content = f.read()
            except FileNotFoundError:
                st.session_state.readme_content = "README.md not found"
        
        st.markdown(st.session_state.readme_content)

if __name__ == "__main__":
    main()