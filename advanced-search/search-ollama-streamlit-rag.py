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

def rag_query(client, llm, collection_name, question):
    """Perform RAG query using direct Milvus client"""
    # Search for relevant documents
    search_res = client.search(
        collection_name=collection_name,
        data=[MilvusUtils.embed_text_ollama(question)],
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"radius": 0.4, "range_filter": 0.7}},
        output_fields=["text"]
    )
    
    # Extract context from search results
    context = "\n\n".join([res["entity"]["text"] for res in search_res[0]])
    
    # Create prompt
    prompt = f"""
Human: You are an AI assistant specialized in Milvus Database Documentation. You ONLY answer questions about Milvus database. 
If a question is not related to Milvus, respond with: "I can only answer questions about Milvus database. Please ask a Milvus-related question."
Use the context below to provide accurate, fact-based answers about Milvus.
 Do not include any XML tags like <response> or </response> in your answer.

<context>
{context}
</context>

<question>
{question}
</question>

Assistant:"""
    
    # Generate response
    response = llm.invoke(prompt)
    return response, len(search_res[0])

def create_streamlit_ui():
    """Create the Streamlit user interface"""
    st.set_page_config(
        page_title="AI Question Answering System",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("🤖 AI Question Answering System")
    
    st.markdown("""
    ### Welcome to the Milvus Question Answering System
    Ask questions about Milvus database and get AI-powered answers. Only Milvus-related questions are accepted.
    """)
    
    # Initialize session state for chat history
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
    # Initialize system components
    result = initialize_qa_system()
    if result is None or result[0] is None:
        return
    
    client, llm, collection_name = result
    
    # Create UI
    question, submit_button = create_streamlit_ui()
    
    # Handle question answering
    if submit_button:
        if question.strip():
            with st.spinner("Generating answer..."):
                try:
                    response, doc_count = rag_query(client, llm, collection_name, question)
                    st.info(f"Retrieved {doc_count} documents from knowledge base")
                    st.success("Answer generated successfully!")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": question, "answer": response})
                    
                    # Display chat history (latest first)
                    st.markdown("### Chat History")
                    for chat in reversed(st.session_state.chat_history):
                        st.markdown(f"**Q:** {chat['question']}")
                        st.markdown(f"**A:** {chat['answer']}")
                        st.markdown("---")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question first.")
    
    # Add footer
    st.markdown("---")
    st.markdown("*Powered by Ollama and Milvus*")

if __name__ == "__main__":
    main()


