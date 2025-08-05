import os
import streamlit as st
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()

from core import has_collection, create_collection, drop_collection

def initialize_qa_system():
    """Initialize the QA system components"""
    collection_name = os.getenv("HF_COLLECTION_NAME") or "demo_collection"
    
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("Please set HUGGINGFACEHUB_API_TOKEN environment variable")
        st.stop()
    
    llm = InferenceClient(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=hf_token,
        timeout=120
    )
    
    embeddingModel = os.getenv("HF_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddingModel)
    
    # Get embedding dimension and ensure collection has correct dimension
    test_embedding = embeddings.embed_query("test")
    dimension = len(test_embedding)
    
    # Recreate collection with correct dimension if needed
    if has_collection(collection_name):
        drop_collection(collection_name)
        create_collection(collection_name, dimension=dimension)

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        drop_old=False
    )
    
    return vectorstore, llm

def answer_question(question: str, vectorstore, llm):
    """Answer question using RAG with InferenceClient"""
    docs = vectorstore.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    prompt = f"""Human: You are an AI assistant that provides answers using factual information.
        Use the context below to answer the question. If you don't know, say so.

        Context:
        {context}

        Question: {question}

        Assistant:"""
    
    messages = [{"role": "user", "content": prompt}]
    response = llm.chat_completion(messages, max_tokens=512)
    return response.choices[0].message.content

def create_streamlit_ui():
    """Create the Streamlit user interface"""
    st.set_page_config(
        page_title="AI Question Answering System",
        page_icon="🤖",
        layout="centered"
    )
    
    # st.title("🤖 AI Question Answering System")
    
    st.markdown("""
    ### Welcome to the Milvus Docs Answering System
    Ask any question and get AI-powered answers based on the available knowledge base.
    """)
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    question = st.text_input(
        "Enter your question:",
        placeholder="Type your question here...",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("Get Answer", type="primary")
    
    return question, submit_button

def main():
    # Initialize system components
    vectorstore, llm = initialize_qa_system()
    
    # Create UI
    question, submit_button = create_streamlit_ui()
    
    # Handle question answering
    if submit_button:
        if question.strip():
            with st.spinner("Generating answer..."):
                try:
                    # No need to vectorize the question manually - the Milvus retriever will handle this internally
                    # The retriever component (vectorstore.as_retriever()) automatically:
                    # 1. Vectorizes the input question using the configured embeddings model (HuggingFaceEmbeddings)
                    # 2. Performs the vector similarity search in Milvus
                    # 3. Returns the relevant documents
                    response = answer_question(question, vectorstore, llm)
                    st.success("Answer generated successfully!")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": question, "answer": response})
                    
                    # Display chat history
                    st.markdown("### Chat History")
                    for chat in st.session_state.chat_history:
                        st.markdown(f"**Q:** {chat['question']}")
                        st.markdown(f"**A:** {chat['answer']}")
                        st.markdown("---")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question first.")
    
    # Add footer
    st.markdown("---")
    st.markdown("*Powered by LangChain and Milvus*")

if __name__ == "__main__":
    main()