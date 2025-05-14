import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import streamlit as st

def initialize_qa_system():
    """Initialize the QA system components"""
    collection_name = os.getenv("MILVUS_OLLAMA_COLLECTION_NAME") or "demo_collection"
    # You can use different LLM models based on your needs:
    # - For better accuracy: OllamaLLM(model="mistral") or OllamaLLM(model="llama2")
    # - For faster responses: OllamaLLM(model="orca-mini")
    # - For balanced performance: OllamaLLM(model="neural-chat")
    llm = OllamaLLM(model="llama2")    
    # You can use other embedding models depending on your needs:
    # - For better accuracy: OllamaEmbeddings(model="mistral") or OllamaEmbeddings(model="llama2")
    # - For faster performance: OllamaEmbeddings(model="nomic-embed-text")
    # - For balanced performance: OllamaEmbeddings(model="neural-chat")
    ollamaEmbeddingModel = os.getenv("MODEL_OLLAMA") or "llama3.2"
    embeddings = OllamaEmbeddings(model=ollamaEmbeddingModel)

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        drop_old=True,  # Drop the old Milvus collection if it exists
    )
    
    return vectorstore, llm

def create_qa_chain(vectorstore, llm):
    """Create the question-answering chain"""
    retriever = vectorstore.as_retriever()
    
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""
    
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def create_streamlit_ui():
    """Create the Streamlit user interface"""
    st.set_page_config(
        page_title="AI Question Answering System",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("🤖 AI Question Answering System")
    
    st.markdown("""
    ### Welcome to the AI Question Answering System
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
    qa_chain = create_qa_chain(vectorstore, llm)
    
    # Create UI
    question, submit_button = create_streamlit_ui()
    
    # Handle question answering
    if submit_button:
        if question.strip():
            with st.spinner("Generating answer..."):
                try:
                    # No need to vectorize the question manually - the Milvus retriever will handle this internally
                    # The retriever component (vectorstore.as_retriever()) automatically:
                    # 1. Vectorizes the input question using the configured embeddings model (OllamaEmbeddings)
                    # 2. Performs the vector similarity search in Milvus
                    # 3. Returns the relevant documents
                    response = qa_chain.invoke(question)
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


