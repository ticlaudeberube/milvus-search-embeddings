import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import streamlit as st  # type: ignore


@dataclass
class QAConfig:
    """Configuration for the QA system"""
    collection_name: str = "demo_collection"
    llm_repo_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    max_tokens: int = 512
    drop_old_collection: bool = True


class MilvusQASystem:
    """Milvus-based Question Answering System"""
    
    def __init__(self, config: Optional[QAConfig] = None):
        """Initialize the QA system with the given configuration"""
        self.config = config or self._load_config_from_env()
        self.vectorstore, self.llm = self._initialize_components()
        self.qa_chain = self._create_qa_chain()
    
    def _load_config_from_env(self) -> QAConfig:
        """Load configuration from environment variables"""
        return QAConfig(
            collection_name=os.getenv("MILVUS_HF_COLLECTION_NAME", "demo_collection"),
            embedding_model=os.getenv("HF_EMBEDDING_MODEL"),
            llm_repo_id=os.getenv("LLM_REPO_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            drop_old_collection=os.getenv("DROP_OLD_COLLECTION", "True").lower() == "true"
        )
    
    def _initialize_components(self):
        """Initialize the LLM and vector store components"""
        llm = HuggingFaceEndpoint(
            repo_id=self.config.llm_repo_id,
            task="text-generation",
            max_new_tokens=self.config.max_tokens,
            do_sample=False,
            repetition_penalty=1.03,
        )
        
        embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        
        vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=self.config.collection_name,
            drop_old=self.config.drop_old_collection,
        )
        
        return vectorstore, llm
    
    def _create_qa_chain(self):
        """Create the question-answering chain"""
        retriever = self.vectorstore.as_retriever()
        
        prompt_template = """
        Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide readable text and avoid using code blocks.
        {context}
        </context>

        <question>
        {question}
        </question>

        The response should be specific and use statistics or numbers when possible.

        Assistant:"""
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def answer_question(self, question: str) -> str:
        """Process a question and return the answer"""
        return self.qa_chain.invoke({"question": question})


class StreamlitUI:
    """Streamlit user interface for the QA system"""
    
    def __init__(self, qa_system: MilvusQASystem):
        """Initialize the UI with the QA system"""
        self.qa_system = qa_system
        self._setup_page()
        
    def _setup_page(self):
        """Set up the Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Question Answering System",
            page_icon="🤖",
            layout="centered"
        )
        
        st.markdown("""
        ### Welcome to the Milvus Docs Answering System
        Ask any question and get AI-powered answers based on the available knowledge base.
        """)
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def render(self):
        """Render the UI and handle user interactions"""
        question = st.text_input(
            "Enter your question:",
            placeholder="Type your question here...",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.button("Get Answer", type="primary")
        
        self._handle_question(question, submit_button)
        self._display_chat_history()
        self._render_footer()
    
    def _handle_question(self, question: str, submit_button: bool):
        """Handle question submission and answer generation"""
        if submit_button and question.strip():
            with st.spinner("Generating answer..."):
                try:
                    response = self.qa_system.answer_question(question)
                    st.success("Answer generated successfully!")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question, 
                        "answer": response
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif submit_button:
            st.warning("⚠️ Please enter a question first.")
    
    def _display_chat_history(self):
        """Display the chat history"""
        if st.session_state.chat_history:
            st.markdown("### Chat History")
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.markdown("---")
    
    def _render_footer(self):
        """Render the page footer"""
        st.markdown("---")
        st.markdown("*Powered by LangChain and Milvus*")


def main():
    """Main application entry point"""
    qa_system = MilvusQASystem()
    ui = StreamlitUI(qa_system)
    ui.render()


if __name__ == "__main__":
    main()