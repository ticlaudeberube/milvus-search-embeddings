import os
import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Any

# Import langchain globals for verbose setting
from langchain.globals import set_verbose

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM
from langchain_milvus import Milvus
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st  # type: ignore

# Set verbose to False to avoid unnecessary output
set_verbose(False)
# ************************************************************
# THIS WAS CODE NEEDS TO BE REFACTORED
# Remove specific model provider in favor of a more generic approach
# ************************************************************


# Global variables for model configuration
GLOBAL_CONFIG = {
    "model_provider": "huggingface",
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "collection": "milvus_hf_collection"
}

class ModelProvider(str, Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

@dataclass
class QAConfig:
    """Configuration for the QA system"""
    collection_name: str = os.getenv("MILVUS_HF_COLLECTION_NAME", "demo_collection")
    embedding_model=os.getenv("MODEL_HF", "sentence-transformers/all-mpnet-base-v2"),
    model_provider: str = "huggingface"
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    model: str = os.getenv("MODEL_HF", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    def __post_init__(self):
        """Set the appropriate model based on provider"""
        self.embedding_model = self.embedding_model[0] if isinstance(self.embedding_model, tuple) else self.embedding_model
        self.model = self.model[0] if isinstance(self.model, tuple) else self.model

class QASystem:
    """Milvus-based Question Answering System"""
    
    def __init__(self, config: Optional[QAConfig] = None):
        self.config = config or QAConfig()
        self.vectorstore, self.llm = self._setup_components()
        self.qa_chain = self._create_chain()
    
    def _setup_components(self) -> Tuple[Any, LLM]:
        """Initialize LLM and vector store"""
        # Initialize LLM based on provider
        try:
            if self.config.model_provider == "huggingface":
                # Import HuggingFace modules only when needed
                try:
                    from langchain_huggingface import HuggingFaceEndpoint
                    from langchain_community.llms import OpenAI
                    
                    # Try to use HuggingFace with appropriate task based on model
                    try:
                        llm = HuggingFaceEndpoint(
                            repo_id=self.config.model,
                            task="conversational",
                            max_new_tokens=self.config.max_tokens,
                            do_sample=False,
                            repetition_penalty=1.03,
                        )
                    except Exception as e:
                        # Fallback to a more reliable model
                        print(f"Error with HuggingFace model: {str(e)}")
                        llm = OpenAI()
                except ImportError:
                    from langchain_community.llms import FakeListLLM
                    llm = FakeListLLM(responses=["Sorry, HuggingFace module could not be loaded."])
                    
                # Use simple embeddings to avoid NumPy issues
                embeddings = FakeEmbeddings(size=768)
                
            else:  # Ollama
                # Import Ollama modules only when needed
                try:
                    from langchain_ollama.llms import OllamaLLM
                    from langchain_ollama import OllamaEmbeddings
                    
                    llm = OllamaLLM(
                        model=self.config.model,
                        num_predict=self.config.max_tokens,
                        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    )
                    ollamaEmbeddingModel = os.getenv("MODEL_OLLAMA") or "llama3.2"
                    embeddings = OllamaEmbeddings(model=ollamaEmbeddingModel)
                except ImportError:
                    from langchain_community.llms import FakeListLLM
                    llm = FakeListLLM(responses=["Sorry, Ollama module could not be loaded."])
                    embeddings = FakeEmbeddings(size=768)
        except Exception as e:
            # Last resort fallback
            from langchain_community.llms import FakeListLLM
            llm = FakeListLLM(responses=[f"Error initializing components: {str(e)}"])
            embeddings = FakeEmbeddings(size=768)
            print(f"Warning: Using fallback components due to error: {str(e)}")

        try:
            import asyncio
            # Try to get or create an event loop for the current thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Get Milvus connection parameters from environment variables
            milvus_host = os.getenv("MILVUS_HOST", "localhost")
            milvus_port = os.getenv("MILVUS_PORT", "19530")
            
            # Now try to connect to Milvus
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=self.config.collection_name,
                connection_args={"host": milvus_host, "port": milvus_port},
                drop_old=False,
            )
        except Exception as e:
            st.warning(f"Milvus connection failed: {str(e)}. Using FAISS as fallback.")
            from langchain_core.documents import Document
            docs = [Document(page_content="Using FAISS as fallback vector store. Milvus connection failed.")]
            vectorstore = FAISS.from_documents(docs, embeddings)
        
        return vectorstore, llm
    
    def _create_chain(self):
        """Create the QA chain"""
        retriever = self.vectorstore.as_retriever()
        
        prompt = PromptTemplate(
            template="""
            Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
            Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Provide readable text and avoid using code blocks.
            <context>
            {context}
            </context>

            <question>
            {question}
            </question>

            The response should be specific and use statistics or numbers when possible.

            Assistant:""", 
            input_variables=["context", "question"]
        )
        
        format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)
        
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def answer(self, question: str) -> str:
        """Get answer for a question"""
        try:
            return self.qa_chain.invoke({"question": question})
        except Exception as e:
            if "Connection refused" in str(e) and self.config.model_provider == "ollama":
                return f"Error: Ollama server not available. Please check if the server is running."
            raise e

class UI:
    """Streamlit UI for the QA system"""
    
    def __init__(self, config: Optional[QAConfig] = None):
        self._setup()
        self.config = config or self._get_config_from_ui()
        self.qa = QASystem(self.config)
        
    def _on_provider_change(self):
        """Enable the Apply button when provider changes"""
        st.session_state.button_enabled = True
        
    def _setup(self):
        """Configure the page"""
        st.set_page_config(page_title="AI QA System", page_icon="🤖", layout="centered")
        st.markdown("### Welcome to the Milvus Docs Answering System")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize button state to disabled on first load
        if 'button_enabled' not in st.session_state:
            st.session_state.button_enabled = False
            
        if 'last_provider' not in st.session_state:
            st.session_state.last_provider = None
    
    def _get_config_from_ui(self) -> QAConfig:
        """Get configuration from UI inputs"""
        global GLOBAL_CONFIG
        
        # Use initial config from session state if available
        initial_config = st.session_state.get("initial_config", GLOBAL_CONFIG)
        
        with st.sidebar:
            st.header("Model Configuration")
            
            # Use initial config for default values
            default_index = 0 if initial_config.get("provider", GLOBAL_CONFIG["model_provider"]) == "huggingface" else 1
            
            provider_value = st.selectbox(
                "Select Model Provider",
                options=[p.value for p in ModelProvider],
                index=default_index
            )
            
            # Update global config when UI changes
            GLOBAL_CONFIG["model_provider"] = provider_value
            
            # Use appropriate model based on provider
            if provider_value == "huggingface":
                default_model = initial_config.get("model", GLOBAL_CONFIG["model"]) if provider_value == initial_config.get("provider", provider_value) else "mistralai/Mixtral-8x7B-Instruct-v0.1"
                model = st.text_input("HuggingFace Model", value=default_model)
                GLOBAL_CONFIG["model"] = model
                GLOBAL_CONFIG["collection_name"] = os.getenv("MILVUS_HF_COLLECTION_NAME", "demo_collection")

            else:
                default_model = initial_config.get("model", GLOBAL_CONFIG["model"]) if provider_value == initial_config.get("provider", provider_value) else "llama2"
                model = st.text_input("Ollama Model", value=default_model)
                GLOBAL_CONFIG["model"] = model
                GLOBAL_CONFIG["collection_name"] = os.getenv("MILVUS_OLLAMA_COLLECTION_NAME", "demo_collection")
                
            return QAConfig(model_provider=provider_value, model=model)
    
    def run(self):
        """Run the UI"""
        global GLOBAL_CONFIG
        
        # Display model change confirmation if existson if exists
        if "model_changed" in st.session_state:
            st.success(st.session_state["model_changed"])
            if not st.session_state.get("message_displayed"):
                st.session_state["message_displayed"] = True
            else:
                del st.session_state["model_changed"]
                del st.session_state["message_displayed"]
                
        # Add model provider selector in sidebar
        with st.sidebar:
            st.header("Model Settings")
            
            # Use initial config from session state if available
            initial_config = st.session_state.get("initial_config", {})
            provider_from_init = initial_config.get("provider", GLOBAL_CONFIG["model_provider"])
            
            provider_index = 0 if provider_from_init == "huggingface" else 1
            # Store current provider to detect changes
            current_provider = GLOBAL_CONFIG["model_provider"]
            
            # Check if this is the first load
            if st.session_state.last_provider is None:
                st.session_state.last_provider = current_provider
            
            # Use a callback to enable the button when provider changes
            provider_value = st.selectbox(
                "Select Model Provider",
                options=[p.value for p in ModelProvider],
                index=provider_index,
                key="model_provider_select",
                on_change=self._on_provider_change
            )
            
            # Update global config when provider changes
            if GLOBAL_CONFIG["model_provider"] != provider_value:
                GLOBAL_CONFIG["model_provider"] = provider_value
                # Set default model for the selected provider
                if provider_value == "huggingface":
                    GLOBAL_CONFIG["model"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                else:
                    GLOBAL_CONFIG["model"] = "llama2"
            
            if provider_value == "huggingface":
                model = st.text_input("HuggingFace Model", 
                                    value=GLOBAL_CONFIG["model"],
                                    key="hf_model_input")
                
                # Apply Changes button - disabled initially, enabled on changes
                if st.button("Apply Changes", key="apply_hf", disabled=not st.session_state.button_enabled):
                    GLOBAL_CONFIG["model"] = model
                    st.session_state["model_changed"] = f"Using HuggingFace model: {model}"
                    st.query_params["model_provider"] = "huggingface"
                    st.query_params["model"] = model
                    # Disable button after click
                    st.session_state.button_enabled = False
                    # Update last provider
                    st.session_state.last_provider = provider_value
                    st.rerun()
            else:
                model = st.text_input("Ollama Model", 
                                    value=GLOBAL_CONFIG["model"],
                                    key="model_input")
                
                # Apply Changes button - disabled initially, enabled on changes
                if st.button("Apply Changes", key="apply_ollama", disabled=not st.session_state.button_enabled):
                    GLOBAL_CONFIG["model"] = model
                    st.session_state["model_changed"] = f"Using Ollama model: {model}"
                    st.query_params["model_provider"] = "ollama"
                    st.query_params["model"] = model
                    # Disable button after click
                    st.session_state.button_enabled = False
                    # Update last provider
                    st.session_state.last_provider = provider_value
                    st.rerun()
            
        question = st.text_input("Enter your question:", placeholder="Type your question here...")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.button("Get Answer", type="primary")        
        if submit and question.strip():
            with st.spinner("Generating answer..."):
                try:
                    # Create a new QA system with the current global config
                    config = QAConfig(
                        model_provider=GLOBAL_CONFIG["model_provider"],
                        model=GLOBAL_CONFIG["model"]
                    )
                    qa_system = QASystem(config)
                    response = qa_system.answer(question)
                    st.session_state.chat_history.append({"question": question, "answer": response})
                    if response.startswith("Error: Ollama server not available"):
                        st.error(response)
                except Exception as e:
                    if "Connection refused" in str(e) and GLOBAL_CONFIG["model_provider"] == "ollama":
                        error_msg = f"Error: Ollama server not available. Please check if the server is running."
                        st.error(error_msg)
                    else:
                        st.error(f"Error: {str(e)}")
        elif submit:
            st.warning("⚠️ Please enter a question first.")
            
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Chat History")
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.markdown("---")
        
        st.markdown("---\n*Powered by LangChain and Milvus*")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set global config from command line args
    global GLOBAL_CONFIG
    
    # Button state is initialized in UI._setup()
    
    # Initialize from environment variables first
    model_provider_env = os.getenv("MODEL_PROVIDER")
    model_env = os.getenv("MODEL_HF")
    
    if model_provider_env:
        GLOBAL_CONFIG["model_provider"] = model_provider_env
    else:
        GLOBAL_CONFIG["model_provider"] = os.getenv("MILVUS_HF_COLLECTION_NAME", "demo_collection")
    
    if model_env:
        GLOBAL_CONFIG["model"] = model_env
    else:
        GLOBAL_CONFIG["model"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Command line args override environment variables
    if args.model_provider:
        GLOBAL_CONFIG["model_provider"] = args.model_provider
        
    if args.model:
        GLOBAL_CONFIG["model"] = args.model
    elif args.model_provider:
        # Set appropriate default model if only provider is specified
        GLOBAL_CONFIG["model"] = "llama2" if args.model_provider == "ollama" else "mistralai/Mixtral-8x7B-Instruct-v0.1"
        
    # Store initial config in session state for persistence
    if 'streamlit.runtime.scriptrunner' in sys.modules:
        st.session_state["initial_config"] = {
            "provider": GLOBAL_CONFIG["model_provider"],
            "model": GLOBAL_CONFIG["model"]
        }
    
    # Create config based on global config
    config = QAConfig(
        model_provider=GLOBAL_CONFIG["model_provider"],
        model=GLOBAL_CONFIG["model"]
    )
    
    # Initialize and run UI
    ui = UI(config)
    ui.run()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Milvus QA System")
    parser.add_argument(
        "--model-provider", 
        type=str, 
        choices=[p.value for p in ModelProvider], 
        default=None,
        help="Model provider to use"
    )
    parser.add_argument(
        "--model", 
        type=str,
        default=None, 
        help="Model name to use"
    )
    # When running with streamlit, we need to parse only the args after --
    if 'streamlit.runtime.scriptrunner' in sys.modules:
        if '--' in sys.argv:
            idx = sys.argv.index('--')
            args = parser.parse_args(sys.argv[idx+1:])
        else:
            args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()