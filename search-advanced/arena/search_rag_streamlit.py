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
from provider import model_registry

# Set verbose to False to avoid unnecessary output
set_verbose(False)
# ************************************************************
# THIS WAS CODE NEEDS TO BE REFACTORED
# Remove specific model provider in favor of a more generic approach
# ************************************************************


# Global variables for model configuration
GLOBAL_CONFIG = {
    "model_provider": model_registry.OLLAMA.value,  # Default to Ollama provider
    "model": "llama2",  # Default Ollama model
    "collection": "milvus_ollama_collection"
}

@dataclass
class QAConfig:
    """Configuration for the QA system"""
    model_provider: str = GLOBAL_CONFIG["model_provider"]
    model: str = ""
    embedding_model: str = ""
    collection_name: str = ""
    max_tokens: int = 512
    
    def __post_init__(self):
        """Get all configuration from the provider config"""

        
        # Get provider config
        provider_config = model_registry.get_provider(self.model_provider)
        
        # Get all configuration from provider config
        self.model = provider_config.default_model
        self.embedding_model = provider_config.embedding_model
        self.collection_name = provider_config.collection_name
        self.max_tokens = provider_config.max_tokens

class QASystem:
    """Milvus-based Question Answering System"""
    
    def __init__(self, config: QAConfig):
        self.config = config
        self.vectorstore, self.llm = self._setup_components()
        self.qa_chain = self._create_chain()
    
    @staticmethod
    def _get_embeddings(provider: str, model_name: str):
        """Get embeddings model with caching"""
        if provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        else:  # Ollama
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=model_name)
    
    def _setup_components(self) -> Tuple[Any, LLM]:
        """Initialize LLM and vector store"""
        
        # Get provider config for additional parameters
        provider_config = model_registry.get_provider(self.config.model_provider)
        additional_params = provider_config.additional_params or {}
        
        # Get model-specific and Milvus-specific parameters
        model_params = additional_params.get("model_params", {})
        milvus_params = additional_params.get("milvus_params", {})
        
        # Initialize LLM based on provider
        try:
            if self.config.model_provider == "huggingface":
                # Import HuggingFace modules only when needed
                try:
                    from langchain_huggingface import HuggingFaceEndpoint
                    
                    # Ensure task is explicitly set to override any defaults
                    params = model_params.copy()
                    params['task'] = 'conversational'
                    
                    llm = HuggingFaceEndpoint(
                        repo_id=self.config.model,
                        max_new_tokens=self.config.max_tokens,
                        **params
                    )
                except ImportError:
                    from langchain_community.llms import FakeListLLM
                    llm = FakeListLLM(responses=["Sorry, HuggingFace module could not be loaded."])
                    
                # Use cached embeddings for better performance
                try:
                    # Use a model with consistent 768-dimensional embeddings
                    embeddings = self._get_embeddings("huggingface", "sentence-transformers/all-mpnet-base-v2")
                except:
                    embeddings = FakeEmbeddings(size=768)
                
            else:  # Ollama
                # Import Ollama modules only when needed
                try:
                    from langchain_ollama.llms import OllamaLLM
                    
                    # Get base_url from model params
                    base_url = model_params.get("base_url", "http://localhost:11434")
                    
                    llm = OllamaLLM(
                        model=self.config.model,
                        num_predict=self.config.max_tokens,
                        base_url=base_url
                    )
                    # Use cached embeddings for better performance
                    embeddings = self._get_embeddings("ollama", os.getenv("HF_LLM_MODEL", ''))
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
                
            # Get Milvus connection parameters from config
            milvus_host = milvus_params.get("host", "localhost")
            milvus_port = milvus_params.get("port", "19530")
            
            # Use provider-specific collection name to avoid dimension conflicts
            provider_collection = f"{self.config.collection_name}_{self.config.model_provider}"
            
            # Now try to connect to Milvus
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=provider_collection,
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
    
    def _handle_dimension_mismatch(self):
        """Handle vector dimension mismatch by recreating the collection"""
        
        # Get provider config
        provider_config = model_registry.get_provider(self.config.model_provider)
        additional_params = provider_config.additional_params or {}
        
        # Get Milvus-specific parameters
        milvus_params = additional_params.get("milvus_params", {})
        
        # Get embedding model based on provider using cached method
        if self.config.model_provider == "huggingface":
            embeddings = self._get_embeddings("huggingface", os.getenv("HF_LLM_MODEL", ''))
        else:  # Ollama
            embeddings = self._get_embeddings("ollama", os.getenv("OLLAMA_LLM_MODEL", ''))
        
        # Get Milvus connection parameters
        milvus_host = milvus_params.get("host", "localhost")
        milvus_port = milvus_params.get("port", "19530")
        
        # Use provider-specific collection name
        provider_collection = f"{self.config.collection_name}_{self.config.model_provider}"
        
        # Recreate the collection with the correct dimensions
        st.warning(f"Vector dimension mismatch detected. Recreating collection {provider_collection}...")
        
        # Create new vectorstore with drop_old=True to force recreation
        from langchain_milvus import Milvus
        self.vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=provider_collection,
            connection_args={"host": milvus_host, "port": milvus_port},
            drop_old=True,  # Force recreation
        )
        
        # Recreate the chain
        self.qa_chain = self._create_chain()
    
    def answer(self, question: str) -> str:
        """Get answer for a question"""
        try:
            # Check if search should be aborted
            if st.session_state.get("abort_search", False):
                # Reset the flag
                st.session_state["abort_search"] = False
                return "Search aborted due to model provider change."
                
            # Pass the question directly as a string, not as a dictionary
            return self.qa_chain.invoke(question)
        except Exception as e:
            # Check again if search was aborted during processing
            if st.session_state.get("abort_search", False):
                # Reset the flag
                st.session_state["abort_search"] = False
                return "Search aborted due to model provider change."
                
            if "Connection refused" in str(e) and self.config.model_provider == "ollama":
                return f"Error: Ollama server not available. Please check if the server is running."
            elif "vector dimension mismatch" in str(e):
                # Handle dimension mismatch by recreating the collection
                self._handle_dimension_mismatch()
                # Try again with the recreated collection
                return self.qa_chain.invoke(question)
            raise e

class UI:
    """Streamlit UI for the QA system"""
    
    def __init__(self, config: Optional[QAConfig] = None):
        self._setup()
        self.config = config or self._get_config_from_ui()
        self.qa = QASystem(self.config)
        
    def _on_provider_change(self):
        """Enable the Apply button when provider changes and abort any ongoing search"""
        st.session_state.button_enabled = True
        
        # Set flag to abort any ongoing search
        st.session_state["abort_search"] = True
        
        # Clear any existing QA system to force recreation
        if 'qa_system' in st.session_state:
            del st.session_state['qa_system']
        
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
            
            # Get provider options
            provider_options = [p.value for p in ModelProvider]
            
            # Use initial config for default values, defaulting to first provider if not set
            current_provider = initial_config.get("provider", GLOBAL_CONFIG.get("model_provider", provider_options[0]))
            default_index = provider_options.index(current_provider) if current_provider in provider_options else 0
            
            provider_value = st.selectbox(
                "Select Model Provider",
                options=provider_options,
                index=default_index,
                key="model_provider_config",
                on_change=self._on_provider_change
            )
            
            # Update global config when UI changes
            GLOBAL_CONFIG["model_provider"] = provider_value
            
            # Get provider config
            provider_config = model_registry.get_provider(provider_value)
            
            # Get model and collection from provider config
            model = provider_config.default_model
            collection = provider_config.collection_name
            
            print(f"UI: Using model '{model}' from provider '{provider_value}' with collection '{collection}'")
            
            # Update global config with model and collection from provider
            GLOBAL_CONFIG["model"] = model
            GLOBAL_CONFIG["collection"] = collection
            
            # Display current model and collection (read-only)
            st.text(f"Current {provider_value} Model: {model}")
            st.text(f"Collection: {collection}")
            
            # Create config with just the provider - it will get all other settings from provider config
            return QAConfig(model_provider=provider_value)
    
    def run(self):
        """Run the UI"""
        global GLOBAL_CONFIG
        
        # Create a placeholder for messages
        message_placeholder = st.empty()
        
        # Add model provider selector in sidebar
        with st.sidebar:
            st.header("Model Settings")
            
            # Get provider options
            provider_options = [p.value for p in ModelProvider]
            
            # Use initial config from session state if available
            initial_config = st.session_state.get("initial_config", {})
            provider_from_init = initial_config.get("provider", GLOBAL_CONFIG.get("model_provider", provider_options[0]))
            
            # Find the index of the current provider in the options list
            try:
                provider_index = provider_options.index(provider_from_init)
            except ValueError:
                provider_index = 0  # Default to first provider if not found
                
            # Store current provider to detect changes
            current_provider = GLOBAL_CONFIG.get("model_provider", provider_options[0])
            
            # Check if this is the first load
            if st.session_state.last_provider is None:
                st.session_state.last_provider = current_provider
            
            # Provider selection
            provider_value = st.selectbox(
                "Select Model Provider",
                options=provider_options,
                index=provider_index,
                key="model_provider_select"
            )
            
            # Update global config when provider changes
            if GLOBAL_CONFIG.get("model_provider") != provider_value:
                # Set the provider
                GLOBAL_CONFIG["model_provider"] = provider_value
                
                # Get provider config - ensure we're using the correct provider
                provider_config = model_registry.get_provider(provider_value)
                
                # Get model and collection from provider config
                model = provider_config.default_model
                collection = provider_config.collection_name
                
                print(f"Run: Using model '{model}' from provider '{provider_value}' with collection '{collection}'")
                
                # Update global config with model and collection from provider
                GLOBAL_CONFIG["model"] = model
                GLOBAL_CONFIG["collection"] = collection
                
                # Show confirmation message
                message_placeholder.success(f"Using {provider_value} provider with model: {model}")
                
                # Update last provider
                st.session_state.last_provider = provider_value
                
                # Update session state
                st.session_state["initial_config"] = {
                    "provider": provider_value
                }
                
                # Flag that we need to create a new QA system on next question
                st.session_state["need_new_qa_system"] = True
            
            # Display current model and collection (read-only)
            try:
                provider_config = model_registry.get_provider(provider_value)
                model = provider_config.default_model
                collection = provider_config.collection_name
                st.text(f"Current model: {model}")
                st.text(f"Collection: {collection}")
            except:
                st.text(f"Current model: {GLOBAL_CONFIG['model']}")
                st.text(f"Collection: {GLOBAL_CONFIG.get('collection', 'unknown')}")
            
        # Error message area
        error_placeholder = st.empty()
        
        question = st.text_input("Enter your question:", placeholder="Type your question here...")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.button("Get Answer", type="primary")        
            
        if submit and question.strip():
            # Create a progress placeholder that can be updated
            progress_placeholder = st.empty()
            
            with st.spinner("Generating answer..."):
                try:
                    # Check if we need to create a new QA system
                    need_new_system = 'qa_system' not in st.session_state or st.session_state.get("need_new_qa_system", False)
                    
                    if need_new_system:
                        # Create config with just the provider - it will get the model from provider config
                        config = QAConfig(
                            model_provider=GLOBAL_CONFIG["model_provider"]
                        )
                        st.session_state.qa_system = QASystem(config)
                        # Reset the flag
                        st.session_state["need_new_qa_system"] = False
                    
                    # Get answer from QA system
                    response = st.session_state.qa_system.answer(question)
                    
                    # Check if search was aborted
                    if response == "Search aborted due to model provider change.":
                        error_placeholder.warning("Search aborted: Model provider was changed")
                        return
                        
                    # Add to chat history
                    st.session_state.chat_history.append({"question": question, "answer": response})
                    
                    # Show error if needed
                    if response.startswith("Error: Ollama server not available"):
                        error_placeholder.error(response)
                    
                except Exception as e:
                    # Check if search was aborted
                    if st.session_state.get("abort_search", False):
                        st.session_state["abort_search"] = False
                        error_placeholder.warning("Search aborted: Model provider was changed")
                        return
                        
                    if "Connection refused" in str(e) and GLOBAL_CONFIG["model_provider"] == "ollama":
                        error_msg = f"Error: Ollama server not available. Please check if the server is running."
                        error_placeholder.error(error_msg)
                    else:
                        error_placeholder.error(f"Error: {str(e)}")
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

def reset_registry():
    """Reset the model registry to defaults"""
    # Reset to defaults
    model_registry.reset_to_defaults()
    # Delete config file to force reload
    if model_registry.CONFIG_FILE.exists():
        model_registry.CONFIG_FILE.unlink()
    # Reload default providers
    model_registry._load_default_providers()
    return model_registry

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set global config from command line args
    global GLOBAL_CONFIG
    
    # Reset registry to ensure correct defaults
    model_registry = reset_registry()
    
    # Print debug info
    print(model_registry.debug_info())
    
    # Check if we already have a QA system in session state
    if 'qa_system' not in st.session_state:
        # First time initialization
        provider_value = args.model_provider if args.model_provider else ModelProvider.OLLAMA.value
        
        # Command line args have highest priority
        GLOBAL_CONFIG["model_provider"] = provider_value
        
        # Check URL parameters
        query_params = st.query_params
        if "model_provider" in query_params:
            GLOBAL_CONFIG["model_provider"] = query_params["model_provider"]
        
        # Get provider config - ensure we're using the correct provider
        provider_config = model_registry.get_provider(GLOBAL_CONFIG["model_provider"])
        
        # Get model and collection from provider config
        model = provider_config.default_model
        collection = provider_config.collection_name
        
        print(f"Using model '{model}' from provider '{GLOBAL_CONFIG['model_provider']}' with collection '{collection}'")
        
        # Update global config with model and collection from provider
        GLOBAL_CONFIG["model"] = model
        GLOBAL_CONFIG["collection"] = collection
        
        # Store provider in session state for persistence
        st.session_state["initial_config"] = {
            "provider": GLOBAL_CONFIG["model_provider"]
        }
        
        # Create config based on global config
        config = QAConfig(
            model_provider=GLOBAL_CONFIG["model_provider"]
        )
        
        # Initialize UI
        ui = UI(config)
        st.session_state['ui'] = ui
    else:
        # Use existing UI from session state
        ui = st.session_state['ui']
    
    # Run UI
    ui.run()

def parse_args():
    """Parse command line arguments"""
    # Create a simple namespace object
    class Args:
        pass
    
    args = Args()
    args.model_provider = None
    
    # Simple direct parsing for model_provider
    for i, arg in enumerate(sys.argv):
        # Handle --model-provider=value format
        if arg.startswith("--model-provider=") or arg.startswith("--model_provider="):
            args.model_provider = arg.split("=", 1)[1]
            break
        # Handle --model-provider value format
        elif (arg == "--model-provider" or arg == "--model_provider") and i + 1 < len(sys.argv):
            args.model_provider = sys.argv[i + 1]
            break
    
    # Validate model_provider if provided
    if args.model_provider and args.model_provider not in [p.value for p in ModelProvider]:
        print(f"Warning: Invalid model_provider '{args.model_provider}'. Using default.")
        args.model_provider = None
    
    return args

if __name__ == "__main__":
    main()