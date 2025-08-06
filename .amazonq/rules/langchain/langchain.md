# LangChain Rules

## Core Principles
- Use LangChain components instead of custom implementations
- Leverage LangChain's built-in integrations for vector stores, embeddings, and LLMs
- Follow LangChain's chain composition patterns
- Use LangChain's memory and callback systems

## Vector Stores
- Use `langchain_milvus.Milvus` instead of custom Milvus client code
- Leverage `from_documents()` and `from_texts()` methods for data ingestion
- Use `similarity_search()` and `similarity_search_with_score()` for retrieval
- Implement `as_retriever()` for chain integration

## Embeddings
- Use `langchain_huggingface.HuggingFaceEmbeddings` for HuggingFace models
- Use `langchain_ollama.OllamaEmbeddings` for Ollama models
- Avoid custom embedding wrapper classes
- Cache embeddings with `CacheBackedEmbeddings` when appropriate

## Document Processing
- Use `langchain_community.document_loaders` for file loading
- Apply `langchain.text_splitter` classes for chunking
- Use `Document` objects with metadata instead of plain text
- Implement `RecursiveCharacterTextSplitter` for text splitting

## Chains and Runnables
- Use `LCEL` (LangChain Expression Language) for chain composition
- Implement `RunnablePassthrough` for data flow control
- Use `RunnableParallel` for concurrent operations
- Prefer `invoke()` over deprecated `run()` methods

## RAG Implementation
- Use `RetrievalQA` or `ConversationalRetrievalChain` for RAG
- Implement `PromptTemplate` for consistent prompting
- Use `StrOutputParser` for response formatting
- Leverage `RunnableWithMessageHistory` for conversation memory

## LLM Integration
- Use `langchain_ollama.ChatOllama` for Ollama chat models
- Use `langchain_huggingface.ChatHuggingFace` for HuggingFace models
- Implement streaming with `stream()` method
- Use `ChatPromptTemplate` for structured prompts

## Memory Management
- Use `ConversationBufferMemory` for basic conversation history
- Implement `ConversationSummaryMemory` for long conversations
- Use `VectorStoreRetrieverMemory` for semantic memory
- Apply `ConversationBufferWindowMemory` for sliding window

## Error Handling
- Use LangChain's built-in retry mechanisms
- Implement `CallbackManager` for monitoring
- Use `LangChainException` base class for custom exceptions
- Apply circuit breaker patterns with `tenacity`

## Configuration
- Use `langchain.globals` for global settings
- Implement environment-based configuration with `langchain.schema`
- Use `BaseSettings` from pydantic for configuration classes
- Apply `SecretStr` for sensitive configuration values

## Testing
- Mock LangChain components with `langchain.schema.BaseLanguageModel`
- Use `FakeEmbeddings` and `FakeLLM` for testing
- Test chains with `langchain.schema.BaseRetriever`
- Implement integration tests with real components

## Performance
- Use `CacheBackedEmbeddings` to avoid re-computing embeddings
- Implement batch processing with `embed_documents()`
- Use async methods (`ainvoke`, `astream`) for concurrent operations
- Apply connection pooling for vector store connections

## Code Organization
- Group LangChain imports: core, community, experimental
- Use factory patterns for component creation
- Implement builder patterns for complex chains
- Apply dependency injection for testability

## Migration Guidelines
- Replace custom vector store clients with LangChain integrations
- Convert custom embedding functions to LangChain embedding classes
- Refactor custom chains to use LCEL syntax
- Update custom document loaders to use LangChain loaders