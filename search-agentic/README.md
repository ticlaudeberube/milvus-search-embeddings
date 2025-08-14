# Classification-Driven RAG Pipeline

**LangChain ReAct Agents Architecture** for intelligent document retrieval and response generation.

## Components

### LangChain ReAct Agents
- **ClassificationAgent**: ReAct agent with classification tools
- **RetrievalAgent**: ReAct agent with vector search tools
- **ResponseAgent**: ReAct agent with response generation tools

### Agent Tools
- **classify_query**: Query classification tool
- **search_documents**: Vector database search tool
- **generate_rag_answer**: Context-based response tool
- **generate_direct_answer**: Direct response tool

### Orchestrator
- **RAGOrchestrator**: Coordinates LangChain ReAct agents
rrr
## Usage

```bash
# Run the agentic RAG application
streamlit run agentic_rag_app.py
```

## Features

- ✅ **LangChain ReAct agents** with reasoning and action loops
- ✅ **Tool-based agent interactions** with hub.pull("hwchase17/react")
- ✅ **Agent reasoning** with thought-action-observation cycles
- ✅ **Semantic search tools** integrated with Milvus vector database
- ✅ **Response caching** for performance optimization
- ✅ **Agent coordination** through orchestrator pattern

## Environment Variables

```bash
OLLAMA_COLLECTION_NAME=your_collection
OLLAMA_LLM_MODEL=llama3.2:1b
```

## Agent Workflow

1. **Query Input** → Classification Agent
2. **Classification** → Route to Retrieval or Direct Response
3. **Retrieval** (if needed) → Vector search via Retrieval Agent
4. **Response Generation** → Response Agent with context
5. **Output** → Formatted response with metadata