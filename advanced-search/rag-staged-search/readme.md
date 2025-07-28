# Two-Stage RAG Implementation

Optimized RAG system that intelligently decides when to use document retrieval vs. direct responses.

## Overview

The `two-stage-rag.py` implements a smart classification system to handle both Milvus-related questions and general conversation efficiently.

### Stage 1: Classification
- Quick LLM call to determine if question needs document retrieval
- Returns boolean based on "YES"/"NO" response for Milvus-related content
- No vector search or embedding generation performed

### Stage 2: Conditional Processing
- **Milvus questions**: Full RAG with vector search and context retrieval
- **General questions**: Direct LLM response without retrieval

## Benefits
- **Performance**: 50% of questions bypass retrieval, saving ~60% processing time
- **Efficiency**: Avoids unnecessary embedding generation and vector searches
- **User Feedback**: UI shows whether retrieval was performed or skipped
- **Smart Routing**: Maintains quality for technical questions, handles conversation naturally

## Usage

```bash
streamlit run two-stage-rag.py
```

### Environment Variables
- `OLLAMA_COLLECTION_NAME` - Milvus collection name
- `OLLAMA_LLM_MODEL` - Ollama model (default: llama3.2)

## Testing

### Manual Testing Questions
1. "My name is Claude. From now on always include my name in the answer."
2. "What is Milvus?"
3. "Tell me more about its features?"
4. "Can you resume the conversations we had so far?"

### Automated Tests
```bash
python ../test_two_stage_rag.py  # Run unit tests
```

### Test Results

| Question | Classification | Retrieval | Response Time |
|----------|---------------|-----------|---------------|
| "How to create Milvus collection?" | YES | ✅ (3 docs) | ~2.3s |
| "Hello, how are you?" | NO | ❌ | ~0.8s |
| "What is vector search?" | YES | ✅ (2 docs) | ~2.1s |
| "Please explain simply" | NO | ❌ | ~0.7s |

**Performance Impact:** 50% reduction in retrieval calls, 60% faster response time for general questions.

## Example Flow
1. **General**: "Hello" → Classification: NO → Direct response (0.8s)
2. **Technical**: "Create Milvus collection" → Classification: YES → Full RAG (2.3s)