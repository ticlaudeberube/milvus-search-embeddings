# Tests Folder

Comprehensive test suites for the filtered RAG system.

## Test Files

### `test_scenarios.py` - Manual Scenario Testing
**Purpose**: Tests 10 real-world scenarios with actual LLM responses
```bash
python Tests/test_scenarios.py
```
**Features**:
- Tests classification accuracy (YES/NO for document retrieval)
- Measures response times
- Tracks conversation history
- Shows actual LLM responses

### `test_rag_core.py` - Unit Tests
**Purpose**: Unit tests with mocked components
```bash
python Tests/test_rag_core.py
# Or with pytest
pytest Tests/test_rag_core.py -v
```
**Coverage**:
- Classification logic
- Document retrieval
- Direct responses
- Full pipeline testing

### `test_classification.py` - Classification Logic
**Purpose**: Tests the needs_retrieval classification
```bash
python Tests/test_classification.py
```
**Tests**:
- Technical questions (should return YES)
- Greetings/social questions (should return NO)
- Edge cases and boundary conditions

### `test_real_llm.py` - Real LLM Testing
**Purpose**: Tests with actual Ollama LLM (no mocks)
```bash
python Tests/test_real_llm.py
```
**Requirements**:
- Ollama running locally
- `OLLAMA_LLM_MODEL` environment variable set
- Shows raw LLM responses vs final classification

### `test_e2e_streamlit.py` - End-to-End GUI Testing
**Purpose**: Tests Streamlit GUI components
```bash
python Tests/test_e2e_streamlit.py
```
**Validates**:
- App imports and initialization
- RAG system setup
- Session state management
- Provides manual testing scenarios

## Running All Tests

```bash
# Quick test suite (mocked)
pytest Tests/test_rag_core.py Tests/test_classification.py -v

# Full test suite (requires Ollama)
python Tests/test_scenarios.py
python Tests/test_real_llm.py

# GUI validation
python Tests/test_e2e_streamlit.py
```

## Test Requirements

**For mocked tests** (test_rag_core.py, test_classification.py):
- No external dependencies
- Fast execution
- Isolated unit testing

**For real LLM tests** (test_scenarios.py, test_real_llm.py):
- Ollama running locally
- Environment variables set:
  - `OLLAMA_COLLECTION_NAME`
  - `OLLAMA_LLM_MODEL`
- Milvus collection with data

**For GUI tests** (test_e2e_streamlit.py):
- Streamlit installed
- All RAG dependencies available

## Expected Test Results

**Classification Accuracy**:
- Technical questions: 90%+ classified as YES
- Social/greeting questions: 95%+ classified as NO
- Response time: <2s per question

**Performance Metrics**:
- 43% faster response times vs full RAG
- 50% reduction in unnecessary retrievals
- Cache hit rate: 80%+ for repeated questions

## Test Scenarios

The test suite validates these scenarios:

1. "How is data stored in milvus?" → YES (retrieves documents)
2. "My name is Claude. From now on always include my name in the answer." → NO (no retrieval)
3. "What is Milvus?" → YES (retrieves documents)
4. "How is data stored in milvus?" → NO (use history)
5. "What is the weather today?" → NO (off-topic redirect)
6. "How vectors are used to retrieve context data?" → YES (retrieves documents)
7. "Tell me more about its features?" → NO (should request more context)
8. "Hello there" → NO (direct greeting response)
9. "How does Milvus indexing work?" → YES (retrieves documents)
10. "Can you resume our conversation?" → NO (should resume history)