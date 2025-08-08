"""Test with real LLM to see actual responses"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama.llms import OllamaLLM
from rag_core import RAGCore

def test_real_classification():
    """Test classification with real LLM"""
    
    try:
        llm = OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b"))
        rag_core = RAGCore(llm, "test_collection")
        
        test_questions = [
            "How is data stored in milvus?",
            "What is Milvus?",
            "Hello there",
            "Tell me about vector storage",
            "How does Milvus indexing work?"
        ]
        
        print("=== REAL LLM CLASSIFICATION TEST ===")
        for question in test_questions:
            # Get the raw LLM response
            prompt_vars = {
                "question": question,
                "recent_context": ""
            }
            
            raw_response = rag_core.classification_chain.invoke(prompt_vars)
            result = rag_core.needs_retrieval(question, [])
            
            print(f"Q: '{question}'")
            print(f"   Raw LLM Response: '{raw_response.strip()}'")
            print(f"   Final Classification: {result}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and the model is available")

if __name__ == "__main__":
    test_real_classification()