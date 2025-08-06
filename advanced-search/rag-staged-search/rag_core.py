from typing import List, Dict, Tuple, Optional
import time
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from core import EmbeddingProvider, get_client

class RAGCore:
    def __init__(self, llm, collection_name: str):
        self.llm = llm
        self.collection_name = collection_name
        self.response_cache = {}
        self.classification_cache = {}
        self._setup_chains()
    
    def _setup_chains(self):
        """Initialize LangChain components lazily"""
        # Create templates but don't build chains until needed
        self._classification_template = PromptTemplate(
            input_variables=["question"],
            template="""Milvus question? YES/NO
{question}
Answer:"""
        )
        
        self._rag_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Context: {context}

                Question: {question}

                Answer:"""
        )
        
        self._direct_template = PromptTemplate(
            input_variables=["question", "history"],
            template="""Previous conversation:
{history}

Question: {question}

If this is a greeting (hello, hi, etc.), respond with a friendly greeting and offer to help with Milvus questions.
If asked to resume or summarize the conversation, list all the topics we discussed in detail.
Otherwise, answer the question directly.

Answer:"""
        )
        
        # Initialize chains as None - build on first use
        self.classification_chain = None
        self.rag_chain = None
        self.direct_chain = None
    
    def needs_retrieval(self, question: str, chat_history: List[Dict]) -> bool:
        """Check if question needs document retrieval"""
        start_time = time.time()
        
        # Expanded keyword-based pre-filter
        no_docs_patterns = ['hello', 'hi', 'thanks', 'thank you', 'bye', 'goodbye', 'how are you', 'good morning', 'good afternoon', 'good evening', 'resume', 'conversation', 'chat history', 'what did we discuss', 'continue', 'summarize', 'weather', 'temperature', 'rain', 'sunny', 'cloudy']
        docs_patterns = ['milvus', 'vector', 'database', 'collection', 'index', 'search', 'embedding', 'insert', 'query', 'schema']
        
        question_lower = question.lower()
        
        # Skip LLM if clearly greeting/social
        if any(pattern in question_lower for pattern in no_docs_patterns):
            print(f"DEBUG - Quick filter: NO DOCS (took {time.time() - start_time:.2f}s)")
            return False
            
        # Skip LLM if clearly technical
        if any(pattern in question_lower for pattern in docs_patterns):
            print(f"DEBUG - Quick filter: NEEDS DOCS (took {time.time() - start_time:.2f}s)")
            return True
        
        # Check classification cache
        cache_key = question.lower().strip()
        if cache_key in self.classification_cache:
            result = self.classification_cache[cache_key]
            print(f"DEBUG - Classification cache hit: {'NEEDS DOCS' if result else 'NO DOCS'} (took {time.time() - start_time:.2f}s)")
            return result
        
        # Build classification chain on first use
        if self.classification_chain is None:
            self.classification_chain = self._classification_template | self.llm | StrOutputParser()
        
        # Simplified LLM classification with minimal context
        llm_result = self.classification_chain.invoke({
            "question": question
        })
        
        needs_docs = "YES" in llm_result.upper()
        self.classification_cache[cache_key] = needs_docs
        print(f"DEBUG - Classification: {'NEEDS DOCS' if needs_docs else 'NO DOCS'} (took {time.time() - start_time:.2f}s)")
        return needs_docs
    
    def _retrieve_documents(self, question: str) -> Tuple[str, int]:
        """Retrieve relevant documents from Milvus"""
        start_time = time.time()
        client = get_client()
        
        # Time embedding generation
        embed_start = time.time()
        embedding = EmbeddingProvider.embed_text(question, provider="ollama")
        print(f"DEBUG - Embedding took {time.time() - embed_start:.2f}s")
        
        # Time search
        search_start = time.time()
        search_results = client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=3,  # Reduced from 5 for faster search
            search_params={"metric_type": "COSINE", "params": {"ef": 32}},  # Reduced ef from 64
            output_fields=["text"]
        )
        print(f"DEBUG - Search took {time.time() - search_start:.2f}s")
        
        if not search_results or not search_results[0]:
            return "", 0
        
        context = "\n\n".join([res["entity"]["text"] for res in search_results[0]])
        print(f"DEBUG - Total retrieval took {time.time() - start_time:.2f}s")
        return context, len(search_results[0])
    
    def rag_query_with_retrieval(self, question: str, chat_history: List[Dict]) -> Tuple[str, int]:
        """Full RAG with document retrieval"""
        context, doc_count = self._retrieve_documents(question)
        
        # Build RAG chain on first use
        if self.rag_chain is None:
            self.rag_chain = self._rag_template | self.llm | StrOutputParser()
        
        # Time LLM response
        llm_start = time.time()
        response = self.rag_chain.invoke({
            "context": context,
            "question": question
        })
        print(f"DEBUG - LLM response took {time.time() - llm_start:.2f}s")
        
        return response, doc_count
    
    def direct_response(self, question: str, chat_history: List[Dict]) -> str:
        """Handle questions without retrieval - optimized for speed"""
        start_time = time.time()
        
        # Build direct chain on first use
        if self.direct_chain is None:
            self.direct_chain = self._direct_template | self.llm | StrOutputParser()
        
        # Only include history for explicit resume/summary requests AND if history exists
        resume_keywords = ['resume', 'summarize', 'summary', 'conversation', 'discuss', 'talked about']
        needs_history = any(keyword in question.lower() for keyword in resume_keywords) and len(chat_history) > 0
        
        if needs_history:
            history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in chat_history[-5:]])
            print(f"DEBUG - Including history for summary request: {len(chat_history)} items")
            print(f"DEBUG - History text length: {len(history_text)} chars")
        else:
            history_text = ""
            print(f"DEBUG - No history included for direct question")
        
        response = self.direct_chain.invoke({
            "question": question,
            "history": history_text
        })
        
        print(f"DEBUG - Direct response took {time.time() - start_time:.2f}s")
        return response
    
    def query(self, question: str, chat_history: List[Dict]) -> Tuple[str, int]:
        """Main query method with two-stage approach"""
        total_start = time.time()
        
        # Check cache first - before any LLM calls
        cache_key = question.lower().strip()
        if cache_key in self.response_cache:
            print(f"DEBUG - Cache hit! (took {time.time() - total_start:.2f}s)")
            return self.response_cache[cache_key]
        
        if self.needs_retrieval(question, chat_history):
            print("DEBUG - Using RAG with retrieval")
            result = self.rag_query_with_retrieval(question, chat_history)
        else:
            print("DEBUG - Using direct response")
            result = self.direct_response(question, chat_history), 0
        
        # Cache the result
        self.response_cache[cache_key] = result
        print(f"DEBUG - Total query time: {time.time() - total_start:.2f}s")
        return result

# Backward compatibility functions
def optimized_rag_query(client, llm, collection_name: str, question: str, chat_history: List[Dict]) -> Tuple[str, int]:
    """Legacy interface for backward compatibility"""
    rag_core = RAGCore(llm, collection_name)
    return rag_core.query(question, chat_history)