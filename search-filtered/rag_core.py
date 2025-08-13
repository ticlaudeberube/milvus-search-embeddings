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
            template="""Does this question require retrieving Milvus documentation to answer properly?

            Question: {question}

            Answer YES ONLY if the question explicitly asks about:
            - Milvus features, capabilities, or functionality
            - Milvus technical concepts, architecture, or how Milvus works
            - Milvus usage instructions, tutorials, or guides
            - Specific Milvus technical details or explanations
            - Vector databases, vector search, or embedding concepts
            - How vectors work, vector retrieval, or context data retrieval
            - Database indexing, similarity search, or vector operations

            Answer NO if the question is:
            - A greeting, thanks, or casual conversation
            - About resuming, continuing, or summarizing conversation
            - Conversation management ("resume our conversation", "continue", "summarize")
            - Vague follow-ups using pronouns ("it", "its", "them", "that") without explicit Milvus context
            - Questions like "Tell me more about its features" or "What about that?"
            - A simple yes/no that doesn't need documentation
            - About weather, health, sports, news, entertainment, or any non-technical topics
            - About topics completely unrelated to databases, vectors, or data storage
            - Vague questions like "anything else" without technical context

            Answer (YES/NO):"""
        )
        
        self._rag_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Milvus database expert. Answer questions about Milvus using the provided context.

Context: {context}

Question: {question}

Provide a comprehensive answer based on the context. Include relevant details, examples, and explanations to fully address the question. If the question is about Milvus, use the context to give a complete and informative response.

Answer:"""
        )
                    
        self._direct_template = PromptTemplate(
            input_variables=["question", "history"],
            template="""Previous conversation:
{history}

Question: {question}

If asked to resume, continue, or summarize the conversation, respond with:
"Here's what we discussed: [list each topic from the conversation history above]"

Otherwise:
- If greeting: respond with friendly greeting and offer to help with Milvus questions
- If about Milvus: answer directly
- If NOT about Milvus: "I'm specialized in Milvus database questions. Please ask about Milvus features, usage, or technical details."

Answer:"""
        )
        
        # Initialize chains as None - build on first use
        self.classification_chain = None
        self.rag_chain = None
        self.direct_chain = None
    
    def needs_retrieval(self, question: str, chat_history: List[Dict]) -> bool:
        """Check if question needs document retrieval using pattern matching + LLM"""
        start_time = time.time()
        question_lower = question.lower()
        
        # Quick pattern-based filtering for obvious cases
        obvious_no_patterns = [
            'weather', 'temperature', 'rain', 'sunny', 'cloudy',
            'hello', 'hi ', 'thanks', 'thank you', 'bye', 'goodbye',
            'resume', 'conversation', 'summarize', 'continue',
            'my name is', 'from now on', 'always include',
            'tell me more about its', 'what about that', 'anything else'
        ]
        
        if any(pattern in question_lower for pattern in obvious_no_patterns):
            result = False
            self.classification_cache[question_lower.strip()] = result
            print(f"DEBUG - Pattern filter: NO DOCS (took {time.time() - start_time:.2f}s)")
            return result
        
        # Check classification cache
        cache_key = question_lower.strip()
        if cache_key in self.classification_cache:
            result = self.classification_cache[cache_key]
            print(f"DEBUG - Classification cache hit: {'NEEDS DOCS' if result else 'NO DOCS'} (took {time.time() - start_time:.2f}s)")
            return result
        
        # Build classification chain on first use
        if self.classification_chain is None:
            self.classification_chain = self._classification_template | self.llm | StrOutputParser()
        
        # Use LLM for ambiguous cases
        llm_result = self.classification_chain.invoke({
            "question": question
        })
        
        # LLM classification - trust the LLM decision
        needs_docs = "YES" in llm_result.upper()
        self.classification_cache[cache_key] = needs_docs
        print(f"DEBUG - LLM Classification: {'NEEDS DOCS' if needs_docs else 'NO DOCS'} (took {time.time() - start_time:.2f}s)")
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
        
        # Include history for resume/summary requests if history exists
        resume_keywords = ['resume', 'summarize', 'summary', 'conversation', 'discuss', 'talked about', 'continue']
        needs_history = any(keyword in question.lower() for keyword in resume_keywords) and len(chat_history) > 0
        
        if needs_history:
            # Build chain with history template
            if self.direct_chain is None:
                self.direct_chain = self._direct_template | self.llm | StrOutputParser()
            
            # Optimize history: only questions and truncated answers
            history_items = []
            for h in chat_history[-3:]:  # Reduced from 5 to 3 items
                answer = h['answer'][:100] + "..." if len(h['answer']) > 100 else h['answer']  # Truncate long answers
                history_items.append(f"Q: {h['question']}\nA: {answer}")
            history_text = "\n".join(history_items)
            print(f"DEBUG - Including history for summary request: {len(chat_history)} items")
            print(f"DEBUG - History text length: {len(history_text)} chars")
            
            response = self.direct_chain.invoke({
                "question": question,
                "history": history_text
            })
        else:
            # Use simple template without history for non-summary questions
            simple_template = PromptTemplate(
                input_variables=["question"],
                template="""Question: {question}

                I'm specialized in Milvus database questions only.

                If this is a greeting (hello, hi, etc.), respond with a friendly greeting and offer to help with Milvus questions.
                If this is a personal instruction (like setting preferences), acknowledge it politely.
                If the question is about Milvus, answer it directly.
                If the question is NOT about Milvus, respond: "I'm specialized in Milvus database questions. Please ask about Milvus features, usage, or technical details."

                Answer:"""
            )
            simple_chain = simple_template | self.llm | StrOutputParser()
            print(f"DEBUG - No history included for direct question")
            
            response = simple_chain.invoke({
                "question": question
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