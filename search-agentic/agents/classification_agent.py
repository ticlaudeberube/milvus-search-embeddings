from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from typing import Dict, List

class ClassificationAgent:
    """LangChain ReAct agent for query classification"""
    
    def __init__(self, llm):
        self.llm = llm
        self.cache = {}
        self._setup_agent()
    
    def _classify_query(self, question: str) -> str:
        """Tool function for classification"""
        question_lower = question.lower()
        
        # Milvus-specific patterns (YES)
        milvus_patterns = ['milvus', 'vector', 'database', 'collection', 'search', 'retrieve', 'embedding', 'index']
        
        # Non-Milvus patterns (NO)
        off_topic_patterns = ['hello', 'hi', 'thanks', 'bye', 'resume', 'conversation', 'another subject', 'different topic', 'interdependence', 'philosophy', 'weather', 'politics']
        
        if any(p in question_lower for p in milvus_patterns):
            return "YES - Milvus technical query"
        if any(p in question_lower for p in off_topic_patterns):
            return "NO - off-topic or social"
        
        # Default to NO for uncertain cases
        return "NO - not clearly Milvus-related"
    
    def _setup_agent(self):
        tools = [
            Tool(
                name="classify_query",
                description="Classify if a question needs Milvus documentation retrieval. Returns YES/NO with reasoning.",
                func=self._classify_query
            )
        ]
        
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=False, max_iterations=3, max_execution_time=10)
    
    def classify(self, question: str, chat_history: List[Dict]) -> bool:
        """Classify using direct pattern matching"""
        # Clear cache to avoid stale results
        cache_key = question.lower().strip()
        
        # Use direct classification tool
        result = self._classify_query(question)
        needs_docs = "YES" in result
        
        # Debug output
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Classification result: {result}")
        print(f"[DEBUG] Needs docs: {needs_docs}")
        
        self.cache[cache_key] = needs_docs
        return needs_docs