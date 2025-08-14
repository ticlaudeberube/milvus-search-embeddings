from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from typing import List, Dict

class ResponseAgent:
    """LangChain ReAct agent for response generation"""
    
    def __init__(self, llm):
        self.llm = llm
        self.response_cache = {}
        self._setup_agent()

    def _generate_rag_answer(self, input_data: str, chat_history: List[Dict] = None) -> str:
        """Tool function for RAG response generation"""
        try:
            parts = input_data.split("|||")
            if len(parts) < 2:
                return "Error: Invalid input format"
            
            context, question = parts[0], parts[1]
            history_context = "\n".join([f"Q: {h['question']} A: {h['answer']}" for h in (chat_history or [])[-3:]])
            
            prompt = f"""Use the context to answer the question. Follow any personal instructions from previous conversation.
            
            Previous conversation:
            {history_context}
            
            Context: {context}
            Question: {question}
            
            Do not mention collection names or technical implementation details unless specifically asked.
            
            Answer:"""
            
            return self.llm.invoke(prompt)
            
        except Exception as e:
            return f"Error generating RAG response: {str(e)}"
    
    def _generate_direct_answer(self, question: str, chat_history: List[Dict] = None) -> str:
        """Tool function for direct response generation"""
        history_context = "\n".join([f"Q: {h['question']} A: {h['answer']}" for h in (chat_history or [])[-3:]])
        
        prompt = f"""You are a Milvus database assistant. Respond to off-topic questions by redirecting to Milvus topics.
        
        Previous conversation:
        {history_context}
        
        Current question: {question}
        
        If the user previously gave personal instructions (like their name), follow them. Otherwise redirect to Milvus topics.
        
        Do not mention collection names or technical implementation details unless specifically asked.
        
        Response:"""
        
        return self.llm.invoke(prompt)
    
    def _setup_agent(self):
        tools = [
            Tool(
                name="generate_rag_answer",
                description="Generate answer using retrieved context. Input format: 'context|||question'",
                func=self._generate_rag_answer
            ),
            Tool(
                name="generate_direct_answer",
                description="Generate direct answer without context for greetings or general questions.",
                func=self._generate_direct_answer
            )
        ]
        
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=False, max_iterations=3, max_execution_time=15)
    
    def generate_rag_response(self, question: str, context: str, chat_history: List[Dict] = None) -> str:
        """Generate response using retrieved context"""
        cache_key = f"rag_{question.lower().strip()}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Use direct method to include name
        response = self._generate_rag_answer(f"{context}|||{question}", chat_history)
        self.response_cache[cache_key] = response
        return response
    
    def generate_direct_response(self, question: str, chat_history: List[Dict]) -> str:
        """Generate direct response without retrieval"""
        cache_key = f"direct_{question.lower().strip()}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Use direct tool call to avoid agent issues
        response = self._generate_direct_answer(question, chat_history)
        self.response_cache[cache_key] = response
        return response