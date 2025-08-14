from typing import List, Dict, Tuple
from agents.classification_agent import ClassificationAgent
from agents.retrieval_agent import RetrievalAgent
from agents.response_agent import ResponseAgent

class RAGOrchestrator:
    """Main orchestrator that coordinates all LangChain ReAct agents"""
    
    def __init__(self, llm, collection_name: str):
        self.classification_agent = ClassificationAgent(llm)
        self.retrieval_agent = RetrievalAgent(llm, collection_name)
        self.response_agent = ResponseAgent(llm)
    
    def process_query(self, question: str, chat_history: List[Dict]) -> Tuple[str, int]:
        """Process query through the complete RAG pipeline"""
        # Stage 1: Classification
        needs_retrieval = self.classification_agent.classify(question, chat_history)
        
        if needs_retrieval:
            # Stage 2a: Retrieval + RAG Response
            context, doc_count = self.retrieval_agent.retrieve(question)
            response = self.response_agent.generate_rag_response(question, context, chat_history)
            return response, doc_count
        else:
            # Stage 2b: Direct Response
            response = self.response_agent.generate_direct_response(question, chat_history)
            return response, 0