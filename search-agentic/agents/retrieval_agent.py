from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from typing import Tuple
from tools.milvus_tool import MilvusTool

class RetrievalAgent:
    """LangChain ReAct agent for document retrieval"""
    
    def __init__(self, llm, collection_name: str):
        self.llm = llm
        self.milvus_tool = MilvusTool(collection_name)
        self._setup_agent()
    
    def _search_documents(self, query: str) -> str:
        """Tool function for vector search"""
        try:
            results = self.milvus_tool.search(query, limit=5)
            
            if not results:
                return "No documents found"
            
            docs = [res["text"] for res in results]
            return "\n\n".join(docs)
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _setup_agent(self):
        tools = [
            Tool(
                name="search_documents",
                description="Search Milvus vector database for relevant documents using semantic similarity.",
                func=self._search_documents
            )
        ]
        
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=False, max_iterations=3, max_execution_time=10)
    
    def retrieve(self, question: str) -> Tuple[str, int]:
        """Retrieve using ReAct agent"""
        try:
            result = self.agent_executor.invoke({
                "input": f"Search for documents related to: {question}"
            })
            
            output = result.get("output", "")
            
            # Return context and count based on content
            if "No documents" in output:
                return "", 0
            elif output.strip():
                # Count paragraphs as document count estimate
                doc_count = len([p for p in output.split("\n\n") if p.strip()])
                return output, max(doc_count, 1)
            else:
                return "", 0
            
        except Exception:
            # Fallback to direct search
            context = self._search_documents(question)
            doc_count = len([p for p in context.split("\n\n") if p.strip()]) if context != "No documents found" else 0
            return context, doc_count