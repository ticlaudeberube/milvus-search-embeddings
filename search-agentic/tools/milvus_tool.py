from typing import List, Dict, Any
from core import get_client, EmbeddingProvider

class MilvusTool:
    """Tool for Milvus vector database operations"""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = get_client()
    
    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        embedding = EmbeddingProvider.embed_text(query, provider="ollama")
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            search_params={"metric_type": "COSINE", "params": {"ef": 32}},
            output_fields=["text"]
        )
        
        if not results or not results[0]:
            return []
        
        return [{"text": res["entity"]["text"], "score": res["distance"]} 
                for res in results[0]]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            self.client.load_collection(self.collection_name)
            return {"status": "loaded", "collection": self.collection_name}
        except Exception as e:
            return {"status": "error", "message": str(e)}