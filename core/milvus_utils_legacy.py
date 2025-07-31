
from sentence_transformers import SentenceTransformer
import os
import torch
import ollama
from typing import Optional, Any
from pymilvus import MilvusClient, Collection, MilvusException, db, utility, model

# Global client instance - initialized lazily
_client: Optional[MilvusClient] = None

class MilvusUtils:
    @staticmethod
    def get_client() -> MilvusClient:
        global _client
        if _client is None:
            try:
                _client = MilvusClient(
                    uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
                    token=os.getenv("MILVUS_TOKEN", "root:Milvus")
                )
                # Test connection
                _client.list_databases()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Milvus: {e}")
        return _client
    @staticmethod
    def create_database(db_name: str | None) -> None:
        if not db_name:
            print("db_name is required")
            return
        try:
            db = MilvusUtils.get_client()
            existing_databases = db.list_databases()
            if db_name in existing_databases:

                # Use the database context
                db.using_database(db_name)

                # Drop all collections in the database
                collections = utility.list_collections()
                for collection_name in collections:
                    collection = Collection(name=collection_name)
                    collection.drop()
                    print(f"Collection '{collection_name}' has been dropped.")

                db.drop_database(db_name)
                print(f"Database '{db_name}' has been deleted.")
            else:
                print(f"Database '{db_name}' does not exist.")
                db.create_database(db_name)
                print(f"Database '{db_name}' created successfully.")
        except MilvusException as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def create_collection(collection_name: str, dimension: int = 1536, metric_type: str = "COSINE", consistency_level: str = "Session") -> None:
        if not collection_name:
            print("collection_name is required")
            return
        try:
            client = MilvusUtils.get_client()
            if client.has_collection(collection_name=collection_name):
                client.drop_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type=metric_type,
                consistency_level=consistency_level
            )
        except MilvusException as e:
            print(f"An error occurred: {e}")
        
    @staticmethod 
    def has_collection(collection: str) -> bool:
        client = MilvusUtils.get_client()
        return client.has_collection(collection_name=collection)

    @staticmethod
    def drop_database(db_name: str) -> None:
        if not db_name:
            print("db_name is required")
            return
        try:
            client = MilvusUtils.get_client()
            client.drop_database(db_name=db_name)
            print("Database dropped successfully")
        except MilvusException as e:
            print(f"An error occurred: {e}")

        

    @staticmethod
    def drop_collection(collection_name: str) -> None:
        if not collection_name:
            print("collection_name is required")
            return
        try:
            client = MilvusUtils.get_client()
            client.drop_collection(collection_name=collection_name)
            print("Collection dropped successfully")
        except MilvusException as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def insert_data(collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:        
        client = MilvusUtils.get_client()
        res = client.insert(collection_name=collection_name, data=data)
        return res
    
    @staticmethod
    def vectorize_documents(collection_name: str, docs: list[str]) -> tuple[dict[str, Any], int]:
        # This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
        embedding_fn = model.DefaultEmbeddingFunction()

        vectors = embedding_fn.encode_documents(docs)
        # The output vector has 768 dimensions, matching the collection that we just created.
        print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)
        dimension = embedding_fn.dim

        MilvusUtils.create_collection(collection_name, dimension=dimension)

        # Each entity has id, vector representation, raw text, and a subject label that we use
        # to demo metadata filtering later.
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]

        print("Data has", len(data), "entities, each with fields: ", data[0].keys())
        print("Vector dim:", len(data[0]["vector"]))

        client = MilvusUtils.get_client()
        res = client.insert(collection_name=collection_name, data=data)

        return res, dimension
    
    @staticmethod
    def embed_text_hf(sentences: list[str], model: Optional[str] = None) -> list[list[float]]:
        """Deprecated: Use embed_text(provider='huggingface') instead"""
        return MilvusUtils.embed_text(sentences, provider='huggingface', model=model)
    
    @staticmethod
    def embed_text(text, provider: str = 'huggingface', model: Optional[str] = None):
        """Unified embedding method supporting multiple providers"""
        if provider == 'huggingface':
            _model = model or os.getenv('HF_EMBEDDING_MODEL')
            if not _model:
                raise ValueError("HF_EMBEDDING_MODEL environment variable not set")
            st = SentenceTransformer(_model)
            text_input = [text] if isinstance(text, str) else text
            embeddings = st.encode(text_input, batch_size=256, show_progress_bar=True)
            return embeddings[0].tolist() if isinstance(text, str) else embeddings.tolist()
        elif provider == 'ollama':
            _model = model or os.getenv('OLLAMA_EMBEDDING_MODEL')
            if not _model:
                raise ValueError("OLLAMA_EMBEDDING_MODEL environment variable not set")
            if isinstance(text, list):
                return [ollama.embeddings(model=_model, prompt=t)["embedding"] for t in text]
            return ollama.embeddings(model=_model, prompt=text)["embedding"]
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def embed_text_ollama(text: str, model: Optional[str] = None) -> list[float]:
        """Deprecated: Use embed_text(provider='ollama') instead"""
        return MilvusUtils.embed_text(text, provider='ollama', model=model)

    @staticmethod
    def get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("WARNING: MPS not available. Falling back to CPU.")
            return torch.device("cpu")