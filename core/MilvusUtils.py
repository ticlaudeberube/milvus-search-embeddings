
from sentence_transformers import SentenceTransformer
import os
import torch
import ollama
from typing import Optional, Any
from pymilvus import MilvusClient, Collection, MilvusException, db, utility, model

#Start/install Milvus container before use
client: MilvusClient = MilvusClient( 
    uri="http://localhost:19530",
    token="root:Milvus"
)

database_name: str = os.getenv("MY_DB_NAME") or "default"
try:
    client.use_database(database_name)  # type: ignore
except Exception:
    pass  # Database might not exist or method not available

class MilvusUtils:
    @staticmethod
    def get_client() -> MilvusClient:
        return client
    @staticmethod
    def create_database(db_name: str) -> None:
        try:
            existing_databases = db.list_database()
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
    def create_collection(collection_name: str, dimension: int = 1536, metric_type="COSINE", consistency_level="Session") -> None:
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type=metric_type,
            consistency_level=consistency_level
        )
    @staticmethod 
    def has_collection(collection: str) -> bool:
        return client.has_collection(collection_name=collection)

    @staticmethod
    def drop_collection(collection_name: str) -> None:
        client.drop_collection(collection_name=collection_name)

    @staticmethod
    def insert_data(collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:        
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

        res = client.insert(collection_name=collection_name, data=data)

        # print(res)

        return res, dimension
    
    @staticmethod
    def embed_text_hf(sentences: list[str], model: Optional[str] = None) -> list[list[float]]:
        _model = model or os.getenv('HF_EMBEDDING_MODEL')
        if not _model:
            raise ValueError("HF_EMBEDDING_MODEL environment variable not set")
        st = SentenceTransformer(_model)
        embeddings = st.encode(sentences, batch_size=256, show_progress_bar=True)
        return embeddings.tolist()
    
    @staticmethod
    def embed_text_ollama(text: str, model: Optional[str] = None) -> list[float]:
        _model = model or os.getenv('OLLAMA_EMBEDDING_MODEL')
        if not _model:
            raise ValueError("OLLAMA_EMBEDDING_MODEL environment variable not set")
        embeddings = ollama.embeddings(model=_model, prompt=text)
        return embeddings["embedding"]

    @staticmethod
    def get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("WARNING: MPS not available. Falling back to CPU.")
            return torch.device("cpu")