
from sentence_transformers import SentenceTransformer
import os, torch, ollama
from pymilvus import MilvusClient, Collection, MilvusException, db, utility, model

#Start/install Milvus container before use
client = MilvusClient( 
    uri="http://localhost:19530",
    token="root:Milvus"
)

client.use_database(os.getenv("MY_DATABASE") or "default")

class MilvusUtils:
    @staticmethod
    def get_client():
        return client
    @staticmethod
    def create_database(db_name):
        try:
            existing_databases = db.list_database()
            if db_name in existing_databases:
                print(f"Database '{db_name}' already exists.")

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
    def create_collection(collection_name: str,dimension=1536):
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
        )
    @staticmethod 
    def has_collection(collection: str) -> bool:
        return client.has_collection(collection_name=collection)

    @staticmethod
    def delete_collection(collection_name: str):
        client.drop_collection(collection_name=collection_name)

    @staticmethod
    def insert_data(collection_name: str, data: list[dict]) -> dict:        
        res = client.insert(collection_name=collection_name, data=data)
        return res
    
    @staticmethod
    def vectorize_documents(collection_name: str, docs: list[str])-> list[dict, int]:
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
    def embed_text_hf(sentences:list[str], model="all-MiniLM-L6-v2") -> list[float]:
        model = SentenceTransformer(model)
        embeddings = model.encode(sentences, batch_size=256, show_progress_bar=True)
        return embeddings.tolist()
    
    @staticmethod
    def embed_text_ollama(text) -> list[float]:
        model = os.getenv('MODEL_OLLAMA') or "nomic-embed-text"
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]

    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("⚠️ MPS not available. Falling back to CPU.")
            return torch.device("cpu")
