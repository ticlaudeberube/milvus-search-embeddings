from sentence_transformers import SentenceTransformer
import os, torch, ollama # type: ignore
from pymilvus import MilvusClient, Collection, MilvusException, db, utility, model

#Start/install Milvus container before use
client = MilvusClient( 
    uri="http://localhost:19530",
    token="root:Milvus"
)

client.use_database(os.getenv("MY_DATABASE") or "default")

class MilvusUtils:
    """
    Utility class for interacting with Milvus vector database, including collection management and document vectorization.
    """

    @staticmethod
    def get_client():
        """Returns the Milvus client."""
        return client

    @staticmethod
    def create_database(db_name):
        """
        Creates a new database or resets an existing one by dropping all its collections.

        Args:
            db_name (str): The name of the database to create or reset.

        Returns:
            None
        """
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
        """
        Creates a new collection in Milvus with the specified name and dimension.

        Args:
            collection_name (str): The name of the collection to create.
            dimension (int): The dimension of the vectors that will be stored in the collection.

        Returns:
            None
        """
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
        )

    @staticmethod 
    def has_collection(collection: str) -> bool:
        """
        Checks if a collection exists in Milvus.

        Args:
            collection (str): The name of the collection to check.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        return client.has_collection(collection_name=collection)

    @staticmethod
    def delete_collection(collection_name: str):
        """
        Deletes a collection from Milvus.

        Args:
            collection_name (str): The name of the collection to delete.

        Returns:
            None
        """
        client.drop_collection(collection_name=collection_name)

    @staticmethod
    def insert_data(collection_name: str, data: list[dict]) -> dict:        
        """
        Inserts data into a Milvus collection.

        Args:
            collection_name (str): The name of the collection to insert data into.
            data (list of dict): The data to insert, where each item is a dictionary representing a vector and its metadata.

        Returns:
            dict: The result of the insert operation.
        """
        res = client.insert(collection_name=collection_name, data=data)
        return res
    
    @staticmethod
    def vectorize_documents(collection_name: str, docs: list[str]) -> list[dict, int]:
        """
        Vectorizes a list of documents and inserts them into the specified Milvus collection.

        Args:
            collection_name (str): The name of the Milvus collection to insert vectors into.
            docs (list of str): The list of text documents to vectorize.

        Returns:
            None
        """
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

        return res, dimension
    
    @staticmethod
    def embed_text(text: str | list[str], provider: str = None, model: str = None) -> list[float]:
        """
        Generic embedding interface that supports multiple providers.

        Args:
            text (str | list[str]): The text to embed. Can be a single string or list of strings.
            provider (str, optional): The provider to use ('hf' or 'ollama'). If not provided, uses EMBEDDING_PROVIDER env var.
            model (str, optional): The model name to use. If not provided, uses MODEL_HF or MODEL_OLLAMA env var based on provider.

        Returns:
            list of float: The embeddings for the text.
        """
        provider = provider or os.getenv('EMBEDDING_PROVIDER', 'hf')
        
        if provider == 'hugginface':
            _model = model or os.getenv('MODEL_HF')
            st = SentenceTransformer(_model)
            # Convert single string to list for consistent handling
            text_input = [text] if isinstance(text, str) else text
            embeddings = st.encode(text_input, batch_size=256, show_progress_bar=True)
            return embeddings.tolist() if len(text_input) > 1 else embeddings[0].tolist()
        
        elif provider == 'ollama':
            _model = model or os.getenv('MODEL_OLLAMA')
            # Ollama only supports single text input
            if isinstance(text, list):
                # For lists, we process one at a time and return list of embeddings
                return [ollama.embeddings(model=_model, prompt=t)["embedding"] for t in text]
            return ollama.embeddings(model=_model, prompt=text)["embedding"]
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
            
    # Keep old methods for backward compatibility but mark as deprecated
    @staticmethod
    def embed_text_hf(sentences:list[str], model = None) -> list[float]:
        """
        DEPRECATED: Use embed_text() with provider='ollama' instead.
      
        Generates embeddings for a list of sentences using a Hugging Face model.

        Args:
            sentences (list of str): The sentences to embed.
            model (str, optional): The name of the Hugging Face model to use. If not provided, the model from the 'MODEL_HF' environment variable is used.

        Returns:
            list of float: The embeddings for the sentences.
        """
        _model =  model or os.getenv('MODEL_HF')
        st = SentenceTransformer(_model)
        embeddings = st.encode(sentences, batch_size=256, show_progress_bar=True)
        return embeddings.tolist()
    @staticmethod
    def embed_text_ollama(text: str, model = None) -> list[float]:
        """
        DEPRECATED: Use embed_text() with provider='ollama' instead.
        
        Generates embeddings for a text using an Ollama model.

        Args:
            text (str): The text to embed.
            model (str, optional): The name of the Ollama model to use. If not provided, the model from the 'MODEL_OLLAMA' environment variable is used.

        Returns:
            list of float: The embedding for the text.
        """
        _model = model or os.getenv('MODEL_OLLAMA')
        embeddings = ollama.embeddings(model=_model, prompt=text)
        return embeddings["embedding"]

    @staticmethod
    def get_device():
        """
        Determines the device to be used for tensor computations.

        Returns:
            torch.device: The device (e.g., CPU or MPS) to be used.
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("⚠️ MPS not available. Falling back to CPU.")
            return torch.device("cpu")
