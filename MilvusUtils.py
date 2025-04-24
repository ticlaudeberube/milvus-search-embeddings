
from pymilvus import MilvusClient, Collection, MilvusException, connections, db, utility, model

#Start/install Milvus container before use
client = connections.connect(host="localhost", port=19530)

class MilvusUtils:
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
    def create_collection(collection_name: str, client: MilvusClient):
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            dimension=768,  # The vectors we will use in this demo has 768 dimensions
        )

    @staticmethod
    def delete_collection(collection_name: str, client: MilvusClient):
        client.drop_collection(collection_name=collection_name)

    @staticmethod
    def insert_data(collection_name: str, data: list[dict], client: MilvusClient) -> dict:        
        res = client.insert(collection_name=collection_name, data=data)
        return res

    @staticmethod
    def vectorize_documents(collection_name: str, docs: list[str], client: MilvusClient)-> list[dict]: 
        # This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
        embedding_fn = model.DefaultEmbeddingFunction()

        vectors = embedding_fn.encode_documents(docs)
        # The output vector has 768 dimensions, matching the collection that we just created.
        print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

        # Each entity has id, vector representation, raw text, and a subject label that we use
        # to demo metadata filtering later.
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]

        print("Data has", len(data), "entities, each with fields: ", data[0].keys())
        print("Vector dim:", len(data[0]["vector"]))

        res = client.insert(collection_name=collection_name, data=data)

        print(res)

        return res
