from dotenv import load_dotenv
load_dotenv()
from core.MilvusUtils import MilvusUtils

collection_name = "demo_collection"

documents = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

MilvusUtils.vectorize_documents(collection_name, documents)