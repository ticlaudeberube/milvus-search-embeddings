import os, wget, sys
import ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import MilvusClient
from pymilvus import model
from tqdm import tqdm
from termcolor import colored, cprint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.MilvusUtils import MilvusUtils

client = MilvusUtils.get_client()
collection="state_of_the_union_default"  # Name of the collection to be created
dimension = 768

# Use the `embed_text` function to convert the question to an embedding vector
def embed_text(text):
    # this embedding method takes 4m longer than ollama
    embedding_fn = model.DefaultEmbeddingFunction()
    vector = embedding_fn.encode_queries([text])
    return vector

def load():
    # Building the Vector Database
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'state_of_the_union.txt')
    url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'

    if not os.path.isfile(filename):
        wget.download(url, out=filename)

    # Split the document into chunks
    loader = TextLoader(filename, encoding='utf-8')
    documents = loader.load()

    # This is a long document we can split up.
    with open(filename, encoding='utf-8') as f:
        documents = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_text(documents)

    # Populate the vector database
    # Populate the vector database
    text_vector = embed_text(docs[0])[0] #Dim: 768
    data = []
    for i, line in enumerate(tqdm(docs, desc="Creating embeddings")):
        data.append({"id": i, "vector": embed_text(line), "text": line})


    MilvusUtils.create_collection(
        collection_name=collection, dimension=len(text_vector)
    )

    res, dim = MilvusUtils.vectorize_documents(collection, docs)

    print(res['insert_count'])

def search(query):
    cprint('\nSearching..\n', 'green', attrs=['blink'])
    # Querying the Vector Database
    query_vectors = embed_text(query)
  
    search_result = client.search(
        collection_name=collection,
        data=query_vectors, 
        output_fields=["text"], 
    )
    # print(search_result)
    response = ''
    for r in search_result[0]:
        response += r["entity"]["text"]

    # print(f"Response: {response}")
    cprint(f'\nDone Searching!:\n\n {response}\n', 'green', attrs=['blink'])

load()
query = "What did the president say about Ketanji Brown Jackson?"
search(query)

