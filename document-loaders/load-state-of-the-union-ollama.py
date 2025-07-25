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
collection="state_of_the_union_ollama"  # Name of the collection to be created

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
    with open(filename , encoding='utf-8') as f:
        documents = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_text(documents)
    # print(texts[0])

    # Populate the vector database
    data = []
    for i, line in enumerate(tqdm(docs, desc="Creating embeddings")):
        data.append({"id": i, "vector": MilvusUtils.embed_text_ollama(line), "text": line})

    # print(data[0])
    dimension= len(MilvusUtils.embed_text_ollama(docs[0]))
    MilvusUtils.create_collection(
        collection_name=collection, dimension=dimension
    )

    res = client.insert(
        collection_name=collection,
        data=data
    )
    #print(f"{len(ids)} documents added to the vector database")

    print(res['insert_count'])

def search(query: str):
    cprint('\nSearching..\n', 'green', attrs=['blink'])
    # Querying the Vector Database

    query_vectors = MilvusUtils.embed_text_ollama(query)

    res = client.search(
        collection_name=collection,
        data=[query_vectors], 
        output_fields=["text"], 
        anns_field="vector",
        limit=3,
        search_params={"metric_type": "COSINE"}
    )
    
    response = ''
    for r in res[0]:
        response += r["entity"]["text"]

    print(response)

load()
search("What did the president say about Ketanji Brown Jackson?")

