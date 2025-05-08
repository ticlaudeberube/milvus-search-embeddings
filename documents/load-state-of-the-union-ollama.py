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

sys.path.insert(1, './utils')
from MilvusUtils import MilvusUtils


client = MilvusUtils.get_client()
collection="state_of_the_union"  # Name of the collection to be created

def load():
     # Building the Vector Database
    filename = 'state_of_the_union.txt'
    url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'

    if not os.path.isfile(filename):
        wget.download(url, out=filename)

    # Split the document into chunks
    loader = TextLoader(filename)
    documents = loader.load()

    # This is a long document we can split up.
    with open(f"./{filename}") as f:
        documents = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_text(documents)
    # print(texts[0])

    # Populate the vector database
    data = []
    for i, line in enumerate(tqdm(texts, desc="Creating embeddings")):
        data.append({"id": i, "vector": MilvusUtils.embed_text_ollama(line), "text": line})

    # print(data[0])

    MilvusUtils.create_collection(
        collection_name=collection, dimension=1024
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
    )
    response = ''
    for r in res[0]:
        response += r["entity"]["text"]

    print(response)

# load()
search("What did the president say about Ketanji Brown Jackson?")

