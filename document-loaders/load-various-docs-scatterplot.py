import os, wget, sys
import asyncio
import ollama
from langchain_community.document_loaders import TextLoader, PyPDFLoader
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
collection_name = "ollama_scatterplot_collection"# Name of the collection to be created
data = []
docs = []

async def load():
    # Building the Vector Database
    files = [
        {
            'filename': 'pantoja-vs-asakura.html',
            'url': 'https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-310-pantoja-vs-asakura'
        },
        {
            'filename': 'state_of_the_union.txt',
            'url': 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'
        },
        { 
            'filename': 'Unified_Rules_MMA.pdf',
            'url': 'https://media.ufc.tv/discover-ufc/Unified_Rules_MMA.pdf'
        }
    ]

    for file in files:
        filename = file['filename']
        url = file['url']
        
        if not os.path.isfile(filename):
            wget.download(url, out=filename)

        filetype = filename.split('.')[-1]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        match filetype:
            case 'txt':
                loader = TextLoader(filename)
                # load and split the document into chunks
                documents = loader.load()
                with open(f"./{filename}") as f:
                    documents = f.read()
                    docs = text_splitter.split_text(documents)
            case 'pdf':
                loader = PyPDFLoader(filename)
                async for page in loader.alazy_load():
                    chunks = text_splitter.split_text(page.page_content)
                    for chunk in chunks:
                        docs.append(chunk)
            case 'html':
                with open(filename, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    docs = text_splitter.split_text(html_content)                        
            case _:
                raise ValueError(f"Unsupported file type: {filetype}")
            
    # Populate the vector database
    for i, line in enumerate(tqdm(docs, desc="Creating (Ollama) embeddings")):
        data.append({"id": i, "vector": MilvusUtils.embed_text_ollama(line), "text": line})

    # get vector dimension
    text_vector = MilvusUtils.embed_text_ollama(docs[0])
    dim = len(text_vector)  #Dim: 1024
    MilvusUtils.create_collection(
        collection_name=collection_name, 
        dimension=dim,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",
    )
        
    client.insert(
        collection_name=collection_name,
        data=data
    )

    print(len(data))

def search(query) -> list[dict, list[float]]:
    query_vectors = MilvusUtils.embed_text_ollama(query)

    search_result = client.search(
        collection_name=collection_name,
        data=[query_vectors],
        limit=10,
        search_params={
            "params": {"radius": 0.4, "range_filter": 0.7}
        },
    )

    response = ''
    for r in search_result[0]:
        response += r["entity"]["text"]

    cprint(f'\nDone Searching!:\n\n {response}\n', 'green', attrs=['blink'])
    return search_result, query_vectors

def show_plot(search_res):
    embeddings = []
    for gp in data:
        embeddings.append(gp["vector"])

    X = np.array(embeddings, dtype=np.float32)
    tsne = TSNE(random_state=0, max_iter=1000)
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])

    similar_ids = [gp["id"] for gp in search_res[0]]

    df_query = pd.DataFrame(df_tsne.iloc[-1]).T

    similar_points = df_tsne[df_tsne.index.isin(similar_ids)]

    fig, ax = plt.subplots(figsize=(8, 6))  # Set figsize

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    sns.scatterplot(
        data=df_tsne, x="TSNE1", y="TSNE2", color="blue", label="All knowledge", ax=ax
    )

    sns.scatterplot(
        data=similar_points,
        x="TSNE1",
        y="TSNE2",
        color="red",
        label="Similar knowledge",
        ax=ax,
    )

    sns.scatterplot(
        data=df_query, x="TSNE1", y="TSNE2", color="green", label="Query", ax=ax
    )

    plt.title("Scatter plot of knowledge using t-SNE")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")

    plt.axis("equal")

    plt.legend()

    plt.show()

async def main():
    await load()

    querySoU = "What did the president say about Ketanji Brown Jackson?"
    queryMMA = "What happens when a competitor is injured?"
    queryUFC = "How much weight allowance is allowed in non championship fights in the UFC?"
    queryUFC310 = "Who won in the Pantoja vs Asakura fight at UFC 310?"

    query = querySoU
    s, queryVector = search(query)
    if len(data) > 0:
        data.append({"id": len(data)+1, "vector": queryVector, "text": f"{query}"})
        show_plot(s)

asyncio.run(main())

