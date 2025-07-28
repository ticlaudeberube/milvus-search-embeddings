#!/usr/bin/env python3
"""
Document Loader with Milvus Vector Database and t-SNE Visualization

This script demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline:
1. Downloads various document types (TXT, PDF, HTML) if not present locally
2. Splits documents into chunks using LangChain text splitters
3. Creates embeddings using Ollama models
4. Stores embeddings in Milvus vector database
5. Performs similarity search on user queries
6. Visualizes results using t-SNE dimensionality reduction

Supported file types: TXT, PDF, HTML
Embedding model: Ollama (default model)
Vector database: Milvus
Visualization: t-SNE scatter plot with matplotlib/seaborn
"""

import os
import wget
import sys
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# Initialize Milvus client and global variables
client = MilvusUtils.get_client()
collection_name = "ollama_scatterplot_collection"  # Name of the collection to be created
data = []  # Store document chunks with embeddings
docs = []  # Store raw document text chunks

async def load():
    """Load documents, create embeddings, and populate Milvus vector database"""
    # Check if collection already exists and has data
    if MilvusUtils.has_collection(collection_name):
        count = client.query(collection_name=collection_name, expr="", output_fields=["count(*)"])
        if count and len(count) > 0:
            print(f"Collection {collection_name} already exists with data. Skipping load.")
            return
    
    # Define files to download and process
    files = {
        'state_of_the_union.txt': 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt',
        'Unified_Rules_MMA.pdf': 'https://media.ufc.tv/discover-ufc/Unified_Rules_MMA.pdf'
    }

    # Get script directory for consistent file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Download files if they don't exist locally
    for filename, url in files.items():
        filepath = os.path.join(script_dir, filename)
        if not os.path.isfile(filepath):
            try:
                print(f"Downloading {filename}...")
                wget.download(url, out=filepath)
                print(f"\nDownloaded {filename} successfully")
            except Exception as e:
                print(f"\nError downloading {filename}: {e}")
                print(f"Please manually download {filename} from {url}")
                continue

        # Get file extension to determine processing method
        filetype = filename.split('.')[-1]

        # Configure text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,  # Size of each text chunk
            chunk_overlap=20,  # Overlap between chunks
            length_function=len,
            is_separator_regex=False,
        )

        # Process different file types
        match filetype:
            case 'txt':
                # Load and split text files
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents = f.read()
                    docs.extend(text_splitter.split_text(documents))
            case 'pdf':
                # Load and split PDF files page by page
                loader = PyPDFLoader(filepath)
                async for page in loader.alazy_load():
                    chunks = text_splitter.split_text(page.page_content)
                    docs.extend(chunks)
            case 'html':
                # Load and split HTML files
                with open(filepath, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    docs.extend(text_splitter.split_text(html_content))                        
            case _:
                raise ValueError(f"Unsupported file type: {filetype}")
            
    # Create embeddings for each document chunk using Ollama
    for i, line in enumerate(tqdm(docs, desc="Creating (Ollama) embeddings")):
        data.append({"id": i, "vector": MilvusUtils.embed_text_ollama(line), "text": line})

    # Get vector dimension from first embedding
    text_vector = MilvusUtils.embed_text_ollama(docs[0])
    dim = len(text_vector)  # Typically 1024 for Ollama embeddings
    
    # Create Milvus collection with appropriate settings
    MilvusUtils.create_collection(
        collection_name=collection_name, 
        dimension=dim,
        metric_type="COSINE",  # Inner product distance for similarity
        consistency_level="Session",
    )
        
    # Insert all embeddings into Milvus collection
    client.insert(
        collection_name=collection_name,
        data=data
    )

    print(f"Loaded {len(data)} document chunks into Milvus")

def search(query) -> tuple[list, list[float]]:
    """Perform similarity search in Milvus and return results"""
    # Create embedding for the search query
    query_vectors = MilvusUtils.embed_text_ollama(query)

    # Search for similar vectors in Milvus
    search_result = client.search(
        collection_name=collection_name,
        data=[query_vectors],
        limit=10,  # Return top 10 similar results
        search_params={
            "params": {"radius": 0.4, "range_filter": 0.7}  # Similarity thresholds
        },
    )

    # Concatenate all search results into response text
    response = ''
    for r in search_result[0]:
        response += r["entity"]["text"]

    # Display search results with colored output
    cprint(f'\nDone Searching!:\n\n {response}\n', 'green', attrs=['blink'])
    return search_result, query_vectors

def show_plot(search_res):
    """Create t-SNE visualization of embeddings and search results"""
    # Extract all embeddings from the data
    embeddings = []
    for gp in data:
        embeddings.append(gp["vector"])

    # Apply t-SNE dimensionality reduction to visualize high-dimensional embeddings in 2D
    X = np.array(embeddings, dtype=np.float32)
    tsne = TSNE(random_state=0, max_iter=1000)
    tsne_results = tsne.fit_transform(X)

    # Create DataFrame with t-SNE coordinates
    df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])

    # Get IDs of similar documents from search results
    similar_ids = [gp["id"] for gp in search_res[0]]

    # Extract query point (last point added to data)
    df_query = pd.DataFrame(df_tsne.iloc[-1]).T

    # Filter points that are similar to the query
    similar_points = df_tsne[df_tsne.index.isin(similar_ids)]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    # Plot all knowledge points in blue
    sns.scatterplot(
        data=df_tsne, x="TSNE1", y="TSNE2", color="blue", label="All knowledge", ax=ax
    )

    # Highlight similar knowledge points in red
    sns.scatterplot(
        data=similar_points,
        x="TSNE1",
        y="TSNE2",
        color="red",
        label="Similar knowledge",
        ax=ax,
    )

    # Show query point in green
    sns.scatterplot(
        data=df_query, x="TSNE1", y="TSNE2", color="green", label="Query", ax=ax
    )

    # Configure plot appearance
    plt.title("Scatter plot of knowledge using t-SNE")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.axis("equal")
    plt.legend()
    plt.show()

async def main():
    """Main function to orchestrate document loading, searching, and visualization"""
    # Load documents and create vector database
    await load()

    # Define sample queries for different document types
    querySoU = "What did the president say about Ketanji Brown Jackson?"  # State of Union query
    queryMMA = "What happens when a competitor is injured?"  # MMA rules query
    queryUFC = "How much weight allowance is allowed in non championship fights in the UFC?"  # UFC rules query
    queryUFC310 = "Who won in the Pantoja vs Asakura fight at UFC 310?"  # UFC 310 results query

    # Execute search with selected query
    query = querySoU
    s, queryVector = search(query)
    
    # Add query to data for visualization and show t-SNE plot
    if len(data) > 0:
        data.append({"id": len(data)+1, "vector": queryVector, "text": f"{query}"})
        show_plot(s)

# Run the main function
asyncio.run(main())

