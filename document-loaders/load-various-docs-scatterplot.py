import os, wget
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from termcolor import cprint
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from core import MilvusUtils

# Initialize Milvus client and global variables
client = MilvusUtils.get_client()
collection_name = "ollama_scatterplot_collection"  # Name of the collection to be created
# To force re-embedding: client.drop_collection(collection_name)
data = []  # Store document chunks with embeddings
docs = []  # Store raw document text chunks

async def load():
    """Load documents, create embeddings, and populate Milvus vector database
    
    Behavior:
    - If collection exists: Loads existing data, skips embedding
    - If collection doesn't exist: Re-embeds documents (but uses cached downloads)
    - To force re-embedding: Drop collection first using client.drop_collection(collection_name)
    """
    # Skip loading if collection already exists, but load data for visualization
    if MilvusUtils.has_collection(collection_name):
        print(f"Collection {collection_name} already exists. Loading existing data.")
        global data
        try:
            all_data = client.query(collection_name, "", output_fields=["id", "vector", "text"], limit=10000)
            data = list(all_data)
        except Exception as e:
            print(f"Could not load existing data: {e}")
            data = []
        return
    
    # Define files to download and process
    files = {
        'pantoja-vs-asakura.html': 'https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-310-pantoja-vs-asakura',
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
            chunk_size=256,
            chunk_overlap=20,
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
                # Load and split HTML files, removing markup
                with open(filepath, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    clean_text = soup.get_text(separator=' ', strip=True)
                    docs.extend(text_splitter.split_text(clean_text))                        
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
        dimension=dim
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
        output_fields=["text"],
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
    if not data:
        print("No data available for plotting")
        return
        
    # Extract all embeddings from the data (including query)
    embeddings = []
    for gp in data:
        embeddings.append(gp["vector"])

    # Apply t-SNE dimensionality reduction to visualize high-dimensional embeddings in 2D
    X = np.array(embeddings, dtype=np.float32)
    tsne = TSNE(random_state=0, max_iter=1000)
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])

    similar_ids = [gp["id"] for gp in search_res[0]]

    # Extract query point (last point in data)
    df_query = pd.DataFrame(df_tsne.iloc[-1:])

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

    sns.scatterplot(
        data=df_query, x="TSNE1", y="TSNE2", color="green", label="Query", ax=ax
    )

    plt.title("Scatter plot of knowledge using t-SNE")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.axis("equal")
    plt.legend()
    plt.show()
    plt.close()

async def main():
    """Main function to orchestrate document loading, searching, and visualization"""
    # Load documents and create vector database
    await load()

    # Define sample queries for different document types
    querySoU = "What did the president say about Ketanji Brown Jackson?"  # State of Union query
    queryMMA = "What happens when a competitor is injured?"  # MMA rules query
    queryUFC = "How much weight allowance is allowed in non championship fights in the UFC?"  # UFC rules query
    queryUFC310 = "Who won in the Pantoja vs Asakura fight at UFC 310?"  # UFC 310 results query

    # Execute search with selected query if collection exists
    if MilvusUtils.has_collection(collection_name):
        query = querySoU
        s, queryVector = search(query)
    
        # Show t-SNE plot if data is available
        if len(data) > 0:
            data.append({"id": len(data)+1, "vector": queryVector, "text": f"{query}"})
            show_plot(s)
            
    print("Script completed successfully.")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())