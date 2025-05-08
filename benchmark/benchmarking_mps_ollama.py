import time
from ollama import Client
from tqdm import tqdm

client = Client(host='http://localhost:11434')  # default port for Ollama

# Sample texts for embedding
sentences = ["This is a test sentence."] * 1000  # simulate batch of 1000

# Choose model that supports embedding (e.g., 'nomic-embed-text')
model = "nomic-embed-text"  # time 17.69s
# model = "mxbai-embed-large" #time 35.78
# model = "llama3.2" #time  212s

# Benchmark embedding time
def benchmark(sentences):
    print(f"Generating embeddings for {len(sentences)} sentences...")
    start = time.time()

    embeddings = []
    for text in tqdm(sentences, desc="Generating embeddings"):
        response = client.embeddings(model=model, prompt=text)
        embeddings.append(response['embedding'])

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    print(f"Average per sentence: {(end - start) / len(sentences):.4f} seconds")

    return embeddings

# Run benchmark
benchmark(sentences)
