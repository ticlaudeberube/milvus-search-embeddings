import time
import torch
from sentence_transformers import SentenceTransformer

# Test data: a list of sentences
sentences = ["This is a sample sentence."] * 1000  # Adjust number as needed

# Model to use
model_name = "all-MiniLM-L6-v2" #mps time: 256/2.01 1000/1.84
# model_name= "multi-qa-minilm-l6-cos-v1" # mps time: 256/2.01, 1000/1.83
# model_name = "mixedbread-ai/mxbai-embed-large-v1" #mps time: 256/21.55 1000/20.21

# Determine available device (MPS, CPU fallback)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print("⚠️ MPS not available. Falling back to CPU.")
        return torch.device("cpu")

# Benchmark function
def benchmark(device):
    print(f"\nRunning on {device}...")
    model = SentenceTransformer(model_name, device=device)

    start = time.time()
    # Augment batch_size for MPS performance optimization
    # cpu time: 2.72 
    # (64) mps time: 3.59
    # (256) mps time: 1.34
    embeddings = model.encode(sentences, batch_size=1000, show_progress_bar=True)
    end = time.time()

    print(f"{device} time: {end - start:.2f} seconds")
    return embeddings

# Run benchmark
device = get_device()
benchmark(device)
