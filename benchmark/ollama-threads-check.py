import subprocess
import time
import os
import matplotlib.pyplot as plt

# Configuration
model = "nomic-embed-text"  # or another Ollama-compatible model
prompt = "This is a test sentence for benchmarking."
thread_counts = [1, 2, 4, 6, 8]  # adjust based on your CPU
num_runs_per_thread = 3  # average out for stability

def run_ollama(thread_count):
    os.environ["OLLAMA_NUM_THREADS"] = str(thread_count)
    total_time = 0.0
    for _ in range(num_runs_per_thread):
        start = time.time()
        subprocess.run(
            ["ollama", "run", model, "-p", prompt],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        total_time += time.time() - start
    avg_time = total_time / num_runs_per_thread
    return avg_time

# Run benchmark
results = {}
for t in thread_counts:
    print(f"Benchmarking with {t} thread(s)...")
    avg = run_ollama(t)
    results[t] = avg
    print(f"  Average time: {avg:.2f} seconds")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.title("Ollama Inference Time vs Thread Count")
plt.xlabel("Threads")
plt.ylabel("Average Time (s)")
plt.grid(True)
plt.tight_layout()
plt.show()
