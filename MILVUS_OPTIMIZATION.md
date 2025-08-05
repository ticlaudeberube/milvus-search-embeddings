# Milvus Docker Desktop Optimization Guide

## Recommended Index Parameters

### 1. HNSW (Best for Docker Desktop)
```python
# Optimal for most use cases on Docker Desktop
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="HNSW",
    index_name="vector_index",
    params={
        "M": 16,              # Connectivity (8-64, higher = better recall)
        "efConstruction": 200  # Build quality (100-500, higher = better quality)
    }
)

# Search parameters
search_params = {"ef": 64}  # Search quality (10-500, higher = better recall)
```

### 2. IVF_FLAT (Memory Efficient)
```python
# Good for limited memory environments
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={
        "nlist": 64  # Reduced from 128 for Docker Desktop
    }
)

# Search parameters
search_params = {"nprobe": 16}  # Balance between speed and accuracy
```

### 3. FLAT (Small Datasets)
```python
# For datasets < 10,000 vectors
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="FLAT",
    index_name="vector_index"
)
```

## Docker Desktop Resource Optimization

### 1. Docker Settings
```yaml
# Recommended Docker Desktop settings
Memory: 4-8 GB
CPUs: 2-4 cores
Swap: 1 GB
```

### 2. Milvus Configuration
```yaml
# docker-compose.yml optimizations
version: '3.5'
services:
  milvus:
    image: milvusdb/milvus:latest
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - ./milvus.yaml:/milvus/configs/milvus.yaml
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### 3. Collection Parameters
```python
# Optimized collection settings
OPTIMAL_PARAMS = {
    "collection_name": "optimized_collection",
    "consistency_level": "Eventually",  # Faster than "Strong"
    "num_shards": 1,                    # Single shard for Docker
    "num_replicas": 1                   # No replication needed locally
}
```

## Performance Tuning by Dataset Size

### Small Dataset (< 10K vectors)
```python
INDEX_TYPE = "FLAT"
PARAMS = {}
SEARCH_PARAMS = {}
```

### Medium Dataset (10K - 100K vectors)
```python
INDEX_TYPE = "HNSW"
PARAMS = {"M": 16, "efConstruction": 200}
SEARCH_PARAMS = {"ef": 64}
```

### Large Dataset (> 100K vectors)
```python
INDEX_TYPE = "IVF_FLAT"
PARAMS = {"nlist": min(4 * sqrt(n_vectors), 65536)}
SEARCH_PARAMS = {"nprobe": 16}
```

## Search Optimization Tips

### 1. Batch Operations
```python
# Insert in batches of 1000-5000
BATCH_SIZE = 2000

# Search multiple vectors at once
results = client.search(
    collection_name=collection_name,
    data=query_vectors,  # List of vectors
    limit=10,
    search_params=search_params
)
```

### 2. Memory Management
```python
# Load collection into memory for faster search
client.load_collection(collection_name)

# Release when not needed
client.release_collection(collection_name)
```

### 3. Connection Pooling
```python
# Reuse connections
from pymilvus import connections

connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    pool_size=10  # Connection pool
)
```

## Monitoring Performance

### 1. Check Index Status
```python
def check_index_status(client, collection_name):
    indexes = client.list_indexes(collection_name)
    for index in indexes:
        print(f"Index: {index}")
```

### 2. Memory Usage
```python
def get_collection_stats(client, collection_name):
    stats = client.get_collection_stats(collection_name)
    print(f"Row count: {stats['row_count']}")
    print(f"Data size: {stats.get('data_size', 'N/A')}")
```

### 3. Search Latency
```python
import time

def benchmark_search(client, collection_name, query_vector, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=10
        )
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average search time: {avg_time:.4f}s")
```

## Quick Performance Checklist

- ✅ Use HNSW for general purpose (best balance)
- ✅ Set Docker memory to 4-8GB
- ✅ Use batch operations (1000-5000 vectors)
- ✅ Load collections into memory before search
- ✅ Use "Eventually" consistency for faster writes
- ✅ Monitor index build progress
- ✅ Adjust search parameters based on accuracy needs
- ✅ Use connection pooling for multiple operations