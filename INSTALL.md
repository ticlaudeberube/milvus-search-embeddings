# Installation Instructions

## Development Setup

1. Install the package in development mode:
```bash
pip install -e .
```

Or run the setup script:
```bash
python setup_dev.py
```

2. Now you can run scripts from any directory:
```bash
# From project root
python utils/create_collection.py my_collection

# From any subdirectory
python ../utils/create_collection.py my_collection

# From anywhere in the project
python -m milvus_search_embeddings.utils.create_collection my_collection
```

## Usage

Import the utilities in your scripts:
```python
from core.utils import MilvusClient

# Use the utilities
client = MilvusClient.get_client()
MilvusClient.create_collection("my_collection")
```

## Testing

Run tests from project root:
```bash
pytest Tests/
```