from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv

from core import MilvusUtils

load_dotenv()

# Constants
COLLECTION_NAME: str = "hello_world_collection"
VECTOR_DIM: int = 5
NLIST_PARAM: int = 128
MAX_TEXT_LENGTH: int = 65535

client: MilvusClient = MilvusUtils.get_client()

# Drop collection if exists
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print(f"Dropped existing collection: {COLLECTION_NAME}")

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH)
schema.add_field(field_name="metadata", datatype=DataType.JSON)

try:
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"Collection '{COLLECTION_NAME}' created successfully")
except Exception as e:
    print(f"Error creating collection: {e}")
    exit(1)

index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={"nlist": NLIST_PARAM}
)

try:
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
        sync=False
    )
    print(f"Index created for collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error creating index: {e}")
    exit(1)