import os, sys
from pymilvus import MilvusClient, DataType

sys.path.insert(1, './utils')
from MilvusUtils import MilvusUtils

client = MilvusUtils.get_client()
collection_name = os.getenv("MY_COLLECTION_NAME") or "demo_collection"

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="metadata", datatype=DataType.JSON)

client.create_collection(
    collection_name=collection_name, 
    schema=schema, 
)

index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

client.create_index(
    collection_name=collection_name,
    index_params=index_params,
    sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
)

