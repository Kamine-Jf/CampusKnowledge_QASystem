"""Milvus连接配置常量。"""

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION_NAME = "campus_qa_vector"
VECTOR_DIM = 768
INDEX_PARAMS = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
