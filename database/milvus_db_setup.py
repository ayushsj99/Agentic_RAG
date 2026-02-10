"""
Milvus vector store setup for Agentic RAG with Hybrid Search.

Uses dense embeddings (Ollama) + sparse embeddings (BM25) for better retrieval.

Requires Milvus running locally via Docker:
  docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
"""

from langchain_milvus import Milvus, BM25BuiltInFunction
from models.ollama_emb import ollama_embeddings

MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "agentic_rag_hybrid"

bm25_function = BM25BuiltInFunction(
    input_field_names="text",
    output_field_names="sparse_vector",
)

vector_store_milvus = Milvus(
    embedding_function=ollama_embeddings,
    builtin_function=bm25_function,
    vector_field="dense_vector",
    collection_name=MILVUS_COLLECTION,
    connection_args={"uri": MILVUS_URI},
    auto_id=True,
    drop_old=False,
    enable_dynamic_field=True,
)
