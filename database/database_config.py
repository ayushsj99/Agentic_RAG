"""
Unified database configuration for Agentic RAG.

Set DATABASE_BACKEND below to switch between vector stores:
  - "chroma": ChromaDB (local, file-based)
  - "milvus": Milvus (requires Docker container running)

The vector_store object exported here is used throughout the project.
"""

DATABASE_BACKEND = "chroma"  # Options: "chroma" or "milvus"

VALID_BACKENDS = ("chroma", "milvus")
if DATABASE_BACKEND not in VALID_BACKENDS:
    raise ValueError(f"DATABASE_BACKEND must be one of {VALID_BACKENDS}, got: {DATABASE_BACKEND}")

if DATABASE_BACKEND == "chroma":
    from database.chroma_db_setup import vector_store_chroma as vector_store
elif DATABASE_BACKEND == "milvus":
    from database.milvus_db_setup import vector_store_milvus as vector_store

__all__ = ["vector_store", "DATABASE_BACKEND"]
