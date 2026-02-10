from langchain_chroma import Chroma
from models.ollama_emb import ollama_embeddings

vector_store_chroma = Chroma(
    collection_name="example_collection",
    embedding_function=ollama_embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)