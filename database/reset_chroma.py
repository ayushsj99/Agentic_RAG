from database.chroma_db_setup import vector_store_chroma


def reset_chroma_db():
    """Delete all documents from the ChromaDB collection."""
    collection = vector_store_chroma._collection
    # Get all IDs in the collection
    results = collection.get()
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    print(f"ChromaDB reset: removed {len(ids)} documents.")
    return len(ids)
