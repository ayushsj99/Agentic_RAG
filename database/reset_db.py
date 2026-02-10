"""
Unified database reset for Agentic RAG.
Supports both ChromaDB and Milvus backends.
"""

from database.database_config import DATABASE_BACKEND, vector_store


def reset_database() -> int:
    """Delete all documents from the vector store. Returns count of deleted docs."""
    if DATABASE_BACKEND == "chroma":
        return _reset_chroma()
    elif DATABASE_BACKEND == "milvus":
        return _reset_milvus()
    else:
        raise ValueError(f"Unknown backend: {DATABASE_BACKEND}")


def _reset_chroma() -> int:
    collection = vector_store._collection
    results = collection.get()
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    print(f"ChromaDB reset: removed {len(ids)} documents.")
    return len(ids)


def _reset_milvus() -> int:
    try:
        from pymilvus import connections, utility, Collection
        from database.milvus_db_setup import MILVUS_COLLECTION, MILVUS_URI
        
        host = MILVUS_URI.replace("http://", "").split(":")[0]
        port = MILVUS_URI.replace("http://", "").split(":")[1]
        
        connections.connect(alias="default", host=host, port=port)
        
        if utility.has_collection(MILVUS_COLLECTION):
            col = Collection(MILVUS_COLLECTION)
            count = col.num_entities
            col.drop()
            print(f"Milvus reset: dropped collection with {count} documents.")
            return count
        else:
            print("Milvus reset: collection does not exist.")
            return 0
    except Exception as e:
        print(f"Milvus reset failed: {e}")
        return 0
    finally:
        try:
            connections.disconnect("default")
        except:
            pass


def get_document_count() -> int:
    """Get the number of documents in the vector store."""
    if DATABASE_BACKEND == "chroma":
        try:
            return vector_store._collection.count()
        except:
            return 0
    elif DATABASE_BACKEND == "milvus":
        try:
            from pymilvus import connections, Collection, utility
            from database.milvus_db_setup import MILVUS_COLLECTION, MILVUS_URI
            
            host = MILVUS_URI.replace("http://", "").split(":")[0]
            port = MILVUS_URI.replace("http://", "").split(":")[1]
            
            connections.connect(alias="default", host=host, port=port)
            
            if utility.has_collection(MILVUS_COLLECTION):
                col = Collection(MILVUS_COLLECTION)
                col.flush()
                return col.num_entities
            return 0
        except:
            return 0
        finally:
            try:
                connections.disconnect("default")
            except:
                pass
    return 0
