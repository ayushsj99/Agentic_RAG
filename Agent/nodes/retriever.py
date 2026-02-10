from langchain.tools import tool
from database.chroma_db_setup import vector_store_chroma
from backend.agent_logger import log
from backend.exceptions import RetrieverError, retry


@retry(max_attempts=2, delay=0.5, exceptions=(Exception,))
def _search_docs(query: str, k: int = 10):
    return vector_store_chroma.similarity_search(query, k=k)


@tool()
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    log("retriever", f"Searching ChromaDB for: {query[:120]}")
    try:
        retrieved_docs = _search_docs(query, k=10)
    except Exception as e:
        log("retriever", f"Search failed: {e}")
        raise RetrieverError(f"Document retrieval failed: {e}")
    
    log("retriever", f"Retrieved {len(retrieved_docs)} documents.")
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "unknown")
        log("retriever", f"  Doc {i+1} [{source}]: {doc.page_content[:150]}...")
    serialized = "\n\n---\n\n".join(
        doc.page_content for doc in retrieved_docs
    )
    return serialized if serialized else "No relevant documents found."

retriever_tool = retrieve_context