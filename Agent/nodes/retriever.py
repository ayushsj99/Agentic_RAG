from langchain.tools import tool
from database.chroma_db_setup import vector_store_chroma
from backend.agent_logger import log


@tool()
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    log("retriever", f"Searching ChromaDB for: {query[:120]}")
    retrieved_docs = vector_store_chroma.similarity_search(query, k=10)
    log("retriever", f"Retrieved {len(retrieved_docs)} documents.")
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "unknown")
        log("retriever", f"  Doc {i+1} [{source}]: {doc.page_content[:150]}...")
    serialized = "\n\n---\n\n".join(
        doc.page_content for doc in retrieved_docs
    )
    return serialized

retriever_tool = retrieve_context