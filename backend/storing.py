from typing import List
from langchain_core.documents import Document
from database.milvus_config import vector_store

def store_docs(documents: List[Document])-> None:
    if not documents:
        print("No documents to store.")
        return
    
    try:
        vector_store.add_documents(documents)
        print(f"Successfully stored {len(documents)} documents in the vector store.")
    except Exception as e:
        print(f"Error storing documents: {e}")
        raise
    
    