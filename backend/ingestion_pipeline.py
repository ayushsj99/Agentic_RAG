from typing import List
from pathlib import Path
from langchain_core.documents import Document
from backend.doc_loader import load_docs
from backend.splitter import split_docs
from backend.storing import store_docs

def ingest_document(file_path: str) -> None:
    print(f"Starting ingestion for {file_path}...")
    
    documents: List[Document] = load_docs(file_path)
    if not documents:
        print("No documents loaded. Ingestion aborted.")
        return
    
    chunks: List[Document] = split_docs(documents)
    if not chunks:
        print("No chunks created. Ingestion aborted.")
        return
    
    store_docs(chunks)
    print(f"Ingestion completed for {file_path}. Total chunks stored: {len(chunks)}")
    
    