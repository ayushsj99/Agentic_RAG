from typing import List
from pathlib import Path
from langchain_core.documents import Document
from backend.doc_loader import load_docs
from backend.splitter import split_docs
from backend.storing import store_docs
from backend.ingestion_logger import log, info, success, warn, error, timed_step

def ingest_document(file_path: str) -> None:
    log("PIPELINE", f"=== Ingestion started for: {file_path} ===")
    info(f"File path resolved: {Path(file_path).resolve()}")

    with timed_step("FULL_INGESTION"):
        # Step 1: Load
        documents: List[Document] = load_docs(file_path)
        if not documents:
            error("No documents loaded. Ingestion aborted.")
            return
        log("PIPELINE", f"Step 1/3 complete -> {len(documents)} sections loaded")

        # Step 2: Split
        chunks: List[Document] = split_docs(documents)
        if not chunks:
            error("No chunks created. Ingestion aborted.")
            return
        log("PIPELINE", f"Step 2/3 complete -> {len(chunks)} chunks created")

        # Step 3: Store
        store_docs(chunks)
        log("PIPELINE", f"Step 3/3 complete -> chunks stored in vector DB")

    success(f"=== Ingestion finished: {Path(file_path).name} | {len(documents)} sections -> {len(chunks)} chunks ===")


# test
if __name__ == "__main__":
    ingest_document("backend/data/sample2.pdf")
    
    