from typing import List
from langchain_core.documents import Document
from uuid import uuid4
from database.database_config import vector_store, DATABASE_BACKEND
from database.reset_db import get_document_count
from langchain_community.vectorstores.utils import filter_complex_metadata
from backend.ingestion_logger import log, info, success, warn, error, timed_step
from backend.exceptions import IngestionError, retry


@retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
def _insert_to_store(documents: List[Document]):
    vector_store.add_documents(documents=documents)


def _collection_has_data() -> bool:
    try:
        return get_document_count() > 0
    except Exception as e:
        warn(f"Failed to check collection: {e}")
        return False


def store_docs(documents: List[Document])-> None:
    if not documents:
        warn("No documents to store.")
        return

    if _collection_has_data():
        error("A document is already stored. Reset the database before adding a new one.")
        raise RuntimeError("A document is already stored. Reset the database before adding a new one.")

    backend_name = DATABASE_BACKEND.upper()
    log("STORE", f"Preparing {len(documents)} chunks for {backend_name}")

    with timed_step("METADATA_FILTER"):
        filtered_documents = filter_complex_metadata(documents)
        dropped = len(documents) - len(filtered_documents)
        if dropped:
            warn(f"Metadata filter dropped {dropped} chunks")
        else:
            info(f"All {len(filtered_documents)} chunks passed metadata filter")

    with timed_step("DB_INSERT"):
        try:
            _insert_to_store(filtered_documents)
            success(f"Stored {len(filtered_documents)} chunks in {backend_name}")
        except Exception as e:
            error(f"{backend_name} insert failed after retries: {e}")
            raise IngestionError(f"Failed to store documents: {e}")
    
    