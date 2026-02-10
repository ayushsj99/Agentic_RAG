from typing import List
from langchain_core.documents import Document
from uuid import uuid4
from database.chroma_db_setup import vector_store_chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from backend.ingestion_logger import log, info, success, warn, error, timed_step
from backend.exceptions import IngestionError, retry


@retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
def _insert_to_chroma(documents: List[Document]):
    vector_store_chroma.add_documents(documents=documents)


def _collection_has_data() -> bool:
    try:
        collection = vector_store_chroma._collection
        return collection.count() > 0
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

    log("STORE", f"Preparing {len(documents)} chunks for ChromaDB")

    with timed_step("METADATA_FILTER"):
        filtered_documents = filter_complex_metadata(documents)
        dropped = len(documents) - len(filtered_documents)
        if dropped:
            warn(f"Metadata filter dropped {dropped} chunks")
        else:
            info(f"All {len(filtered_documents)} chunks passed metadata filter")

    with timed_step("CHROMA_INSERT"):
        try:
            _insert_to_chroma(filtered_documents)
            success(f"Stored {len(filtered_documents)} chunks in ChromaDB")
        except Exception as e:
            error(f"ChromaDB insert failed after retries: {e}")
            raise IngestionError(f"Failed to store documents: {e}")
    
    