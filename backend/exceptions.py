import time
import functools
from backend.ingestion_logger import warn, error


class RAGException(Exception):
    pass


class LLMError(RAGException):
    pass


class EmbeddingError(RAGException):
    pass


class RetrieverError(RAGException):
    pass


class IngestionError(RAGException):
    pass


class ServiceUnavailableError(RAGException):
    pass


class ValidationError(RAGException):
    pass


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        warn(f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}")
                        time.sleep(wait)
                        wait *= backoff
                    else:
                        error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
            raise last_exception
        return wrapper
    return decorator


def check_ollama_health(base_url: str = "http://localhost:11434") -> bool:
    import requests
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except:
        return False


def check_chroma_health() -> bool:
    try:
        from database.chroma_db_setup import vector_store_chroma
        vector_store_chroma._collection.count()
        return True
    except:
        return False


def validate_query(query: str, max_length: int = 2000) -> str:
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty")
    query = query.strip()
    if len(query) > max_length:
        query = query[:max_length]
        warn(f"Query truncated to {max_length} characters")
    return query
