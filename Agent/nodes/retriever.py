import hashlib
from langchain.tools import tool
from database.chroma_db_setup import vector_store_chroma
from models.ollama_emb import ollama_embeddings
from models.ollama_LLM import ollama_model
from backend.agent_logger import log
from backend.exceptions import RetrieverError, retry

FETCH_K = 20
FINAL_K = 5
LAMBDA_MULT = 0.5
USE_MMR = True
USE_MULTI_QUERY = True
USE_HYDE = True
USE_RERANK = True

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            log("retriever", "Loaded cross-encoder reranker")
        except ImportError:
            log("retriever", "sentence-transformers not installed, skipping rerank")
            _reranker = False
    return _reranker if _reranker else None


@retry(max_attempts=2, delay=0.5, exceptions=(Exception,))
def _mmr_search(query: str, k: int = FINAL_K, fetch_k: int = FETCH_K):
    return vector_store_chroma.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=LAMBDA_MULT
    )


@retry(max_attempts=2, delay=0.5, exceptions=(Exception,))
def _similarity_search(query: str, k: int = 10):
    return vector_store_chroma.similarity_search(query, k=k)


def _generate_query_variants(query: str, n: int = 3) -> list[str]:
    prompt = (
        f"Generate {n} different search queries to find information about:\n"
        f"Original: {query}\n\n"
        f"Write {n} alternative queries, one per line. No numbering."
    )
    try:
        response = ollama_model.invoke([{"role": "user", "content": prompt}])
        variants = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        variants = [query] + variants[:n]
        log("retriever", f"Generated {len(variants)} query variants")
        return variants
    except Exception as e:
        log("retriever", f"Multi-query generation failed: {e}")
        return [query]


def _generate_hypothetical_doc(query: str) -> str:
    prompt = (
        f"Write a short paragraph (2-3 sentences) that directly answers this question:\n"
        f"{query}\n\n"
        f"Write as if you are quoting from a document that contains the answer."
    )
    try:
        response = ollama_model.invoke([{"role": "user", "content": prompt}])
        hypothetical = response.content.strip()
        log("retriever", f"Generated HyDE doc: {hypothetical[:100]}...")
        return hypothetical
    except Exception as e:
        log("retriever", f"HyDE generation failed: {e}")
        return query


def _rerank_docs(query: str, docs: list, top_k: int = FINAL_K) -> list:
    reranker = _get_reranker()
    if not reranker or not docs:
        return docs[:top_k]
    
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    reranked = [doc for doc, score in scored_docs[:top_k]]
    log("retriever", f"Reranked {len(docs)} docs -> top {len(reranked)}")
    return reranked


def _deduplicate(docs: list) -> list:
    seen = set()
    unique = []
    for doc in docs:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


def _advanced_retrieve(query: str) -> list:
    all_docs = []
    
    if USE_HYDE:
        hypothetical = _generate_hypothetical_doc(query)
        hyde_docs = _mmr_search(hypothetical, k=FINAL_K) if USE_MMR else _similarity_search(hypothetical, k=FINAL_K)
        all_docs.extend(hyde_docs)
        log("retriever", f"HyDE retrieved {len(hyde_docs)} docs")
    
    if USE_MULTI_QUERY:
        variants = _generate_query_variants(query, n=3)
        for v in variants:
            v_docs = _mmr_search(v, k=3) if USE_MMR else _similarity_search(v, k=3)
            all_docs.extend(v_docs)
        log("retriever", f"Multi-query retrieved {len(all_docs)} total docs")
    else:
        direct_docs = _mmr_search(query, k=FETCH_K) if USE_MMR else _similarity_search(query, k=FETCH_K)
        all_docs.extend(direct_docs)
    
    unique_docs = _deduplicate(all_docs)
    log("retriever", f"After dedup: {len(unique_docs)} unique docs")
    
    if USE_RERANK and len(unique_docs) > FINAL_K:
        final_docs = _rerank_docs(query, unique_docs, top_k=FINAL_K)
    else:
        final_docs = unique_docs[:FINAL_K]
    
    return final_docs


@tool()
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    log("retriever", f"Searching for: {query[:120]}")
    
    try:
        retrieved_docs = _advanced_retrieve(query)
    except Exception as e:
        log("retriever", f"Advanced retrieval failed, falling back: {e}")
        try:
            retrieved_docs = _similarity_search(query, k=FINAL_K)
        except Exception as e2:
            log("retriever", f"Fallback also failed: {e2}")
            raise RetrieverError(f"Document retrieval failed: {e2}")
    
    log("retriever", f"Final result: {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
        log("retriever", f"  Doc {i+1} [{source}]: {doc.page_content[:100]}...")
    
    serialized = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    return serialized if serialized else "No relevant documents found."


retriever_tool = retrieve_context