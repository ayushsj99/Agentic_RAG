import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.tools import tool
from database.database_config import vector_store, DATABASE_BACKEND
from models.ollama_emb import ollama_embeddings
from models.ollama_LLM import ollama_model
from backend.agent_logger import log
from backend.exceptions import RetrieverError, retry

FETCH_K = 20
FINAL_K = 5
LAMBDA_MULT = 0.5
RRF_K = 60
MIN_RERANK_SCORE = 0.1
MAX_WORKERS = 3

USE_MMR = True
USE_MULTI_QUERY = True
USE_HYDE = True
USE_RERANK = True
USE_HYBRID = True  # Use Milvus native hybrid search (dense + BM25)

_reranker = None
_milvus_ranker = None


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


def _get_milvus_ranker():
    global _milvus_ranker
    if _milvus_ranker is None:
        try:
            from langchain_milvus import WeightedRanker
            _milvus_ranker = WeightedRanker(0.6, 0.4)  # 60% dense, 40% sparse (BM25)
            log("retriever", "Loaded Milvus WeightedRanker for hybrid search")
        except ImportError:
            log("retriever", "WeightedRanker not available, using default")
            _milvus_ranker = False
    return _milvus_ranker if _milvus_ranker else None


@retry(max_attempts=2, delay=0.5, exceptions=(Exception,))
def _hybrid_search(query: str, k: int = FINAL_K):
    ranker = _get_milvus_ranker()
    if ranker:
        return vector_store.similarity_search(query, k=k, ranker=ranker)
    return vector_store.similarity_search(query, k=k)


@retry(max_attempts=2, delay=0.5, exceptions=(Exception,))
def _mmr_search(query: str, k: int = FINAL_K, fetch_k: int = FETCH_K):
    return vector_store.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=LAMBDA_MULT
    )


@retry(max_attempts=2, delay=0.5, exceptions=(Exception,))
def _similarity_search_with_scores(query: str, k: int = 10):
    return vector_store.similarity_search_with_relevance_scores(query, k=k)


@retry(max_attempts=2, delay=0.5, exceptions=(Exception,))
def _similarity_search(query: str, k: int = 10):
    return vector_store.similarity_search(query, k=k)


def _generate_query_variants(query: str, n: int = 3) -> list[str]:
    prompt = (
        f"You are an expert at reformulating search queries to find relevant documents.\n"
        f"Given the original query, generate {n} DIFFERENT search queries that:\n"
        f"1. Use synonyms and alternative phrasings\n"
        f"2. Focus on different aspects of the question\n"
        f"3. Vary between specific and general formulations\n\n"
        f"Original query: {query}\n\n"
        f"Write exactly {n} alternative queries, one per line. No numbering or explanations."
    )
    try:
        response = ollama_model.invoke([{"role": "user", "content": prompt}])
        lines = response.content.strip().split('\n')
        variants = [q.strip() for q in lines if q.strip() and len(q.strip()) > 5]
        variants = variants[:n]  # Don't include original - it's searched separately
        log("retriever", f"Generated {len(variants)} query variants")
        return variants
    except Exception as e:
        log("retriever", f"Multi-query generation failed: {e}")
        return []


def _generate_hypothetical_doc(query: str) -> str:
    prompt = (
        f"You are a knowledgeable assistant. Write a short, factual paragraph (2-3 sentences) "
        f"that directly and specifically answers this question:\n\n"
        f"Question: {query}\n\n"
        f"Write as if you are quoting from an authoritative document. "
        f"Include specific details, names, numbers, or facts that would appear in such a document. "
        f"Do not say 'I don't know' - make a reasonable educated guess."
    )
    try:
        response = ollama_model.invoke([{"role": "user", "content": prompt}])
        hypothetical = response.content.strip()
        log("retriever", f"Generated HyDE doc: {hypothetical[:100]}...")
        return hypothetical
    except Exception as e:
        log("retriever", f"HyDE generation failed: {e}")
        return ""


def _reciprocal_rank_fusion(ranked_lists: list[list], k: int = RRF_K) -> list:
    doc_scores = {}
    doc_objects = {}
    
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_hash in doc_scores:
                doc_scores[doc_hash] += rrf_score
            else:
                doc_scores[doc_hash] = rrf_score
                doc_objects[doc_hash] = doc
    
    sorted_hashes = sorted(doc_scores.keys(), key=lambda h: doc_scores[h], reverse=True)
    fused_docs = [doc_objects[h] for h in sorted_hashes]
    
    log("retriever", f"RRF fused {sum(len(rl) for rl in ranked_lists)} docs -> {len(fused_docs)} unique")
    return fused_docs


def _rerank_docs(query: str, docs: list, top_k: int = FINAL_K) -> list:
    reranker = _get_reranker()
    if not reranker or not docs:
        return docs[:top_k]
    
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    filtered = [(doc, score) for doc, score in scored_docs if score >= MIN_RERANK_SCORE]
    
    if not filtered:
        log("retriever", f"All docs below threshold {MIN_RERANK_SCORE}, taking top {top_k}")
        filtered = scored_docs[:top_k]
    
    reranked = [doc for doc, score in filtered[:top_k]]
    top_scores = [f"{score:.3f}" for _, score in filtered[:top_k]]
    log("retriever", f"Reranked: {len(docs)} -> {len(reranked)} (scores: {', '.join(top_scores)})")
    return reranked


def _parallel_search(queries: list[str], search_fn, k: int = 5) -> list[list]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {executor.submit(search_fn, q, k): q for q in queries}
        for future in as_completed(future_to_query):
            try:
                docs = future.result()
                results.append(docs)
            except Exception as e:
                log("retriever", f"Parallel search failed for one query: {e}")
                results.append([])
    return results


def _advanced_retrieve(query: str) -> list:
    ranked_lists = []
    
    # Select search function based on backend and settings
    if DATABASE_BACKEND == "milvus" and USE_HYBRID:
        search_fn = _hybrid_search
        log("retriever", "Using Milvus hybrid search (dense + BM25)")
    elif USE_MMR:
        search_fn = _mmr_search
    else:
        search_fn = _similarity_search
    
    original_docs = search_fn(query, k=FETCH_K if not USE_MMR else FINAL_K * 2)
    ranked_lists.append(original_docs)
    log("retriever", f"Original query retrieved {len(original_docs)} docs")
    
    if USE_HYDE:
        hypothetical = _generate_hypothetical_doc(query)
        if hypothetical and hypothetical != query:
            hyde_docs = search_fn(hypothetical, k=FINAL_K * 2)
            ranked_lists.append(hyde_docs)
            log("retriever", f"HyDE retrieved {len(hyde_docs)} docs")
    
    if USE_MULTI_QUERY:
        variants = _generate_query_variants(query, n=3)
        if variants:
            variant_results = _parallel_search(variants, search_fn, k=FINAL_K)
            ranked_lists.extend(variant_results)
            total_variant_docs = sum(len(r) for r in variant_results)
            log("retriever", f"Multi-query ({len(variants)} variants) retrieved {total_variant_docs} docs")
    
    if len(ranked_lists) > 1:
        fused_docs = _reciprocal_rank_fusion(ranked_lists)
    else:
        fused_docs = ranked_lists[0] if ranked_lists else []
    
    if USE_RERANK and len(fused_docs) > FINAL_K:
        final_docs = _rerank_docs(query, fused_docs, top_k=FINAL_K)
    else:
        final_docs = fused_docs[:FINAL_K]
    
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