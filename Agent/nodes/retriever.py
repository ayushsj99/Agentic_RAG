import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from langchain.tools import tool
from database.database_config import vector_store, DATABASE_BACKEND
from models.ollama_emb import ollama_embeddings
from models.ollama_LLM import ollama_model
from backend.agent_logger import log
from backend.exceptions import RetrieverError, retry

# Configuration
FETCH_K = 20
FINAL_K = 5
LAMBDA_MULT = 0.5
RRF_K = 60
MIN_RERANK_SCORE = -3.0
MAX_WORKERS = 3

USE_MMR = True
USE_MULTI_QUERY = True
USE_HYDE = False
USE_RERANK = True
USE_HYBRID = True

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
            _milvus_ranker = WeightedRanker(0.6, 0.4)
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
def _similarity_search(query: str, k: int = 10):
    return vector_store.similarity_search(query, k=k)


@lru_cache(maxsize=100)
def _classify_query_complexity(query: str) -> str:
    """Classify query as simple, medium, or complex."""
    prompt = (
        "Classify this search query as 'simple', 'medium', or 'complex':\n\n"
        "- simple: Single factual lookup (one specific fact, name, number, date)\n"
        "  Examples: 'what is the roll number', 'who is the author', 'what is the price'\n\n"
        "- medium: Concept explanation, process description, single cohesive answer\n"
        "  Examples: 'how does the system work', 'explain the workflow', 'what are the benefits'\n\n"
        "- complex: List/comparison/synthesis questions needing multiple chunks\n"
        "  Examples: 'what all X are covered', 'list requirements', 'compare X and Y',\n"
        "           'what is covered and not covered', 'analyze', 'summarize all'\n\n"
        f"Query: {query}\n\n"
        "Respond with ONLY one word: simple, medium, or complex"
    )
    
    try:
        response = ollama_model.invoke([{"role": "user", "content": prompt}])
        complexity = response.content.strip().lower()
        if complexity in ['simple', 'medium', 'complex']:
            log("retriever", f"Query classified as: {complexity.upper()}")
            return complexity
        else:
            log("retriever", f"Invalid classification, defaulting to medium")
            return "medium"
    except Exception as e:
        log("retriever", f"Classification failed: {e}, defaulting to medium")
        return "medium"


def _get_optimal_k(complexity: str) -> tuple[int, int]:
    """Return (fetch_k, final_k) based on query complexity."""
    if complexity == "simple":
        return 10, 8
    elif complexity == "medium":
        return 20, 15
    else:
        return 30, 15  # Increased for complex queries


def _is_list_query(query: str) -> bool:
    """Detect if query asks for multiple items/list."""
    list_keywords = ['all', 'list', 'covered', 'not covered', 'which', 'requirements', 
                     'objectives', 'what are', 'enumerate', 'every']
    return any(keyword in query.lower() for keyword in list_keywords)


def _generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Generate alternative query formulations."""
    prompt = (
        f"Generate {n} alternative search queries for:\n{query}\n\n"
        f"Use synonyms and different phrasings. One per line, no numbering."
    )
    try:
        response = ollama_model.invoke([{"role": "user", "content": prompt}])
        lines = response.content.strip().split('\n')
        variants = [q.strip() for q in lines if q.strip() and len(q.strip()) > 5][:n]
        log("retriever", f"Generated {len(variants)} query variants")
        return variants
    except Exception as e:
        log("retriever", f"Multi-query generation failed: {e}")
        return []


def _reciprocal_rank_fusion(ranked_lists: list[list], k: int = RRF_K) -> list:
    """Fuse multiple ranked lists using RRF."""
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


def _rerank_docs(query: str, docs: list, top_k: int = FINAL_K, force_diversity: bool = False) -> tuple[list, bool]:
    """Rerank documents using cross-encoder. Returns (docs, is_confident)."""
    reranker = _get_reranker()
    if not reranker or not docs:
        return docs[:top_k], True
    
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    best_score = scored_docs[0][1] if scored_docs else -100
    is_confident = best_score >= MIN_RERANK_SCORE
    
    # For list/multi-part queries, return top_k regardless of threshold
    if force_diversity:
        reranked = [doc for doc, score in scored_docs[:top_k]]
        log("retriever", f"Diversity mode: returning top {len(reranked)} docs")
    else:
        filtered = [(doc, score) for doc, score in scored_docs if score >= MIN_RERANK_SCORE]
        
        if not filtered:
            log("retriever", f"All docs below threshold {MIN_RERANK_SCORE}, best={best_score:.3f}")
            filtered = scored_docs[:top_k]
        
        reranked = [doc for doc, score in filtered[:top_k]]
    
    top_scores = [f"{score:.3f}" for doc, score in scored_docs[:len(reranked)]]
    log("retriever", f"Reranked: {len(docs)} -> {len(reranked)} (scores: {', '.join(top_scores)})")
    
    return reranked, is_confident


def _parallel_search(queries: list[str], search_fn, k: int = 5) -> list[list]:
    """Execute multiple searches in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {executor.submit(search_fn, q, k): q for q in queries}
        for future in as_completed(future_to_query):
            try:
                docs = future.result()
                results.append(docs)
            except Exception as e:
                log("retriever", f"Parallel search failed: {e}")
                results.append([])
    return results


def _simple_retrieve(query: str, fetch_k: int, final_k: int) -> tuple[list, bool]:
    """Simple retrieval - basic search only, no reranking."""
    log("retriever", "Using SIMPLE retrieval (basic search, no reranking)")
    
    if DATABASE_BACKEND == "milvus" and USE_HYBRID:
        docs = _hybrid_search(query, k=final_k)
        log("retriever", f"Hybrid search returned {len(docs)} docs")
    else:
        docs = _similarity_search(query, k=final_k)
        log("retriever", f"Similarity search returned {len(docs)} docs")
    
    return docs, True


def _medium_retrieve(query: str, fetch_k: int, final_k: int) -> tuple[list, bool]:
    """Medium retrieval - hybrid/MMR + light reranking."""
    log("retriever", "Using MEDIUM retrieval (hybrid + reranking)")
    
    if DATABASE_BACKEND == "milvus" and USE_HYBRID:
        docs = _hybrid_search(query, k=fetch_k)
    else:
        docs = _mmr_search(query, k=final_k, fetch_k=fetch_k)
    
    # Detect list queries
    is_list = _is_list_query(query)
    if is_list:
        final_k = min(final_k + 2, 10)
        log("retriever", f"List query detected, increasing final_k to {final_k}")
    
    if USE_RERANK and len(docs) > final_k:
        docs, is_confident = _rerank_docs(query, docs, top_k=final_k, force_diversity=is_list)
    else:
        docs = docs[:final_k]
        is_confident = True
    
    return docs, is_confident


def _complex_retrieve(query: str, fetch_k: int, final_k: int) -> tuple[list, bool]:
    """Complex retrieval - multi-query + fusion + reranking."""
    log("retriever", "Using COMPLEX retrieval (multi-query + fusion + reranking)")
    
    ranked_lists = []
    
    if DATABASE_BACKEND == "milvus" and USE_HYBRID:
        search_fn = _hybrid_search
    elif USE_MMR:
        search_fn = _mmr_search
    else:
        search_fn = _similarity_search
    
    # Original query
    original_docs = search_fn(query, k=fetch_k)
    ranked_lists.append(original_docs)
    log("retriever", f"Original query: {len(original_docs)} docs")
    
    # Multi-query expansion
    if USE_MULTI_QUERY:
        variants = _generate_query_variants(query, n=3)
        if variants:
            variant_results = _parallel_search(variants, search_fn, k=final_k)
            ranked_lists.extend(variant_results)
            log("retriever", f"Multi-query: {sum(len(r) for r in variant_results)} docs")
    
    # Fusion
    if len(ranked_lists) > 1:
        fused_docs = _reciprocal_rank_fusion(ranked_lists)
    else:
        fused_docs = ranked_lists[0] if ranked_lists else []
    
    # Detect list queries and increase final_k
    is_list = _is_list_query(query)
    if is_list:
        final_k = min(final_k + 3, 12)
        log("retriever", f"List query detected, increasing final_k to {final_k}")
    
    # Reranking
    if USE_RERANK and len(fused_docs) > final_k:
        final_docs, is_confident = _rerank_docs(query, fused_docs, top_k=final_k, force_diversity=is_list)
    else:
        final_docs = fused_docs[:final_k]
        is_confident = True
    
    return final_docs, is_confident


def _adaptive_retrieve(query: str) -> tuple[list, bool]:
    """Route to appropriate retrieval strategy based on query complexity."""
    
    complexity = _classify_query_complexity(query)
    fetch_k, final_k = _get_optimal_k(complexity)
    log("retriever", f"Optimal K values: fetch={fetch_k}, final={final_k}")
    
    try:
        if complexity == "simple":
            docs, is_confident = _simple_retrieve(query, fetch_k, final_k)
        elif complexity == "medium":
            docs, is_confident = _medium_retrieve(query, fetch_k, final_k)
        else:
            docs, is_confident = _complex_retrieve(query, fetch_k, final_k)
        
        return docs, is_confident
        
    except Exception as e:
        log("retriever", f"{complexity.title()} retrieval failed, falling back: {e}")
        try:
            docs = _similarity_search(query, k=final_k)
            return docs, False
        except Exception as e2:
            log("retriever", f"Fallback failed: {e2}")
            raise RetrieverError(f"All retrieval strategies failed: {e2}")


@tool()
def retrieve_context(query: str):
    """Retrieve information to help answer a query using adaptive retrieval strategy."""
    log("retriever", f"Searching for: {query[:120]}")
    
    try:
        retrieved_docs, is_confident = _adaptive_retrieve(query)
    except Exception as e:
        log("retriever", f"Adaptive retrieval failed: {e}")
        raise RetrieverError(f"Document retrieval failed: {e}")
    
    log("retriever", f"Final result: {len(retrieved_docs)} documents (confident={is_confident})")
    
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
        log("retriever", f"  Doc {i+1} [{source}]: {doc.page_content[:100]}...")
    
    serialized = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    
    if not is_confident:
        log("retriever", "Low confidence retrieval - flagging for query rewrite")
        serialized = "[LOW_CONFIDENCE_RETRIEVAL]\n\n" + serialized
    
    return serialized if serialized else "No relevant documents found."


retriever_tool = retrieve_context