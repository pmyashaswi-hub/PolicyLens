"""
PolicyLens Hybrid Retrieval Module
Combines FAISS dense search and BM25 sparse search using Reciprocal Rank Fusion (RRF),
with metadata document filtering and optional Cross-Encoder reranking.
"""

from typing import List, Dict, Any, Optional, Set
import logging
import numpy as np

from src.config import (
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_K,
    DEFAULT_TOP_K,
    BGE_QUERY_PREFIX,
    CROSS_ENCODER_MODEL_NAME,
)
from src.index import get_embedding_model, tokenize_text

logger = logging.getLogger(__name__)

# Cached cross-encoder singleton
_CROSS_ENCODER = None


def get_cross_encoder():
    """Lazy loader for cross-encoder reranking model."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading Cross-Encoder model: %s", CROSS_ENCODER_MODEL_NAME)
            _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        except Exception as e:
            logger.warning("Could not load CrossEncoder model: %s", e)
            _CROSS_ENCODER = None
    return _CROSS_ENCODER


def dense_search(
    query: str,
    faiss_index: Any,
    top_k: int = DENSE_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Execute dense vector search against FAISS index.
    Returns list of dicts with chunk index and cosine similarity score.
    """
    embedder = get_embedding_model()
    # Apply BGE instruction prefix to queries
    prefixed_query = BGE_QUERY_PREFIX + query
    query_emb = embedder.encode([prefixed_query], normalize_embeddings=True)
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)

    total_vectors = faiss_index.ntotal
    search_k = min(top_k, total_vectors)
    if search_k <= 0:
        return []

    scores, indices = faiss_index.search(query_emb, search_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx >= 0:
            results.append({
                "chunk_idx": int(idx),
                "dense_score": float(score),
                "dense_rank": rank,
            })
    return results


def bm25_search(
    query: str,
    bm25_index: Any,
    top_k: int = BM25_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Execute sparse BM25 search over tokenized documents.
    Returns list of dicts with chunk index and BM25 score.
    """
    tokens = tokenize_text(query)
    if not tokens:
        return []

    scores = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = scores[idx]
        if score > 0:
            results.append({
                "chunk_idx": int(idx),
                "bm25_score": float(score),
                "bm25_rank": rank,
            })
    return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge dense and sparse ranked lists using Reciprocal Rank Fusion:
    RRF_score(d) = sum( 1 / (k + rank_i(d)) )
    """
    candidates: Dict[int, Dict[str, Any]] = {}

    for item in dense_results:
        idx = item["chunk_idx"]
        candidates[idx] = {
            "chunk_idx": idx,
            "dense_score": item["dense_score"],
            "dense_rank": item["dense_rank"],
            "bm25_score": 0.0,
            "bm25_rank": None,
            "fused_score": 1.0 / (k + item["dense_rank"]),
        }

    for item in bm25_results:
        idx = item["chunk_idx"]
        bm25_rrf = 1.0 / (k + item["bm25_rank"])
        if idx in candidates:
            candidates[idx]["bm25_score"] = item["bm25_score"]
            candidates[idx]["bm25_rank"] = item["bm25_rank"]
            candidates[idx]["fused_score"] += bm25_rrf
        else:
            candidates[idx] = {
                "chunk_idx": idx,
                "dense_score": 0.0,
                "dense_rank": None,
                "bm25_score": item["bm25_score"],
                "bm25_rank": item["bm25_rank"],
                "fused_score": bm25_rrf,
            }

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: x["fused_score"],
        reverse=True,
    )
    return sorted_candidates


def retrieve_hybrid(
    query: str,
    faiss_index: Any,
    bm25_index: Any,
    metadata: List[Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    filter_docs: Optional[List[str]] = None,
    use_reranker: bool = False,
    query_variants: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Main hybrid retrieval function:
    1. Multi-query dense & sparse candidate retrieval.
    2. Reciprocal Rank Fusion across runs.
    3. Document metadata filtering.
    4. Optional Cross-Encoder reranking.
    5. Returns top_k enriched chunk dictionaries.
    """
    all_queries = [query]
    if query_variants:
        for qv in query_variants:
            if qv.strip() and qv not in all_queries:
                all_queries.append(qv.strip())

    # Pool candidates across all query variants
    aggregated_dense: Dict[int, Dict[str, Any]] = {}
    aggregated_bm25: Dict[int, Dict[str, Any]] = {}

    for q in all_queries:
        d_res = dense_search(q, faiss_index, top_k=DENSE_TOP_K)
        for item in d_res:
            idx = item["chunk_idx"]
            if idx not in aggregated_dense or item["dense_score"] > aggregated_dense[idx]["dense_score"]:
                aggregated_dense[idx] = item

        b_res = bm25_search(q, bm25_index, top_k=BM25_TOP_K)
        for item in b_res:
            idx = item["chunk_idx"]
            if idx not in aggregated_bm25 or item["bm25_score"] > aggregated_bm25[idx]["bm25_score"]:
                aggregated_bm25[idx] = item

    fused = reciprocal_rank_fusion(
        list(aggregated_dense.values()),
        list(aggregated_bm25.values()),
        k=RRF_K,
    )

    # Apply metadata filtering if specified
    filtered_results: List[Dict[str, Any]] = []
    filter_set = set(filter_docs) if filter_docs else None

    for item in fused:
        idx = item["chunk_idx"]
        if idx >= len(metadata):
            continue
        chunk_meta = metadata[idx]
        if filter_set and chunk_meta.get("doc_name") not in filter_set:
            continue

        item["chunk"] = chunk_meta
        item["rerank_score"] = None
        filtered_results.append(item)

    # Optional Cross-Encoder Reranking
    if use_reranker and filtered_results:
        reranker = get_cross_encoder()
        if reranker is not None:
            # Score top candidates (e.g. top 15)
            candidates_to_rerank = filtered_results[:15]
            pairs = [(query, item["chunk"]["text"]) for item in candidates_to_rerank]
            try:
                rerank_scores = reranker.predict(pairs)
                for item, r_score in zip(candidates_to_rerank, rerank_scores):
                    item["rerank_score"] = float(r_score)

                # Sort by rerank score descending
                candidates_to_rerank.sort(key=lambda x: x["rerank_score"], reverse=True)
                # Recombine with any remainder
                filtered_results = candidates_to_rerank + filtered_results[15:]
            except Exception as e:
                logger.error("Reranking failed: %s", e)

    return filtered_results[:top_k]
