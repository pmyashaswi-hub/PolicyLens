"""
PolicyLens Indexing Module
Builds, serializes, and loads FAISS dense index (L2-normalized cosine inner product)
and BM25 sparse index over structured document chunks.
"""

import os
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.config import (
    FAISS_INDEX_PATH,
    BM25_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
)
from src.ingest import DocumentChunk, ingest_all_documents

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

# Cached singletons
_EMBEDDING_MODEL = None


def get_embedding_model():
    """Lazy loader for SentenceTransformer embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.warning("Could not load SentenceTransformer embedding model: %s", e)
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25 matching."""
    import re
    return [token.lower() for token in re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", text) if len(token) > 1]


def build_indices(chunks: List[DocumentChunk], force_rebuild: bool = False) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """
    Construct FAISS dense index and BM25 sparse index, persisting both to disk.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    import faiss
    from rank_bm25 import BM25Okapi

    if not chunks:
        raise ValueError("Cannot build index on empty chunk list.")

    print(f"\n🔨 Building indices for {len(chunks)} document chunks...")

    # 1. Sparse Index (BM25)
    print("  • Tokenizing corpus for BM25...")
    tokenized_corpus = [tokenize_text(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  ✓ BM25 index persisted to {BM25_INDEX_PATH}")

    # 2. Dense Index (FAISS)
    index = None
    try:
        import faiss
        print("  • Computing dense embeddings via BAAI/bge-small-en-v1.5...")
        embedder = get_embedding_model()
        if embedder:
            texts = [c.text for c in chunks]
            embeddings = embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

            faiss.write_index(index, str(FAISS_INDEX_PATH))
            print(f"  ✓ FAISS index ({index.ntotal} vectors, {dimension}-dim) persisted to {FAISS_INDEX_PATH}")
        else:
            print("  ⚠ Embedding model unavailable, dense index build skipped.")
    except Exception as e:
        print(f"  ⚠ Dense index creation skipped: {e}")
        index = None

    # 3. Chunks Metadata
    metadata_list = [c.to_dict() for c in chunks]
    with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata dictionary persisted to {CHUNKS_METADATA_PATH}")

    print("✨ Index build complete!\n")
    return index, bm25, metadata_list


def indices_exist() -> bool:
    """Check if pre-built indices exist on disk."""
    return (
        BM25_INDEX_PATH.exists()
        and CHUNKS_METADATA_PATH.exists()
    )


def load_indices() -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """
    Load persisted FAISS, BM25, and metadata from disk.
    If not found, builds them automatically from data/.
    """
    if not indices_exist():
        print("Indices not found on disk. Ingesting documents and building indices now...")
        chunks = ingest_all_documents()
        return build_indices(chunks)

    index = None
    if FAISS_INDEX_PATH.exists():
        try:
            import faiss
            logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
            index = faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception as e:
            logger.warning("FAISS index could not be loaded (falling back to BM25): %s", e)
            index = None

    bm25 = None
    if BM25_INDEX_PATH.exists():
        try:
            logger.info("Loading BM25 index from %s", BM25_INDEX_PATH)
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25 = pickle.load(f)
        except Exception as e:
            logger.error("BM25 index loading failed: %s", e)

    metadata = []
    if CHUNKS_METADATA_PATH.exists():
        try:
            logger.info("Loading metadata from %s", CHUNKS_METADATA_PATH)
            with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error("Chunks metadata loading failed: %s", e)

    return index, bm25, metadata


def rebuild_index() -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """Force re-ingest and re-indexing."""
    chunks = ingest_all_documents()
    return build_indices(chunks, force_rebuild=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="PolicyLens Index Builder")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index from scratch")
    args = parser.parse_args()

    if args.rebuild or not indices_exist():
        rebuild_index()
    else:
        load_indices()
        print("Existing index loaded successfully.")
