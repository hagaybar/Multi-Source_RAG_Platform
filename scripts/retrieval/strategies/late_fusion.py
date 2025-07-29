# scripts/retrieval/strategies/late_fusion.py

from typing import List, Dict
from scripts.chunking.models import Chunk
from scripts.retrieval.base import BaseRetriever


def late_fusion(
    query_vector: List[float], retrievers: Dict[str, BaseRetriever], top_k: int, filters: Dict
) -> List[Chunk]:
    """
    Simple late-fusion strategy: queries each retriever independently,
    collects all results, sorts them by similarity, and returns top-K.

    Args:
        query: User query string.
        retrievers: Mapping of doc_type -> BaseRetriever.
        top_k: Number of global top results to return.
        filters: Optional metadata filters (e.g., date range).

    Returns:
        List of top-K scored chunks across all retrievers.
    """
    candidates: List[Chunk] = []

    for doc_type, retriever in retrievers.items():
        try:
            chunks = retriever.retrieve_vector(query_vector, top_k=top_k, filters=filters)
            for chunk in chunks:
                chunk.meta["_retriever"] = doc_type  # Track where it came from
            candidates.extend(chunks)
        except Exception as e:
            print(f"[WARN] Skipping {doc_type} retriever: {e}")
            continue

    # Global sort by similarity score (descending)
    candidates.sort(key=lambda c: c.meta.get("similarity", 0), reverse=True)

    return candidates[:top_k]
