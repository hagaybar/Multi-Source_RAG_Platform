# scripts/retrieval/strategies.py

from typing import List, Dict
from scripts.chunking.models import Chunk
from scripts.retrieval.base import BaseRetriever

def late_fusion(query: str, retrievers: Dict[str, BaseRetriever], top_k: int, filters: Dict) -> List[Chunk]:
    candidates: List[Chunk] = []

    for doc_type, retriever in retrievers.items():
        try:
            chunks = retriever.retrieve(query, top_k=top_k, filters=filters)
            for chunk in chunks:
                chunk.meta["_retriever"] = doc_type
            candidates.extend(chunks)
        except Exception as e:
            print(f"[WARN] Skipping {doc_type} due to error: {e}")
            continue

    # Simple sort by similarity (descending)
    candidates.sort(key=lambda c: c.meta.get("similarity", 0), reverse=True)

    return candidates[:top_k]
