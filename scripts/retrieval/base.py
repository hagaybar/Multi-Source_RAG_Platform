from abc import ABC, abstractmethod
from typing import List
import json
import faiss
import numpy as np
from pathlib import Path

from scripts.chunking.models import Chunk
from scripts.api_clients.embedder import get_embedder  # Assumes shared embedder utility

class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.
    """

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Chunk]:
        pass


class FaissRetriever(BaseRetriever):
    """
    FAISS-based retriever for a specific doc type.
    """

    def __init__(self, index_path: Path, metadata_path: Path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.read_index(str(index_path))
        self.metadata = self._load_metadata()
        self.embedder = get_embedder()  # Use shared embedding model

    def _load_metadata(self) -> List[dict]:
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def search(self, query: str, top_k: int) -> List[Chunk]:
        query_vec = self.embedder.encode([query])[0].astype("float32")
        scores, indices = self.index.search(np.array([query_vec]), top_k)
        results: List[Chunk] = []

        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            score = float(scores[0][i])

            # Add score to metadata for fusion
            meta["score"] = score

            # Reconstruct Chunk object (text is not stored in FAISS, only metadata)
            results.append(Chunk(
                doc_id=meta.get("doc_id", "unknown"),
                text=meta.get("text", ""),  # Optional: enrich with full text if needed
                token_count=meta.get("token_count", 0),
                meta=meta,
                id=meta.get("id", None)
            ))

        return results