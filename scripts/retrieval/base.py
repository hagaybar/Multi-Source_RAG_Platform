from abc import ABC, abstractmethod
from typing import List, Dict
import json
import faiss
import numpy as np
from pathlib import Path

from scripts.chunking.models import Chunk
from scripts.embeddings.embedder_registry import get_embedder
# Assumes shared embedder utility

class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.
    """
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Chunk]:
        pass

class FaissRetriever(BaseRetriever):
    def __init__(self, index_path, metadata_path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.read_index(str(index_path))

        # Load metadata into memory
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

        assert self.index.ntotal == len(self.metadata), (
            f"Mismatch: index has {self.index.ntotal} vectors, but metadata has {len(self.metadata)} entries"
        )

    def retrieve_vector(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Dict
    ) -> List[Chunk]:
        """
        Returns top-k matching chunks given a query vector.

        Args:
            query_vector: Embedded query as list of floats.
            top_k: Max number of results to return.
            filters: Metadata filters (currently ignored, placeholder).

        Returns:
            List[Chunk] with metadata and similarity scores.
        """
        # Convert to NumPy array
        query_np = np.array([query_vector]).astype("float32")

        # Search
        distances, indices = self.index.search(query_np, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(Chunk(
                id=meta.get("id", f"chunk-{idx}"),
                doc_id=meta.get("doc_id", "unknown"),
                text=meta.get("text", "[no text]"),
                token_count=meta.get("token_count", 0),
                meta={**meta, "similarity": float(1.0 - score)}  # similarity = 1 - L2
            ))

        return results
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