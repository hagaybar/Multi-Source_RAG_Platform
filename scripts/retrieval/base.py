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
    @abstractmethod
    def retrieve_vector(self, query_vector: List[float], top_k: int, filters: Dict) -> List[Chunk]:
        pass

class FaissRetriever(BaseRetriever):
    def __init__(self, index_path: Path, metadata_path: Path):
        print(f"DEBUG: FaissRetriever init for {index_path.name}")
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.read_index(str(index_path))
        print(f"DEBUG: Loaded FAISS index with {self.index.ntotal} vectors")
        self.metadata = self._load_metadata()
        print(f"DEBUG: Loaded {len(self.metadata)} metadata records")
        assert self.index.ntotal == len(self.metadata), (
            f"Mismatch: index has {self.index.ntotal}, metadata has {len(self.metadata)}"
        )

    def retrieve_vector(self, query_vector: list[float], top_k: int, filters: dict) -> list[Chunk]:
        print(f"[DEBUG] retrieve_vector() called for {self.index_path.name}")
        print(f"[DEBUG] Query vector length: {len(query_vector)}")

        try:
            query_np = np.array([query_vector], dtype="float32")
            distances, indices = self.index.search(query_np, top_k)

            print(f"[DEBUG] FAISS distances: {distances.tolist()}")
            print(f"[DEBUG] FAISS indices: {indices.tolist()}")

            results = []
            for score, idx in zip(distances[0], indices[0]):
                print(f"[DEBUG] Scoring idx={idx}, score={score}")
                if idx < 0 or idx >= len(self.metadata):
                    print(f"[DEBUG] Skipping invalid index {idx}")
                    continue

                meta = self.metadata[idx]
                print(f"[DEBUG] Meta keys: {list(meta.keys())}")

                results.append(Chunk(
                    id=meta.get("id", f"chunk-{idx}"),
                    doc_id=meta.get("doc_id", "unknown"),
                    text=meta.get("text", "[no text]"),
                    token_count=meta.get("token_count", 0),
                    meta={**meta, "similarity": float(1.0 - score)}
                ))

            print(f"[DEBUG] Returning {len(results)} results from {self.index_path.name}")
            return results

        except Exception as e:
            import traceback
            print(f"[ERROR] retrieve_vector failed for {self.index_path.name}: {e}")
            traceback.print_exc()
            raise



    def _load_metadata(self) -> List[dict]:
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # def search(self, query: str, top_k: int) -> List[Chunk]:
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