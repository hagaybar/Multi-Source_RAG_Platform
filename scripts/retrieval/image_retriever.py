import json
from typing import List
import numpy as np
import faiss

from scripts.chunking.models import ImageChunk


class ImageRetriever:
    def __init__(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self.metadata_path = metadata_path
        self.index_path = index_path

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"ImageRetriever: metadata count ({len(self.metadata)}) does not match FAISS entries ({self.index.ntotal})."
            )

    def _load_metadata(self, path: str) -> List[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        return records

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[ImageChunk]:
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector.astype("float32"), axis=0)
        elif query_vector.dtype != np.float32:
            query_vector = query_vector.astype("float32")

        distances, indices = self.index.search(query_vector, top_k)
        top_chunks = []

        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            top_chunks.append(
                ImageChunk(
                    id=meta["id"],
                    description=meta["description"],
                    meta={k: v for k, v in meta.items() if k not in ("id", "description", "text")},
                )
            )
            top_chunks[-1].meta["similarity"] = float(score)
            top_chunks[-1].meta["_retriever"] = "image"

        return top_chunks
