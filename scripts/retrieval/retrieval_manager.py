from pathlib import Path
from typing import List, Dict, Optional
import traceback


from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.embeddings.embedder_registry import get_embedder
from scripts.retrieval.base import BaseRetriever, FaissRetriever
from scripts.retrieval.strategies.strategy_registry import STRATEGY_REGISTRY
from scripts.utils.chunk_utils import deduplicate_chunks
from scripts.retrieval.image_retriever import ImageRetriever


class RetrievalManager:
    """
    Central manager for querying over multiple retrievers and applying retrieval strategies.

    Usage:
        rm = RetrievalManager(project)
        results = rm.retrieve("alma analytics", top_k=10, strategy="late_fusion")
    """

    def __init__(self, project: ProjectManager):
        self.project = project
        self.retrievers: Dict[str, BaseRetriever] = self._load_retrievers()
        self.embedder = get_embedder(project)
        image_index = project.output_dir / "image_index.faiss"
        image_meta = project.output_dir / "image_metadata.jsonl"
        if image_index.exists() and image_meta.exists():
            self.image_retriever = ImageRetriever(str(image_index), str(image_meta))
        else:
            self.image_retriever = None

    def _load_retrievers(self) -> Dict[str, BaseRetriever]:
        retrievers = {}
        doc_types = [
            f.stem for f in self.project.faiss_dir.glob("*.faiss")
            if (self.project.get_metadata_path(f.stem)).exists()
        ]

        print(f"DEBUG: Discovered doc_types: {doc_types}")

        for doc_type in doc_types:
            index_path = self.project.get_faiss_path(doc_type)
            metadata_path = self.project.get_metadata_path(doc_type)

            print(f"DEBUG: Loading retriever for {doc_type}")
            print(f"       - FAISS path:    {index_path}")
            print(f"       - Metadata path: {metadata_path}")

            try:
                retrievers[doc_type] = FaissRetriever(index_path, metadata_path)
            except Exception as e:
                print(f"[WARN] Failed to load retriever for {doc_type}: {e}")
                traceback.print_exc()

        return retrievers

    def embed_query(self, query: str) -> List[float]:
        """Returns the embedding vector for the query string."""
        return self.embedder.encode([query])[0]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "late_fusion",
        filters: Optional[Dict] = None
    ) -> List[Chunk]:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}")

        strategy_fn = STRATEGY_REGISTRY[strategy]
        query_vector = self.embed_query(query)

        # Text-based retrieval
        chunk_results = strategy_fn(
            query_vector=query_vector,
            retrievers=self.retrievers,
            top_k=top_k,
            filters=filters or {}
        )

        # Image-based retrieval
        image_results = []
        if self.image_retriever:
            image_results = self.image_retriever.search(query_vector, top_k=top_k)

        # Optional: promote source chunks tied to image matches
        chunk_map = {chunk.id: chunk for chunk in chunk_results}
        for img in image_results:
            chunk_id = img.meta.get("source_chunk_id")
            if chunk_id in chunk_map:
                # Boost score or annotate
                chunk_map[chunk_id].meta["promoted_by_image"] = True
                chunk_map[chunk_id].meta["image_similarity"] = img.meta.get("similarity", 0)

        combined = list(chunk_map.values()) + image_results  # image results can be returned as-is
        return deduplicate_chunks(combined, existing_hashes=set(), skip_duplicates=True)
