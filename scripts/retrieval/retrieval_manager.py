from pathlib import Path
from typing import List, Dict, Optional

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.embeddings.embedder_registry import get_embedder
from scripts.retrieval.base import BaseRetriever, FaissRetriever
from scripts.retrieval.strategies.strategy_registry import STRATEGY_REGISTRY
from scripts.utils.chunk_utils import deduplicate_chunks


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

    def _load_retrievers(self) -> Dict[str, BaseRetriever]:
        """
        Loads a FAISS-based retriever for each document type that has an index.
        """
        retrievers = {}
        doc_types = [
            f.stem for f in self.project.faiss_dir.glob("*.faiss")
            if (self.project.get_metadata_path(f.stem)).exists()
        ]

        for doc_type in doc_types:
            index_path = self.project.get_faiss_path(doc_type)
            metadata_path = self.project.get_metadata_path(doc_type)
            retrievers[doc_type] = FaissRetriever(index_path, metadata_path)

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
        """
        Main entry point to run a retrieval strategy.

        Args:
            query (str): The search query string.
            top_k (int): Number of results to return (after fusion/ranking).
            strategy (str): Name of the retrieval strategy to apply.
            filters (dict): Optional metadata filters (e.g., date range).

        Returns:
            List[Chunk]: Top-matching chunks across all doc types.
        """
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}")

        strategy_fn = STRATEGY_REGISTRY[strategy]
        query_vector = self.embed_query(query)

        results = strategy_fn(
            query_vector=query_vector,
            retrievers=self.retrievers,
            top_k=top_k,
            filters=filters or {}
        )

        return deduplicate_chunks(results, existing_hashes=set(), skip_duplicates=True)
