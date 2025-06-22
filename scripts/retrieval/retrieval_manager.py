from pathlib import Path

retrieval_manager_code = """
from typing import List, Dict, Optional
from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.base import BaseRetriever, FaissRetriever
from scripts.retrieval.strategies import STRATEGY_REGISTRY
from scripts.retrieval.utils import dedupe_chunks

class RetrievalManager:
    \"\"\"
    Central manager for querying over multiple retrievers and applying retrieval strategies.

    Usage:
        rm = RetrievalManager(project)
        results = rm.retrieve("alma analytics", top_k=10, strategy="late_fusion")
    \"\"\"

    def __init__(self, project: ProjectManager):
        self.project = project
        self.retrievers: Dict[str, BaseRetriever] = self._load_retrievers()

    def _load_retrievers(self) -> Dict[str, BaseRetriever]:
        \"\"\"
        Loads a FAISS-based retriever for each document type that has an index.
        In the future, this could dynamically load BM25 or hybrid retrievers too.
        \"\"\"
        retrievers = {}

        # TODO: Read doc types from config or available FAISS files
        doc_types = ["pdf", "docx", "pptx", "eml", "xlsx", "csv"]  # hardcoded for now

        for doc_type in doc_types:
            index_path = self.project.get_faiss_path(doc_type)
            metadata_path = self.project.get_metadata_path(doc_type)
            if index_path.exists() and metadata_path.exists():
                retrievers[doc_type] = FaissRetriever(index_path, metadata_path)
            else:
                # TODO: optionally log or skip missing retrievers gracefully
                continue

        return retrievers

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "late_fusion",
        filters: Optional[Dict] = None
    ) -> List[Chunk]:
        \"\"\"
        Main entry point to run a retrieval strategy.

        Args:
            query (str): The search query.
            top_k (int): Number of results to return (after fusion/ranking).
            strategy (str): Name of the retrieval strategy to apply.
            filters (dict): Optional metadata filters (e.g., date range).

        Returns:
            List[Chunk]: Top-matching chunks with metadata and similarity scores.
        \"\"\"
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}")

        strategy_fn = STRATEGY_REGISTRY[strategy]

        results = strategy_fn(
            query=query,
            retrievers=self.retrievers,
            top_k=top_k,
            filters=filters or {}
        )

        # Optional post-filtering (deduplication, score threshold, etc.)
        return dedupe_chunks(results)
"""

retrieval_path = Path("/mnt/data/retrieval_manager.py")
retrieval_path.write_text(retrieval_manager_code.strip(), encoding="utf-8")
retrieval_path
