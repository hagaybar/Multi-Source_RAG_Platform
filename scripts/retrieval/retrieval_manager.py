from typing import List, Dict, Optional
import traceback


from scripts.chunking.models import Chunk
from scripts.utils.logger import LoggerManager
from scripts.core.project_manager import ProjectManager
from scripts.embeddings.embedder_registry import get_embedder
from scripts.retrieval.base import BaseRetriever, FaissRetriever
from scripts.retrieval.strategies.strategy_registry import STRATEGY_REGISTRY
from scripts.utils.chunk_utils import deduplicate_chunks
from scripts.utils.translation_utils import translate_to_english
from scripts.retrieval.image_retriever import ImageRetriever


class RetrievalManager:
    """
    Central manager for querying over multiple retrievers and applying retrieval
    strategies.

    Usage:
        rm = RetrievalManager(project)
        results = rm.retrieve("alma analytics", top_k=10, strategy="late_fusion")
    """

    def __init__(self, project: ProjectManager, run_id: Optional[str] = None):
        self.project = project
        self.config = project.config  # required for 'embedding.translate_query'
        self.run_id = run_id
        self.logger = LoggerManager.get_logger("retrieval", run_id=run_id)
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
            f.stem
            for f in self.project.faiss_dir.glob("*.faiss")
            if (self.project.get_metadata_path(f.stem)).exists()
        ]

        self.logger.debug(f"Discovered doc_types: {doc_types}", extra={"run_id": self.run_id} if self.run_id else {})

        for doc_type in doc_types:
            index_path = self.project.get_faiss_path(doc_type)
            metadata_path = self.project.get_metadata_path(doc_type)

            self.logger.debug(f"Loading retriever for {doc_type}", extra={"run_id": self.run_id, "doc_type": doc_type, "faiss_path": str(index_path), "metadata_path": str(metadata_path)} if self.run_id else {"doc_type": doc_type, "faiss_path": str(index_path), "metadata_path": str(metadata_path)})

            try:
                retrievers[doc_type] = FaissRetriever(index_path, metadata_path)
            except Exception as e:
                self.logger.warning(f"Failed to load retriever for {doc_type}: {e}", extra={"run_id": self.run_id, "doc_type": doc_type} if self.run_id else {"doc_type": doc_type}, exc_info=True)

        return retrievers

    def embed_query(self, query: str) -> List[float]:
        """Returns the embedding vector for the query string."""
        return self.embedder.encode([query])[0]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "late_fusion",
        filters: Optional[Dict] = None,
    ) -> List[Chunk]:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}")

        strategy_fn = STRATEGY_REGISTRY[strategy]
        query_vectors = [self.embed_query(query)]

        # Optional translation for multilingual fallback
        if self.config.get("embedding", {}).get("translate_query", False):
            translated = translate_to_english(query)
            if translated and translated != query:
                self.logger.info(f"Added translated query: {translated}", extra={"run_id": self.run_id, "original_query": query, "translated_query": translated} if self.run_id else {"original_query": query, "translated_query": translated})
                query_vectors.append(self.embed_query(translated))

        all_results = []

        for q_vec in query_vectors:
            chunk_results = strategy_fn(
                query_vector=q_vec,
                retrievers=self.retrievers,
                top_k=top_k,
                filters=filters or {},
            )

            image_results = []
            if self.image_retriever:
                image_results = self.image_retriever.search(q_vec, top_k=top_k)

            all_results.extend(chunk_results)
            all_results.extend(image_results)

        # Promote or enrich matching text chunks using image insights
        chunk_map = {chunk.id: chunk for chunk in all_results if hasattr(chunk, "text")}
        promoted_ids = []

        for chunk in all_results:
            if hasattr(chunk, "description") and hasattr(chunk, "meta"):
                source_id = chunk.meta.get("source_chunk_id")
                if source_id and source_id in chunk_map:
                    parent = chunk_map[source_id]
                    parent.meta.setdefault("image_summaries", []).append(
                        {
                            "image_path": chunk.meta.get("image_path"),
                            "description": chunk.description,
                        }
                    )
                    parent.meta["promoted_by_image"] = True
                    parent.meta["image_similarity"] = chunk.meta.get("similarity", 0)
                    promoted_ids.append(source_id)

        if promoted_ids:
            self.logger.debug(f"Promoted {len(promoted_ids)} text chunks from image hits", extra={"run_id": self.run_id, "promoted_count": len(promoted_ids)} if self.run_id else {"promoted_count": len(promoted_ids)})

        # Only return image chunks that don't have a matching text chunk
        image_results_filtered = [
            chunk
            for chunk in all_results
            if hasattr(chunk, "description") and chunk.meta.get("source_chunk_id") not in chunk_map
        ]

        final = list(chunk_map.values()) + image_results_filtered
        return deduplicate_chunks(final, existing_hashes=set(), skip_duplicates=True)

    def translate_to_english(text: str) -> str:
        # Placeholder translation logic (real version should call a translation service)
        if text.strip().startswith("שלום"):
            return (
                "The university is replacing its encryption certificate. What should we do in Alma?"
            )
        return text
