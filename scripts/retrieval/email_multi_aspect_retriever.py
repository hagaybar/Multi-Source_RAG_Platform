"""
MultiAspectRetriever - Combines multiple retrieval strategies.

This retriever implements a composer pattern that can combine:
- Semantic search (base)
- Thread expansion (retrieve full conversations)
- Temporal filtering (date range)
- Sender filtering (specific person)

The retriever applies filters in a pipeline:
1. Start with semantic search results
2. Apply sender filter (if specified)
3. Apply temporal filter (if specified)
4. Optionally expand to complete threads

Example:
    retriever = MultiAspectRetriever(project)
    intent = {
        "primary_intent": "sender_query",
        "metadata": {"sender": "Alice", "time_range": "last_week"}
    }
    chunks = retriever.retrieve("budget", intent, top_k=15)
    # Returns: Alice's emails about budget from last week
"""

from typing import List, Dict, Optional
from datetime import datetime

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.retrieval.email_thread_retriever import ThreadRetriever
from scripts.retrieval.email_temporal_retriever import TemporalRetriever
from scripts.retrieval.email_sender_retriever import SenderRetriever
from scripts.utils.logger import LoggerManager


class MultiAspectRetriever:
    """
    Combines multiple retrieval strategies for complex email queries.

    This retriever uses a pipeline approach:
    1. Semantic search (always)
    2. Sender filtering (optional)
    3. Temporal filtering (optional)
    4. Thread expansion (optional)
    """

    def __init__(self, project: ProjectManager, run_id: Optional[str] = None):
        """
        Initialize MultiAspectRetriever.

        Args:
            project: ProjectManager instance for the current project
            run_id: Optional run ID for logging
        """
        self.project = project
        self.run_id = run_id
        self.logger = LoggerManager.get_logger(
            "email_multi_aspect_retriever",
            task_paths=project.get_task_paths(),
            run_id=run_id
        )
        self.retrieval_manager = RetrievalManager(project, run_id=run_id)
        self.thread_retriever = ThreadRetriever(project, run_id=run_id)
        self.temporal_retriever = TemporalRetriever(project, run_id=run_id)
        self.sender_retriever = SenderRetriever(project, run_id=run_id)

    def retrieve(
        self,
        query: str,
        intent: Optional[Dict] = None,
        top_k: int = 15,
        doc_type: str = "outlook_eml"
    ) -> List[Chunk]:
        """
        Retrieve emails using multiple aspect filtering.

        Args:
            query: User query for semantic search
            intent: Intent detection result with metadata
                   {
                       "primary_intent": "sender_query",
                       "metadata": {"sender": "Alice", "time_range": "last_week"},
                       "secondary_signals": ["temporal_query"]
                   }
            top_k: Number of chunks to return (default: 15)
            doc_type: Document type to filter (default: outlook_eml)

        Returns:
            List of email chunks matching all specified aspects
        """
        intent = intent or {}
        metadata = intent.get("metadata", {})
        primary_intent = intent.get("primary_intent", "factual_lookup")
        secondary_signals = intent.get("secondary_signals", [])

        self.logger.info(
            f"MultiAspectRetriever: query='{query}', intent={primary_intent}, metadata={metadata}",
            extra={
                "run_id": self.run_id,
                "query": query,
                "primary_intent": primary_intent,
                "metadata": metadata
            } if self.run_id else {}
        )

        # Determine which aspects to apply
        use_sender_filter = bool(metadata.get("sender"))
        use_temporal_filter = bool(metadata.get("time_range"))
        use_thread_expansion = primary_intent == "thread_summary"

        self.logger.debug(
            f"Filters: sender={use_sender_filter}, temporal={use_temporal_filter}, thread={use_thread_expansion}",
            extra={
                "run_id": self.run_id,
                "use_sender": use_sender_filter,
                "use_temporal": use_temporal_filter,
                "use_thread": use_thread_expansion
            } if self.run_id else {}
        )

        # Strategy: If thread expansion is needed, use ThreadRetriever as primary
        if use_thread_expansion:
            chunks = self.thread_retriever.retrieve(
                query=query,
                top_threads=2,
                doc_type=doc_type
            )
            self.logger.debug(
                f"Thread expansion: {len(chunks)} chunks",
                extra={"run_id": self.run_id, "chunk_count": len(chunks)} if self.run_id else {}
            )

        # Otherwise, use standard semantic search
        else:
            # Get more candidates if we're going to filter
            candidate_k = top_k
            if use_sender_filter or use_temporal_filter:
                candidate_k = min(top_k * 10, 100)

            all_chunks = self.retrieval_manager.retrieve(
                query=query,
                top_k=candidate_k
            )

            # Filter by doc_type
            chunks = [c for c in all_chunks if c.meta.get("doc_type") == doc_type]
            self.logger.debug(
                f"Semantic search: {len(chunks)} email chunks",
                extra={"run_id": self.run_id, "chunk_count": len(chunks)} if self.run_id else {}
            )

        # Apply sender filter
        if use_sender_filter:
            sender_name = metadata["sender"]
            chunks = self._filter_by_sender(chunks, sender_name)
            self.logger.debug(
                f"After sender filter ('{sender_name}'): {len(chunks)} chunks",
                extra={"run_id": self.run_id, "sender": sender_name, "chunk_count": len(chunks)} if self.run_id else {}
            )

        # Apply temporal filter
        if use_temporal_filter:
            time_range = self.temporal_retriever.parse_time_range(metadata["time_range"])
            chunks = self._filter_by_time_range(chunks, time_range)
            self.logger.debug(
                f"After temporal filter ({metadata['time_range']}): {len(chunks)} chunks",
                extra={
                    "run_id": self.run_id,
                    "time_range": metadata['time_range'],
                    "chunk_count": len(chunks)
                } if self.run_id else {}
            )

        # Limit to top_k
        chunks = chunks[:top_k]

        # Sort based on intent
        if "temporal_query" in secondary_signals or use_temporal_filter:
            # Sort by date (newest first) for temporal queries
            chunks.sort(key=lambda c: c.meta.get("date", ""), reverse=True)
        elif use_thread_expansion:
            # Already chronologically sorted by ThreadRetriever
            pass
        else:
            # Keep relevance order from semantic search
            pass

        # Add retriever tagging
        for chunk in chunks:
            # Only tag if not already tagged by a specialized retriever
            if "_retriever" not in chunk.meta:
                chunk.meta["_retriever"] = "multi-aspect"

        self.logger.info(
            f"Retrieved {len(chunks)} emails with multi-aspect filtering",
            extra={"run_id": self.run_id, "result_count": len(chunks)} if self.run_id else {}
        )

        return chunks

    def _filter_by_sender(self, chunks: List[Chunk], sender_name: str) -> List[Chunk]:
        """
        Filter chunks by sender name (fuzzy matching).

        Args:
            chunks: List of email chunks
            sender_name: Name to filter by

        Returns:
            Filtered list of chunks
        """
        query_lower = sender_name.lower().strip()
        filtered = []

        for chunk in chunks:
            meta = chunk.meta
            sender_display = meta.get("sender_name", "").lower()
            sender_email = meta.get("sender", "").lower()

            # Check if query matches sender_name or sender_email
            if query_lower in sender_display or query_lower in sender_email:
                filtered.append(chunk)
                continue

            # Check first name match
            if sender_display:
                name_parts = sender_display.replace(',', ' ').split()
                if name_parts and query_lower == name_parts[0].lower():
                    filtered.append(chunk)

        return filtered

    def _filter_by_time_range(
        self,
        chunks: List[Chunk],
        time_range: Dict[str, str]
    ) -> List[Chunk]:
        """
        Filter chunks by time range.

        Args:
            chunks: List of email chunks
            time_range: Dictionary with "start" and "end" dates (YYYY-MM-DD)

        Returns:
            Filtered list of chunks
        """
        filtered = []

        try:
            start = datetime.strptime(time_range["start"], "%Y-%m-%d")
            end = datetime.strptime(time_range["end"], "%Y-%m-%d")

            for chunk in chunks:
                date_str = chunk.meta.get("date", "")
                if not date_str:
                    continue

                try:
                    # Parse date (format: "2025-11-18 13:48:42")
                    date_part = date_str.split()[0] if ' ' in date_str else date_str
                    date = datetime.strptime(date_part, "%Y-%m-%d")

                    if start <= date <= end:
                        filtered.append(chunk)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse date '{date_str}': {e}",
                        extra={"run_id": self.run_id} if self.run_id else {}
                    )
                    continue

        except Exception as e:
            self.logger.error(
                f"Failed to parse time range: {e}",
                extra={"run_id": self.run_id} if self.run_id else {}
            )
            return chunks  # Return unfiltered if time range parsing fails

        return filtered
