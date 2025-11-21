"""
ThreadRetriever - Retrieves complete email threads with deduplication.

This retriever implements a 5-stage process:
1. Find seed emails using semantic search
2. Group by normalized subject
3. Score threads by relevance
4. Expand to complete threads (all emails in conversation)
5. Sort chronologically

Example:
    retriever = ThreadRetriever(project)
    chunks = retriever.retrieve("Budget discussion", top_threads=2)
"""

from typing import List, Dict, Optional
import re
import json
from pathlib import Path
from datetime import datetime

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.utils.logger import LoggerManager


class ThreadRetriever:
    """Retrieves complete email threads with deduplication and chronological ordering."""

    def __init__(self, project: ProjectManager, run_id: Optional[str] = None):
        """
        Initialize ThreadRetriever.

        Args:
            project: ProjectManager instance for the current project
            run_id: Optional run ID for logging
        """
        self.project = project
        self.run_id = run_id
        self.logger = LoggerManager.get_logger(
            "email_thread_retriever",
            task_paths=project.get_task_paths(),
            run_id=run_id
        )
        self.retrieval_manager = RetrievalManager(project, run_id=run_id)

    def retrieve(
        self,
        query: str,
        top_threads: int = 2,
        doc_type: str = "outlook_eml",
        seed_k: int = 10,
        days_back: Optional[int] = None
    ) -> List[Chunk]:
        """
        Retrieve complete email threads.

        Args:
            query: User query for semantic search
            top_threads: Number of complete threads to return (default: 2)
            doc_type: Document type to search (default: outlook_eml)
            seed_k: Number of seed emails to retrieve initially (default: 10)
            days_back: Optional filter - only include emails from last N days

        Returns:
            List of chunks representing complete threads, chronologically sorted
        """
        self.logger.info(
            f"ThreadRetriever: query='{query}', top_threads={top_threads}",
            extra={"run_id": self.run_id, "query": query, "top_threads": top_threads} if self.run_id else {}
        )

        # Stage 1: Find seed emails using semantic search
        seed_emails = self._get_seed_emails(query, doc_type, seed_k)
        if not seed_emails:
            self.logger.warning("No seed emails found", extra={"run_id": self.run_id} if self.run_id else {})
            return []

        self.logger.debug(
            f"Found {len(seed_emails)} seed emails",
            extra={"run_id": self.run_id, "seed_count": len(seed_emails)} if self.run_id else {}
        )

        # Stage 2: Group by normalized subject
        threads = self._group_by_thread(seed_emails)
        self.logger.debug(
            f"Grouped into {len(threads)} threads",
            extra={"run_id": self.run_id, "thread_count": len(threads)} if self.run_id else {}
        )

        # Stage 3: Score threads by relevance
        scored_threads = self._score_threads(threads, seed_emails)

        # Stage 4: Get complete threads (expand to full conversations)
        complete_threads = []
        for thread_id in scored_threads[:top_threads]:
            thread_emails = self._get_full_thread(thread_id, doc_type)
            complete_threads.extend(thread_emails)
            self.logger.debug(
                f"Thread '{thread_id[:50]}...': {len(thread_emails)} emails",
                extra={"run_id": self.run_id, "thread_id": thread_id, "email_count": len(thread_emails)} if self.run_id else {}
            )

        # Stage 5: Sort chronologically
        complete_threads.sort(key=lambda c: c.meta.get("date", ""))

        # Stage 6: Apply temporal filter if specified
        if days_back is not None:
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            before_filter = len(complete_threads)
            complete_threads = [
                c for c in complete_threads
                if c.meta.get("date", "").split()[0] >= cutoff_date
            ]

            self.logger.info(
                f"Temporal filter: {before_filter} → {len(complete_threads)} emails (last {days_back} days, cutoff: {cutoff_date})",
                extra={"run_id": self.run_id, "days_back": days_back, "before": before_filter, "after": len(complete_threads)} if self.run_id else {}
            )

        self.logger.info(
            f"Retrieved {len(complete_threads)} emails from {len(scored_threads[:top_threads])} threads",
            extra={"run_id": self.run_id, "total_emails": len(complete_threads), "threads_used": len(scored_threads[:top_threads])} if self.run_id else {}
        )

        return complete_threads

    def _get_seed_emails(self, query: str, doc_type: str, top_k: int) -> List[Chunk]:
        """
        Get initial seed emails using semantic search.

        Args:
            query: Search query
            doc_type: Document type to filter
            top_k: Number of results

        Returns:
            List of most relevant email chunks
        """
        # Use retrieval manager's retrieve method
        # Note: We need to filter by doc_type, but current retrieve() doesn't support that
        # For now, retrieve all and filter
        all_results = self.retrieval_manager.retrieve(query=query, top_k=top_k)

        # Filter by doc_type
        filtered = [c for c in all_results if c.meta.get("doc_type") == doc_type]

        return filtered

    def _group_by_thread(self, chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
        """
        Group emails by normalized subject (thread ID).

        Args:
            chunks: List of email chunks

        Returns:
            Dictionary mapping thread_id -> list of chunks
        """
        threads = {}

        for chunk in chunks:
            subject = chunk.meta.get("subject", "")
            thread_id = self._normalize_subject(subject)

            if thread_id not in threads:
                threads[thread_id] = []
            threads[thread_id].append(chunk)

        return threads

    def _normalize_subject(self, subject: str) -> str:
        """
        Normalize email subject for thread grouping.

        Removes Re:, Fwd:, and bracketed prefixes like [Primo], [EXTERNAL].

        Examples:
            "[Primo] Budget Discussion" → "budget discussion"
            "Re: Budget Discussion" → "budget discussion"
            "Fwd: Re: Budget Discussion" → "budget discussion"
            "[EXTERNAL] Re: Budget" → "budget"

        Args:
            subject: Email subject line

        Returns:
            Normalized subject (lowercase, stripped of prefixes)
        """
        if not subject:
            return ""

        normalized = subject

        # Remove bracketed prefixes like [Primo], [EXTERNAL], [EXTERNAL *]
        normalized = re.sub(r'\[[\w\s*]+\]\s*', '', normalized)

        # Remove Re:, Fwd:, Fw: prefixes (loop until all removed)
        while True:
            new_normalized = re.sub(r'^(re:|fwd?:|fw:)\s*', '', normalized, flags=re.I)
            if new_normalized == normalized:
                break
            normalized = new_normalized

        return normalized.lower().strip()

    def _score_threads(
        self,
        threads: Dict[str, List[Chunk]],
        seed_emails: List[Chunk]
    ) -> List[str]:
        """
        Score threads by relevance.

        Scoring factors:
        - Number of seed emails in thread (higher = more relevant)
        - Thread size (prefer moderate size: 3-15 emails)
        - Recency of most recent email

        Args:
            threads: Dictionary of thread_id -> chunks
            seed_emails: Original seed emails from semantic search

        Returns:
            List of thread IDs sorted by score (descending)
        """
        thread_scores = {}
        seed_ids = {chunk.id for chunk in seed_emails}

        for thread_id, chunks in threads.items():
            # Count seed emails in this thread
            seed_count = sum(1 for c in chunks if c.id in seed_ids)

            # Thread size score (prefer 3-15 emails)
            # Score = 1.0 for size in [3, 15], linearly scaled otherwise
            size = len(chunks)
            if size >= 3 and size <= 15:
                size_score = 1.0
            elif size < 3:
                size_score = size / 3.0
            else:
                # Decay for very large threads (>15)
                size_score = max(0.5, 1.0 - (size - 15) * 0.05)

            # Recency score (most recent email in thread)
            recency_score = self._calculate_recency_score(chunks)

            # Combined score
            # Relevance (seed_count) is weighted 2x more than size/recency
            thread_scores[thread_id] = (
                seed_count * 2.0 +  # Relevance is most important
                size_score +
                recency_score
            )

            self.logger.debug(
                f"Thread score: '{thread_id[:40]}...' = {thread_scores[thread_id]:.2f} "
                f"(seeds={seed_count}, size={size}, recency={recency_score:.2f})",
                extra={
                    "run_id": self.run_id,
                    "thread_id": thread_id,
                    "score": thread_scores[thread_id],
                    "seed_count": seed_count,
                    "size": size,
                    "recency_score": recency_score
                } if self.run_id else {}
            )

        # Return thread IDs sorted by score (descending)
        return sorted(
            thread_scores.keys(),
            key=lambda tid: thread_scores[tid],
            reverse=True
        )

    def _calculate_recency_score(self, chunks: List[Chunk]) -> float:
        """
        Calculate recency score based on most recent email in thread.

        Args:
            chunks: List of email chunks in thread

        Returns:
            Recency score (0.0 to 1.0)
        """
        if not chunks:
            return 0.0

        # Find most recent date
        dates = [c.meta.get("date", "") for c in chunks if c.meta.get("date")]
        if not dates:
            return 0.0

        most_recent = max(dates)

        try:
            # Parse date (format: "2025-11-18 13:48:42")
            date_obj = datetime.strptime(most_recent, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            days_ago = (now - date_obj).days

            # Score: 1.0 for today, decay by 0.1 per day, min 0.0
            score = max(0.0, 1.0 - (days_ago * 0.1))
            return score

        except Exception as e:
            self.logger.warning(
                f"Failed to parse date '{most_recent}': {e}",
                extra={"run_id": self.run_id} if self.run_id else {}
            )
            return 0.5  # Default to moderate score if parsing fails

    def _get_full_thread(self, thread_id: str, doc_type: str) -> List[Chunk]:
        """
        Get ALL emails in a thread, not just those in seed set.

        This ensures complete conversation context by loading all metadata
        and filtering by normalized subject.

        Args:
            thread_id: Normalized subject line (thread identifier)
            doc_type: Document type (e.g., "outlook_eml")

        Returns:
            List of all chunks in this thread
        """
        # Load metadata file directly
        metadata_path = self.project.get_metadata_path(doc_type)
        if not metadata_path.exists():
            self.logger.warning(
                f"Metadata file not found: {metadata_path}",
                extra={"run_id": self.run_id} if self.run_id else {}
            )
            return []

        # Load all metadata
        all_metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_metadata.append(json.loads(line))

        # Filter by normalized subject
        thread_chunks = []
        for meta in all_metadata:
            subject = meta.get("subject", "")
            if self._normalize_subject(subject) == thread_id:
                # Reconstruct Chunk object with retriever tagging
                chunk = Chunk(
                    id=meta.get("id", f"chunk-{len(thread_chunks)}"),
                    doc_id=meta.get("doc_id", "unknown"),
                    text=meta.get("text", ""),
                    token_count=meta.get("token_count", 0),
                    meta={
                        **meta,
                        "_retriever": "thread"  # Tag with retriever name
                    }
                )
                thread_chunks.append(chunk)

        return thread_chunks
