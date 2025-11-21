"""
TemporalRetriever - Retrieves emails filtered by time range.

This retriever filters emails based on temporal expressions like:
- "last week", "yesterday", "last month"
- "this week", "this month"
- "recent" (default: last 7 days)

Example:
    retriever = TemporalRetriever(project)
    intent_metadata = {"time_range": "last_week"}
    chunks = retriever.retrieve("budget", intent_metadata, top_k=15)
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.utils.logger import LoggerManager


class TemporalRetriever:
    """Retrieves emails filtered by time range with chronological sorting."""

    def __init__(self, project: ProjectManager, run_id: Optional[str] = None):
        """
        Initialize TemporalRetriever.

        Args:
            project: ProjectManager instance for the current project
            run_id: Optional run ID for logging
        """
        self.project = project
        self.run_id = run_id
        self.logger = LoggerManager.get_logger(
            "email_temporal_retriever",
            task_paths=project.get_task_paths(),
            run_id=run_id
        )
        self.retrieval_manager = RetrievalManager(project, run_id=run_id)

    def retrieve(
        self,
        query: str,
        intent_metadata: Optional[Dict] = None,
        top_k: int = 15,
        doc_type: str = "outlook_eml"
    ) -> List[Chunk]:
        """
        Retrieve emails in specific time range.

        Args:
            query: User query for semantic search
            intent_metadata: {"time_range": "last_week"} from intent detector
            top_k: Number of chunks to return (default: 15)
            doc_type: Document type to filter (default: outlook_eml)

        Returns:
            Chronologically sorted chunks from time range (newest first)
        """
        intent_metadata = intent_metadata or {}
        time_expr = intent_metadata.get("time_range", "recent")

        self.logger.info(
            f"TemporalRetriever: query='{query}', time_range={time_expr}, top_k={top_k}",
            extra={"run_id": self.run_id, "query": query, "time_range": time_expr, "top_k": top_k} if self.run_id else {}
        )

        # Parse time range
        time_range = self.parse_time_range(time_expr)
        self.logger.debug(
            f"Parsed time range: {time_range['start']} to {time_range['end']}",
            extra={"run_id": self.run_id, "start": time_range['start'], "end": time_range['end']} if self.run_id else {}
        )

        # Get candidate chunks (retrieve more than top_k to allow for filtering)
        # Use 10x multiplier to ensure we have enough after date filtering
        candidate_k = min(top_k * 10, 100)
        all_chunks = self.retrieval_manager.retrieve(
            query=query,
            top_k=candidate_k
        )

        # Filter by doc_type
        email_chunks = [c for c in all_chunks if c.meta.get("doc_type") == doc_type]
        self.logger.debug(
            f"Found {len(email_chunks)} email chunks (from {len(all_chunks)} total)",
            extra={"run_id": self.run_id, "email_count": len(email_chunks)} if self.run_id else {}
        )

        # Filter by date
        filtered = [
            c for c in email_chunks
            if self._is_in_range(c.meta.get("date", ""), time_range)
        ]
        self.logger.debug(
            f"After date filtering: {len(filtered)} chunks",
            extra={"run_id": self.run_id, "filtered_count": len(filtered)} if self.run_id else {}
        )

        # Take top K by relevance (they're already sorted by similarity)
        filtered = filtered[:top_k]

        # Sort by date (most recent first)
        filtered.sort(key=lambda c: c.meta.get("date", ""), reverse=True)

        # Add retriever tagging
        for chunk in filtered:
            chunk.meta["_retriever"] = "temporal"

        self.logger.info(
            f"Retrieved {len(filtered)} emails from {time_range['start']} to {time_range['end']}",
            extra={"run_id": self.run_id, "result_count": len(filtered)} if self.run_id else {}
        )

        return filtered

    def parse_time_range(self, time_expr: str) -> Dict[str, str]:
        """
        Parse time expression to date range.

        Supported expressions:
        - "yesterday" → previous day
        - "last_week" / "last week" → last 7 days
        - "last_month" / "last month" → last 30 days
        - "this_week" / "this week" → current week (Monday to today)
        - "this_month" / "this month" → current month (1st to today)
        - "recent" → last 7 days (default)

        Args:
            time_expr: Time expression string

        Returns:
            Dictionary with "start" and "end" dates (YYYY-MM-DD format)

        Examples:
            >>> parse_time_range("last_week")
            {"start": "2025-11-13", "end": "2025-11-20"}
            >>> parse_time_range("yesterday")
            {"start": "2025-11-19", "end": "2025-11-19"}
        """
        now = datetime.now()
        time_expr_normalized = time_expr.lower().replace(" ", "_")

        if time_expr_normalized == "yesterday":
            start = end = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        elif time_expr_normalized == "last_week":
            start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")

        elif time_expr_normalized == "last_month":
            start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")

        elif time_expr_normalized == "this_week":
            # Start of week (Monday)
            start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")

        elif time_expr_normalized == "this_month":
            # Start of month (1st)
            start = now.replace(day=1).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")

        else:  # "recent" or unknown
            # Default to last 7 days
            start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")

        return {"start": start, "end": end}

    def _is_in_range(self, date_str: str, time_range: Dict[str, str]) -> bool:
        """
        Check if date is within the specified range.

        Args:
            date_str: Date string in "YYYY-MM-DD HH:MM:SS" format
            time_range: Dictionary with "start" and "end" dates

        Returns:
            True if date is within range, False otherwise
        """
        if not date_str:
            return False

        try:
            # Parse date (format: "2025-11-18 13:48:42")
            # Extract just the date part (YYYY-MM-DD)
            date_part = date_str.split()[0] if ' ' in date_str else date_str
            date = datetime.strptime(date_part, "%Y-%m-%d")

            start = datetime.strptime(time_range["start"], "%Y-%m-%d")
            end = datetime.strptime(time_range["end"], "%Y-%m-%d")

            return start <= date <= end

        except Exception as e:
            self.logger.warning(
                f"Failed to parse date '{date_str}': {e}",
                extra={"run_id": self.run_id, "date_str": date_str} if self.run_id else {}
            )
            return False
