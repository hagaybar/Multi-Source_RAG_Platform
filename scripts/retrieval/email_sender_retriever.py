"""
SenderRetriever - Retrieves emails from specific sender with fuzzy matching.

This retriever filters emails based on sender name or email address.
Supports fuzzy matching:
- "Alice" matches "Alice Johnson" or "alice.j@company.com"
- First name matching
- Email address partial matching

Example:
    retriever = SenderRetriever(project)
    intent_metadata = {"sender": "Alice"}
    chunks = retriever.retrieve("budget", intent_metadata, top_k=10)
"""

from typing import List, Dict, Optional

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.utils.logger import LoggerManager


class SenderRetriever:
    """Retrieves emails from specific sender with fuzzy name matching."""

    def __init__(self, project: ProjectManager, run_id: Optional[str] = None):
        """
        Initialize SenderRetriever.

        Args:
            project: ProjectManager instance for the current project
            run_id: Optional run ID for logging
        """
        self.project = project
        self.run_id = run_id
        self.logger = LoggerManager.get_logger(
            "email_sender_retriever",
            task_paths=project.get_task_paths(),
            run_id=run_id
        )
        self.retrieval_manager = RetrievalManager(project, run_id=run_id)

    def retrieve(
        self,
        query: str,
        intent_metadata: Optional[Dict] = None,
        top_k: int = 10,
        doc_type: str = "outlook_eml"
    ) -> List[Chunk]:
        """
        Retrieve emails from specific sender.

        Args:
            query: User query for semantic search
            intent_metadata: {"sender": "Alice"} from intent detector
            top_k: Number of chunks to return (default: 10)
            doc_type: Document type to filter (default: outlook_eml)

        Returns:
            List of email chunks from specified sender, sorted by relevance
        """
        intent_metadata = intent_metadata or {}
        sender_name = intent_metadata.get("sender")

        if not sender_name:
            # No sender specified, fall back to standard retrieval
            self.logger.warning(
                "No sender specified in metadata, using standard retrieval",
                extra={"run_id": self.run_id} if self.run_id else {}
            )
            return self.retrieval_manager.retrieve(query=query, top_k=top_k)

        self.logger.info(
            f"SenderRetriever: query='{query}', sender='{sender_name}', top_k={top_k}",
            extra={"run_id": self.run_id, "query": query, "sender": sender_name, "top_k": top_k} if self.run_id else {}
        )

        # Get candidate chunks (retrieve more than top_k to allow for filtering)
        # Use 10x multiplier to ensure we have enough after sender filtering
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

        # Filter by sender (fuzzy match)
        filtered = [
            c for c in email_chunks
            if self._sender_matches(c.meta, sender_name)
        ]

        self.logger.debug(
            f"After sender filtering: {len(filtered)} chunks",
            extra={"run_id": self.run_id, "filtered_count": len(filtered)} if self.run_id else {}
        )

        # Take top K (already sorted by relevance from semantic search)
        result = filtered[:top_k]

        # Add retriever tagging
        for chunk in result:
            chunk.meta["_retriever"] = "sender"

        self.logger.info(
            f"Retrieved {len(result)} emails from sender '{sender_name}'",
            extra={"run_id": self.run_id, "result_count": len(result)} if self.run_id else {}
        )

        return result

    def _sender_matches(self, meta: Dict, query_name: str) -> bool:
        """
        Fuzzy match sender name.

        Matches against:
        1. sender_name field (display name)
        2. sender field (email address)
        3. First name from sender_name

        Args:
            meta: Chunk metadata containing sender info
            query_name: Name to search for

        Returns:
            True if sender matches, False otherwise

        Examples:
            "Alice" matches:
            - sender_name: "Alice Johnson"
            - sender_name: "Alice J"
            - sender: "alice.j@company.com"
            - sender_name: "Alice" (exact match)
        """
        query_lower = query_name.lower().strip()

        # Empty query doesn't match anything
        if not query_lower:
            return False

        # Check sender_name field (display name)
        sender_name = meta.get("sender_name", "").lower()
        if query_lower in sender_name:
            self.logger.debug(
                f"Match: '{query_name}' in sender_name '{sender_name}'",
                extra={"run_id": self.run_id} if self.run_id else {}
            )
            return True

        # Check sender email
        sender_email = meta.get("sender", "").lower()
        if query_lower in sender_email:
            self.logger.debug(
                f"Match: '{query_name}' in sender_email '{sender_email}'",
                extra={"run_id": self.run_id} if self.run_id else {}
            )
            return True

        # Check if query matches first name from sender_name
        if sender_name:
            # Split on space, comma, or other common separators
            name_parts = sender_name.replace(',', ' ').split()
            if name_parts:
                first_name = name_parts[0].lower()
                if query_lower == first_name:
                    self.logger.debug(
                        f"Match: '{query_name}' == first_name '{first_name}'",
                        extra={"run_id": self.run_id} if self.run_id else {}
                    )
                    return True

        return False
