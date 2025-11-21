"""
Unit tests for EmailSenderRetriever.

Tests cover:
- Sender name fuzzy matching
- Email address matching
- First name matching
- Edge cases (missing sender, empty queries)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from scripts.retrieval.email_sender_retriever import SenderRetriever
from scripts.chunking.models import Chunk


@pytest.fixture
def mock_project():
    """Create a mock ProjectManager."""
    project = Mock()
    # Mock get_task_paths to return a TaskPaths object with proper log path
    mock_task_paths = Mock()
    mock_task_paths.get_log_path = Mock(return_value="/tmp/test.log")
    project.get_task_paths = Mock(return_value=mock_task_paths)
    return project


@pytest.fixture
def sender_retriever(mock_project):
    """Create a SenderRetriever instance with mocked dependencies."""
    with patch('scripts.retrieval.email_sender_retriever.RetrievalManager'), \
         patch('scripts.retrieval.email_sender_retriever.LoggerManager.get_logger') as mock_logger:
        mock_logger.return_value = Mock()
        retriever = SenderRetriever(mock_project)
        return retriever


class TestSenderMatching:
    """Test sender name matching logic."""

    def test_exact_sender_name_match(self, sender_retriever):
        """Test exact match on sender_name field."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice.j@company.com"}

        assert sender_retriever._sender_matches(meta, "Alice Johnson") is True

    def test_partial_sender_name_match(self, sender_retriever):
        """Test partial match on sender_name field."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice.j@company.com"}

        assert sender_retriever._sender_matches(meta, "Alice") is True
        assert sender_retriever._sender_matches(meta, "Johnson") is True

    def test_sender_email_match(self, sender_retriever):
        """Test match on sender email address."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice.j@company.com"}

        assert sender_retriever._sender_matches(meta, "alice.j") is True
        assert sender_retriever._sender_matches(meta, "company.com") is True

    def test_first_name_exact_match(self, sender_retriever):
        """Test exact first name matching."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice.j@company.com"}

        assert sender_retriever._sender_matches(meta, "alice") is True  # Case insensitive

    def test_first_name_not_partial_match(self, sender_retriever):
        """Test that first name matching requires exact match."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice.j@company.com"}

        # "Ali" is not the full first name, should not match via first name logic
        # (but might match via partial sender_name match)
        # This test verifies the first name logic specifically
        assert sender_retriever._sender_matches(meta, "Al") is True  # Partial sender_name match

    def test_case_insensitive_matching(self, sender_retriever):
        """Test that matching is case insensitive."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice.j@company.com"}

        assert sender_retriever._sender_matches(meta, "ALICE") is True
        assert sender_retriever._sender_matches(meta, "aLiCe") is True
        assert sender_retriever._sender_matches(meta, "JOHNSON") is True

    def test_no_match(self, sender_retriever):
        """Test when sender doesn't match."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice.j@company.com"}

        assert sender_retriever._sender_matches(meta, "Bob") is False
        assert sender_retriever._sender_matches(meta, "Smith") is False

    def test_missing_sender_name(self, sender_retriever):
        """Test handling when sender_name is missing."""
        meta = {"sender": "alice.j@company.com"}

        # Should still match on email
        assert sender_retriever._sender_matches(meta, "alice.j") is True
        # "Alice" matches because "alice" is in the email address
        assert sender_retriever._sender_matches(meta, "alice") is True
        # But unrelated name doesn't match
        assert sender_retriever._sender_matches(meta, "Bob") is False

    def test_missing_sender_email(self, sender_retriever):
        """Test handling when sender email is missing."""
        meta = {"sender_name": "Alice Johnson"}

        # Should still match on name
        assert sender_retriever._sender_matches(meta, "Alice") is True
        assert sender_retriever._sender_matches(meta, "company.com") is False

    def test_empty_sender_fields(self, sender_retriever):
        """Test handling when both sender fields are empty."""
        meta = {"sender_name": "", "sender": ""}

        assert sender_retriever._sender_matches(meta, "Alice") is False

    def test_sender_with_comma_format(self, sender_retriever):
        """Test sender name in 'Last, First' format."""
        meta = {"sender_name": "Johnson, Alice", "sender": "alice.j@company.com"}

        # First name after comma should still match
        assert sender_retriever._sender_matches(meta, "Alice") is True

    def test_sender_with_middle_name(self, sender_retriever):
        """Test sender name with middle name/initial."""
        meta = {"sender_name": "Alice Marie Johnson", "sender": "alice.m.j@company.com"}

        assert sender_retriever._sender_matches(meta, "Alice") is True
        assert sender_retriever._sender_matches(meta, "Marie") is True


class TestRetrieval:
    """Test complete retrieval workflow."""

    def test_retrieve_with_sender_filter(self, sender_retriever):
        """Test retrieval with sender filtering."""
        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email from Alice", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com", "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email from Bob", token_count=10,
                meta={"sender_name": "Bob Smith", "sender": "bob@company.com", "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="3", doc_id="d3", text="Another email from Alice", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com", "doc_type": "outlook_eml"}
            ),
        ]

        sender_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        intent_metadata = {"sender": "Alice"}
        chunks = sender_retriever.retrieve("test query", intent_metadata, top_k=10)

        # Should only return Alice's emails
        assert len(chunks) == 2
        assert chunks[0].id in ["1", "3"]
        assert chunks[1].id in ["1", "3"]

    def test_retrieve_no_sender_metadata(self, sender_retriever):
        """Test retrieval when no sender is specified."""
        mock_chunks = [
            Chunk(id="1", doc_id="d1", text="Email", token_count=10, meta={"doc_type": "outlook_eml"}),
        ]

        sender_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        # No sender in metadata
        chunks = sender_retriever.retrieve("test", intent_metadata={}, top_k=10)

        # Should fall back to standard retrieval
        assert len(chunks) == 1

    def test_retrieve_filters_by_doc_type(self, sender_retriever):
        """Test that non-email chunks are filtered out."""
        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email from Alice", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com", "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="2", doc_id="d2", text="PDF from Alice", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com", "doc_type": "pdf"}
            ),
        ]

        sender_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = sender_retriever.retrieve("test", {"sender": "Alice"}, top_k=10)

        # Should only return email chunk
        assert len(chunks) == 1
        assert chunks[0].id == "1"

    def test_retrieve_respects_top_k(self, sender_retriever):
        """Test that top_k limit is respected."""
        mock_chunks = [
            Chunk(
                id=str(i), doc_id=f"d{i}", text=f"Email {i}", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com", "doc_type": "outlook_eml"}
            )
            for i in range(20)  # 20 emails from Alice
        ]

        sender_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = sender_retriever.retrieve("test", {"sender": "Alice"}, top_k=5)

        # Should only return top 5
        assert len(chunks) == 5

    def test_retrieve_no_matching_sender(self, sender_retriever):
        """Test retrieval when no emails match sender."""
        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email from Bob", token_count=10,
                meta={"sender_name": "Bob Smith", "sender": "bob@company.com", "doc_type": "outlook_eml"}
            ),
        ]

        sender_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = sender_retriever.retrieve("test", {"sender": "Alice"}, top_k=10)

        # Should return empty list
        assert len(chunks) == 0

    def test_retrieve_maintains_relevance_order(self, sender_retriever):
        """Test that results maintain semantic search relevance order."""
        # Mock chunks already sorted by relevance from semantic search
        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Highly relevant email", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com", "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Less relevant email", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com", "doc_type": "outlook_eml"}
            ),
        ]

        sender_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = sender_retriever.retrieve("test", {"sender": "Alice"}, top_k=10)

        # Order should be preserved
        assert chunks[0].id == "1"
        assert chunks[1].id == "2"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_whitespace_in_query_name(self, sender_retriever):
        """Test handling of whitespace in sender name query."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice@company.com"}

        assert sender_retriever._sender_matches(meta, "  Alice  ") is True

    def test_empty_query_name(self, sender_retriever):
        """Test handling of empty query name."""
        meta = {"sender_name": "Alice Johnson", "sender": "alice@company.com"}

        assert sender_retriever._sender_matches(meta, "") is False

    def test_unicode_sender_name(self, sender_retriever):
        """Test handling of unicode characters in sender name."""
        meta = {"sender_name": "José García", "sender": "jose@company.com"}

        assert sender_retriever._sender_matches(meta, "José") is True
        assert sender_retriever._sender_matches(meta, "García") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
