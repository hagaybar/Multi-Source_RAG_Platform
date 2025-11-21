"""
Unit tests for EmailMultiAspectRetriever.

Tests cover:
- Combined sender + temporal filtering
- Thread expansion with filters
- Pipeline order (semantic -> sender -> temporal)
- Intent-based sorting
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from scripts.retrieval.email_multi_aspect_retriever import MultiAspectRetriever
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
def multi_retriever(mock_project):
    """Create a MultiAspectRetriever instance with mocked dependencies."""
    with patch('scripts.retrieval.email_multi_aspect_retriever.RetrievalManager'), \
         patch('scripts.retrieval.email_multi_aspect_retriever.ThreadRetriever'), \
         patch('scripts.retrieval.email_multi_aspect_retriever.TemporalRetriever'), \
         patch('scripts.retrieval.email_multi_aspect_retriever.SenderRetriever'), \
         patch('scripts.retrieval.email_multi_aspect_retriever.LoggerManager.get_logger') as mock_logger:
        mock_logger.return_value = Mock()
        retriever = MultiAspectRetriever(mock_project)
        return retriever


class TestSenderFiltering:
    """Test sender filtering in multi-aspect retrieval."""

    def test_filter_by_sender(self, multi_retriever):
        """Test filtering chunks by sender name."""
        chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email from Alice", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email from Bob", token_count=10,
                meta={"sender_name": "Bob Smith", "sender": "bob@company.com"}
            ),
        ]

        filtered = multi_retriever._filter_by_sender(chunks, "Alice")

        assert len(filtered) == 1
        assert filtered[0].id == "1"

    def test_filter_by_sender_email(self, multi_retriever):
        """Test filtering by email address."""
        chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email", token_count=10,
                meta={"sender_name": "Bob Smith", "sender": "bob@company.com"}
            ),
        ]

        filtered = multi_retriever._filter_by_sender(chunks, "alice@company")

        assert len(filtered) == 1
        assert filtered[0].id == "1"

    def test_filter_by_sender_case_insensitive(self, multi_retriever):
        """Test case-insensitive sender filtering."""
        chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email", token_count=10,
                meta={"sender_name": "Alice Johnson", "sender": "alice@company.com"}
            ),
        ]

        filtered = multi_retriever._filter_by_sender(chunks, "ALICE")

        assert len(filtered) == 1

    def test_filter_by_sender_no_matches(self, multi_retriever):
        """Test filtering when no sender matches."""
        chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email", token_count=10,
                meta={"sender_name": "Bob Smith", "sender": "bob@company.com"}
            ),
        ]

        filtered = multi_retriever._filter_by_sender(chunks, "Alice")

        assert len(filtered) == 0


class TestTemporalFiltering:
    """Test temporal filtering in multi-aspect retrieval."""

    def test_filter_by_time_range(self, multi_retriever):
        """Test filtering chunks by time range."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d %H:%M:%S")
        week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

        chunks = [
            Chunk(id="1", doc_id="d1", text="Today", token_count=10, meta={"date": today}),
            Chunk(id="2", doc_id="d2", text="Week ago", token_count=10, meta={"date": week_ago}),
            Chunk(id="3", doc_id="d3", text="Month ago", token_count=10, meta={"date": month_ago}),
        ]

        # Filter for last 10 days
        time_range = {
            "start": (now - timedelta(days=10)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d")
        }

        filtered = multi_retriever._filter_by_time_range(chunks, time_range)

        assert len(filtered) == 2  # today and week_ago
        assert filtered[0].id in ["1", "2"]
        assert filtered[1].id in ["1", "2"]

    def test_filter_by_time_range_boundaries(self, multi_retriever):
        """Test time range boundary conditions."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        chunks = [
            Chunk(id="1", doc_id="d1", text="Start", token_count=10, meta={"date": "2025-11-15 00:00:00"}),
            Chunk(id="2", doc_id="d2", text="End", token_count=10, meta={"date": "2025-11-20 23:59:59"}),
            Chunk(id="3", doc_id="d3", text="Before", token_count=10, meta={"date": "2025-11-14 23:59:59"}),
            Chunk(id="4", doc_id="d4", text="After", token_count=10, meta={"date": "2025-11-21 00:00:01"}),
        ]

        filtered = multi_retriever._filter_by_time_range(chunks, time_range)

        # Should include start and end boundaries
        assert len(filtered) == 2
        assert filtered[0].id == "1"
        assert filtered[1].id == "2"

    def test_filter_by_time_range_missing_date(self, multi_retriever):
        """Test handling of chunks with missing dates."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        chunks = [
            Chunk(id="1", doc_id="d1", text="With date", token_count=10, meta={"date": "2025-11-17 10:00:00"}),
            Chunk(id="2", doc_id="d2", text="No date", token_count=10, meta={}),
        ]

        filtered = multi_retriever._filter_by_time_range(chunks, time_range)

        # Should only include chunk with valid date
        assert len(filtered) == 1
        assert filtered[0].id == "1"

    def test_filter_by_time_range_invalid_date(self, multi_retriever):
        """Test handling of invalid date formats."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        chunks = [
            Chunk(id="1", doc_id="d1", text="Valid", token_count=10, meta={"date": "2025-11-17 10:00:00"}),
            Chunk(id="2", doc_id="d2", text="Invalid", token_count=10, meta={"date": "invalid-date"}),
        ]

        filtered = multi_retriever._filter_by_time_range(chunks, time_range)

        # Should skip invalid dates
        assert len(filtered) == 1
        assert filtered[0].id == "1"


class TestCombinedFiltering:
    """Test combined sender + temporal filtering."""

    def test_sender_and_temporal_filters(self, multi_retriever):
        """Test applying both sender and temporal filters."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d %H:%M:%S")
        month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

        mock_chunks = [
            # Alice, recent (MATCH)
            Chunk(
                id="1", doc_id="d1", text="Email", token_count=10,
                meta={
                    "sender_name": "Alice Johnson",
                    "sender": "alice@company.com",
                    "date": today,
                    "doc_type": "outlook_eml"
                }
            ),
            # Alice, old (NO MATCH - wrong date)
            Chunk(
                id="2", doc_id="d2", text="Email", token_count=10,
                meta={
                    "sender_name": "Alice Johnson",
                    "sender": "alice@company.com",
                    "date": month_ago,
                    "doc_type": "outlook_eml"
                }
            ),
            # Bob, recent (NO MATCH - wrong sender)
            Chunk(
                id="3", doc_id="d3", text="Email", token_count=10,
                meta={
                    "sender_name": "Bob Smith",
                    "sender": "bob@company.com",
                    "date": today,
                    "doc_type": "outlook_eml"
                }
            ),
        ]

        multi_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)
        multi_retriever.temporal_retriever.parse_time_range = Mock(return_value={
            "start": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d")
        })

        intent = {
            "primary_intent": "sender_query",
            "metadata": {"sender": "Alice", "time_range": "last_week"},
            "secondary_signals": ["temporal_query"]
        }

        chunks = multi_retriever.retrieve("test", intent, top_k=10)

        # Should only match Alice's recent email
        assert len(chunks) == 1
        assert chunks[0].id == "1"

    def test_combined_filters_order(self, multi_retriever):
        """Test that filters are applied in correct order."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d %H:%M:%S")

        # Start with 100 chunks, filter should reduce to subset
        mock_chunks = [
            Chunk(
                id=str(i), doc_id=f"d{i}", text=f"Email {i}", token_count=10,
                meta={
                    "sender_name": "Alice Johnson" if i < 10 else "Bob Smith",
                    "sender": "alice@company.com" if i < 10 else "bob@company.com",
                    "date": today,
                    "doc_type": "outlook_eml"
                }
            )
            for i in range(100)
        ]

        multi_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)
        multi_retriever.temporal_retriever.parse_time_range = Mock(return_value={
            "start": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": now.strftime("%Y-%m-%d")
        })

        intent = {
            "primary_intent": "sender_query",
            "metadata": {"sender": "Alice", "time_range": "last_week"}
        }

        chunks = multi_retriever.retrieve("test", intent, top_k=5)

        # Should have 5 results (top_k limit)
        assert len(chunks) == 5
        # All should be from Alice
        for chunk in chunks:
            assert "Alice" in chunk.meta["sender_name"]


class TestThreadExpansion:
    """Test thread expansion in multi-aspect retrieval."""

    def test_thread_expansion_primary_intent(self, multi_retriever):
        """Test that thread_summary intent uses ThreadRetriever."""
        mock_thread_chunks = [
            Chunk(id="1", doc_id="d1", text="Email 1", token_count=10, meta={"date": "2025-11-20 10:00:00"}),
            Chunk(id="2", doc_id="d2", text="Email 2", token_count=10, meta={"date": "2025-11-20 11:00:00"}),
        ]

        multi_retriever.thread_retriever.retrieve = Mock(return_value=mock_thread_chunks)

        intent = {
            "primary_intent": "thread_summary",
            "metadata": {}
        }

        chunks = multi_retriever.retrieve("test", intent, top_k=10)

        # Should call thread retriever
        multi_retriever.thread_retriever.retrieve.assert_called_once()
        assert len(chunks) == 2

    def test_standard_retrieval_no_thread_intent(self, multi_retriever):
        """Test that non-thread intents use standard retrieval."""
        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email", token_count=10,
                meta={"doc_type": "outlook_eml", "date": "2025-11-20 10:00:00"}
            ),
        ]

        multi_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        intent = {
            "primary_intent": "factual_lookup",
            "metadata": {}
        }

        chunks = multi_retriever.retrieve("test", intent, top_k=10)

        # Should call standard retrieval manager
        multi_retriever.retrieval_manager.retrieve.assert_called_once()
        assert len(chunks) == 1


class TestSorting:
    """Test sorting behavior based on intent."""

    def test_temporal_query_sorts_by_date(self, multi_retriever):
        """Test that temporal queries sort by date (newest first)."""
        now = datetime.now()
        dates = [
            (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
            (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
        ]

        # Return in random order
        mock_chunks = [
            Chunk(id="2", doc_id="d2", text="Middle", token_count=10, meta={"date": dates[1], "doc_type": "outlook_eml"}),
            Chunk(id="1", doc_id="d1", text="Oldest", token_count=10, meta={"date": dates[0], "doc_type": "outlook_eml"}),
            Chunk(id="3", doc_id="d3", text="Newest", token_count=10, meta={"date": dates[2], "doc_type": "outlook_eml"}),
        ]

        multi_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        intent = {
            "primary_intent": "factual_lookup",
            "metadata": {},
            "secondary_signals": ["temporal_query"]
        }

        chunks = multi_retriever.retrieve("test", intent, top_k=10)

        # Should be sorted newest first
        assert chunks[0].id == "3"  # Newest
        assert chunks[1].id == "2"  # Middle
        assert chunks[2].id == "1"  # Oldest

    def test_non_temporal_preserves_relevance_order(self, multi_retriever):
        """Test that non-temporal queries preserve semantic search order."""
        mock_chunks = [
            Chunk(id="1", doc_id="d1", text="Most relevant", token_count=10, meta={"date": "2025-11-20", "doc_type": "outlook_eml"}),
            Chunk(id="2", doc_id="d2", text="Less relevant", token_count=10, meta={"date": "2025-11-21", "doc_type": "outlook_eml"}),
        ]

        multi_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        intent = {
            "primary_intent": "factual_lookup",
            "metadata": {},
            "secondary_signals": []
        }

        chunks = multi_retriever.retrieve("test", intent, top_k=10)

        # Should preserve relevance order (not sort by date)
        assert chunks[0].id == "1"
        assert chunks[1].id == "2"


class TestDocTypeFiltering:
    """Test document type filtering."""

    def test_filters_non_email_doc_types(self, multi_retriever):
        """Test that non-email documents are filtered out."""
        mock_chunks = [
            Chunk(id="1", doc_id="d1", text="Email", token_count=10, meta={"doc_type": "outlook_eml"}),
            Chunk(id="2", doc_id="d2", text="PDF", token_count=10, meta={"doc_type": "pdf"}),
            Chunk(id="3", doc_id="d3", text="Word", token_count=10, meta={"doc_type": "docx"}),
        ]

        multi_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = multi_retriever.retrieve("test", intent={}, top_k=10)

        # Should only return email
        assert len(chunks) == 1
        assert chunks[0].id == "1"


class TestTopKLimit:
    """Test top_k limit enforcement."""

    def test_respects_top_k_limit(self, multi_retriever):
        """Test that results are limited to top_k."""
        mock_chunks = [
            Chunk(
                id=str(i), doc_id=f"d{i}", text=f"Email {i}", token_count=10,
                meta={"doc_type": "outlook_eml", "date": "2025-11-20"}
            )
            for i in range(20)
        ]

        multi_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = multi_retriever.retrieve("test", intent={}, top_k=5)

        assert len(chunks) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
