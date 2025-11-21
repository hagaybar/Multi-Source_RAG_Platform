"""
Unit tests for EmailTemporalRetriever.

Tests cover:
- Time range parsing
- Date filtering
- Chronological sorting
- Edge cases (invalid dates, missing dates)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from scripts.retrieval.email_temporal_retriever import TemporalRetriever
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
def temporal_retriever(mock_project):
    """Create a TemporalRetriever instance with mocked dependencies."""
    with patch('scripts.retrieval.email_temporal_retriever.RetrievalManager'), \
         patch('scripts.retrieval.email_temporal_retriever.LoggerManager.get_logger') as mock_logger:
        mock_logger.return_value = Mock()
        retriever = TemporalRetriever(mock_project)
        return retriever


class TestTimeRangeParsing:
    """Test parsing of time expressions."""

    def test_parse_yesterday(self, temporal_retriever):
        """Test parsing 'yesterday'."""
        result = temporal_retriever.parse_time_range("yesterday")

        expected_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert result["start"] == expected_date
        assert result["end"] == expected_date

    def test_parse_last_week(self, temporal_retriever):
        """Test parsing 'last_week'."""
        result = temporal_retriever.parse_time_range("last_week")

        expected_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        expected_end = datetime.now().strftime("%Y-%m-%d")
        assert result["start"] == expected_start
        assert result["end"] == expected_end

    def test_parse_last_month(self, temporal_retriever):
        """Test parsing 'last_month'."""
        result = temporal_retriever.parse_time_range("last_month")

        expected_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        expected_end = datetime.now().strftime("%Y-%m-%d")
        assert result["start"] == expected_start
        assert result["end"] == expected_end

    def test_parse_this_week(self, temporal_retriever):
        """Test parsing 'this_week'."""
        result = temporal_retriever.parse_time_range("this_week")

        # Start of week (Monday)
        now = datetime.now()
        expected_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        expected_end = now.strftime("%Y-%m-%d")
        assert result["start"] == expected_start
        assert result["end"] == expected_end

    def test_parse_this_month(self, temporal_retriever):
        """Test parsing 'this_month'."""
        result = temporal_retriever.parse_time_range("this_month")

        expected_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
        expected_end = datetime.now().strftime("%Y-%m-%d")
        assert result["start"] == expected_start
        assert result["end"] == expected_end

    def test_parse_recent_default(self, temporal_retriever):
        """Test parsing 'recent' (default: last 7 days)."""
        result = temporal_retriever.parse_time_range("recent")

        expected_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        expected_end = datetime.now().strftime("%Y-%m-%d")
        assert result["start"] == expected_start
        assert result["end"] == expected_end

    def test_parse_unknown_defaults_to_recent(self, temporal_retriever):
        """Test that unknown expressions default to 'recent'."""
        result = temporal_retriever.parse_time_range("unknown_expression")

        expected_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        expected_end = datetime.now().strftime("%Y-%m-%d")
        assert result["start"] == expected_start
        assert result["end"] == expected_end

    def test_parse_with_spaces(self, temporal_retriever):
        """Test parsing with spaces instead of underscores."""
        result = temporal_retriever.parse_time_range("last week")

        expected_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        expected_end = datetime.now().strftime("%Y-%m-%d")
        assert result["start"] == expected_start
        assert result["end"] == expected_end


class TestDateFiltering:
    """Test filtering of chunks by date range."""

    def test_is_in_range_within_range(self, temporal_retriever):
        """Test that dates within range return True."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        assert temporal_retriever._is_in_range("2025-11-17 10:00:00", time_range) is True
        assert temporal_retriever._is_in_range("2025-11-15 00:00:00", time_range) is True  # Start boundary
        assert temporal_retriever._is_in_range("2025-11-20 23:59:59", time_range) is True  # End boundary

    def test_is_in_range_outside_range(self, temporal_retriever):
        """Test that dates outside range return False."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        assert temporal_retriever._is_in_range("2025-11-14 23:59:59", time_range) is False  # Before
        assert temporal_retriever._is_in_range("2025-11-21 00:00:01", time_range) is False  # After

    def test_is_in_range_empty_date(self, temporal_retriever):
        """Test handling of empty date string."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        assert temporal_retriever._is_in_range("", time_range) is False

    def test_is_in_range_invalid_date(self, temporal_retriever):
        """Test handling of invalid date format."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        assert temporal_retriever._is_in_range("invalid-date", time_range) is False
        assert temporal_retriever._is_in_range("2025-13-45", time_range) is False  # Invalid month/day

    def test_is_in_range_date_only(self, temporal_retriever):
        """Test date string without time component."""
        time_range = {"start": "2025-11-15", "end": "2025-11-20"}

        assert temporal_retriever._is_in_range("2025-11-17", time_range) is True


class TestRetrieval:
    """Test complete retrieval workflow."""

    def test_retrieve_with_temporal_filter(self, temporal_retriever):
        """Test retrieval with temporal filtering."""
        # Create mock chunks with dates
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d %H:%M:%S")
        week_ago_str = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        month_ago_str = (now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Recent email", token_count=10,
                meta={"date": today_str, "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Week old email", token_count=10,
                meta={"date": week_ago_str, "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="3", doc_id="d3", text="Month old email", token_count=10,
                meta={"date": month_ago_str, "doc_type": "outlook_eml"}
            ),
        ]

        temporal_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        # Retrieve with "last_week" filter
        intent_metadata = {"time_range": "last_week"}
        chunks = temporal_retriever.retrieve("test query", intent_metadata, top_k=10)

        # Should only get chunks from last 7 days
        assert len(chunks) == 2  # today and week_ago, not month_ago
        assert chunks[0].id in ["1", "2"]
        assert chunks[1].id in ["1", "2"]

    def test_retrieve_chronological_sorting(self, temporal_retriever):
        """Test that results are sorted by date (newest first)."""
        now = datetime.now()
        dates = [
            (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
            (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
        ]

        mock_chunks = [
            Chunk(
                id=str(i), doc_id=f"d{i}", text=f"Email {i}", token_count=10,
                meta={"date": date, "doc_type": "outlook_eml"}
            )
            for i, date in enumerate(dates)
        ]

        temporal_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = temporal_retriever.retrieve("test", {"time_range": "recent"}, top_k=10)

        # Should be sorted newest first
        assert chunks[0].meta["date"] == dates[2]  # Most recent
        assert chunks[1].meta["date"] == dates[1]
        assert chunks[2].meta["date"] == dates[0]  # Oldest

    def test_retrieve_no_time_range(self, temporal_retriever):
        """Test retrieval with no time_range specified (defaults to 'recent')."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d %H:%M:%S")

        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email", token_count=10,
                meta={"date": today_str, "doc_type": "outlook_eml"}
            ),
        ]

        temporal_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        # No intent_metadata provided
        chunks = temporal_retriever.retrieve("test", intent_metadata=None, top_k=10)

        assert len(chunks) == 1  # Should use 'recent' as default

    def test_retrieve_filters_by_doc_type(self, temporal_retriever):
        """Test that non-email chunks are filtered out."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d %H:%M:%S")

        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email", token_count=10,
                meta={"date": today_str, "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="2", doc_id="d2", text="PDF", token_count=10,
                meta={"date": today_str, "doc_type": "pdf"}
            ),
        ]

        temporal_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = temporal_retriever.retrieve("test", {"time_range": "recent"}, top_k=10)

        # Should only return email chunk
        assert len(chunks) == 1
        assert chunks[0].id == "1"

    def test_retrieve_respects_top_k(self, temporal_retriever):
        """Test that top_k limit is respected."""
        now = datetime.now()

        mock_chunks = [
            Chunk(
                id=str(i), doc_id=f"d{i}", text=f"Email {i}", token_count=10,
                meta={
                    "date": (now - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"),
                    "doc_type": "outlook_eml"
                }
            )
            for i in range(20)  # 20 chunks
        ]

        temporal_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = temporal_retriever.retrieve("test", {"time_range": "recent"}, top_k=5)

        # Should only return top 5
        assert len(chunks) == 5

    def test_retrieve_handles_missing_dates(self, temporal_retriever):
        """Test that chunks with missing dates are excluded."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d %H:%M:%S")

        mock_chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email with date", token_count=10,
                meta={"date": today_str, "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email without date", token_count=10,
                meta={"doc_type": "outlook_eml"}  # No date
            ),
        ]

        temporal_retriever.retrieval_manager.retrieve = Mock(return_value=mock_chunks)

        chunks = temporal_retriever.retrieve("test", {"time_range": "recent"}, top_k=10)

        # Should only return chunk with valid date
        assert len(chunks) == 1
        assert chunks[0].id == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
