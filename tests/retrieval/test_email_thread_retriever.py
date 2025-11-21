"""
Unit tests for EmailThreadRetriever.

Tests cover:
- Subject normalization
- Thread grouping
- Thread scoring
- Complete thread retrieval
- Chronological sorting
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

from scripts.retrieval.email_thread_retriever import ThreadRetriever
from scripts.chunking.models import Chunk


@pytest.fixture
def mock_project():
    """Create a mock ProjectManager."""
    project = Mock()
    # Mock get_task_paths to return a TaskPaths object with proper log path
    mock_task_paths = Mock()
    mock_task_paths.get_log_path = Mock(return_value="/tmp/test.log")
    project.get_task_paths = Mock(return_value=mock_task_paths)
    project.get_metadata_path = Mock(return_value=Path("/tmp/metadata.jsonl"))
    return project


@pytest.fixture
def thread_retriever(mock_project):
    """Create a ThreadRetriever instance with mocked dependencies."""
    with patch('scripts.retrieval.email_thread_retriever.RetrievalManager'), \
         patch('scripts.retrieval.email_thread_retriever.LoggerManager.get_logger') as mock_logger:
        mock_logger.return_value = Mock()
        retriever = ThreadRetriever(mock_project)
        return retriever


class TestSubjectNormalization:
    """Test subject line normalization for thread grouping."""

    def test_remove_re_prefix(self, thread_retriever):
        """Test removal of Re: prefix."""
        assert thread_retriever._normalize_subject("Re: Budget Discussion") == "budget discussion"
        assert thread_retriever._normalize_subject("RE: Budget Discussion") == "budget discussion"
        assert thread_retriever._normalize_subject("re: Budget Discussion") == "budget discussion"

    def test_remove_fwd_prefix(self, thread_retriever):
        """Test removal of Fwd: prefix."""
        assert thread_retriever._normalize_subject("Fwd: Meeting") == "meeting"
        assert thread_retriever._normalize_subject("FWD: Meeting") == "meeting"
        assert thread_retriever._normalize_subject("Fw: Meeting") == "meeting"

    def test_remove_multiple_prefixes(self, thread_retriever):
        """Test removal of multiple Re:/Fwd: prefixes."""
        assert thread_retriever._normalize_subject("Re: Fwd: Re: Budget") == "budget"
        assert thread_retriever._normalize_subject("Fwd: Re: Meeting") == "meeting"

    def test_remove_bracketed_prefixes(self, thread_retriever):
        """Test removal of bracketed prefixes like [Primo], [EXTERNAL]."""
        assert thread_retriever._normalize_subject("[Primo] Budget Discussion") == "budget discussion"
        assert thread_retriever._normalize_subject("[EXTERNAL] Meeting") == "meeting"
        assert thread_retriever._normalize_subject("[EXTERNAL *] Re: Budget") == "budget"

    def test_complex_subject_cleaning(self, thread_retriever):
        """Test complex subject with multiple prefixes."""
        subject = "[Primo] Re: [EXTERNAL *] RE: [External] Budget Discussion"
        assert thread_retriever._normalize_subject(subject) == "budget discussion"

    def test_empty_subject(self, thread_retriever):
        """Test handling of empty subject."""
        assert thread_retriever._normalize_subject("") == ""
        assert thread_retriever._normalize_subject(None) == ""

    def test_preserve_content(self, thread_retriever):
        """Test that actual content is preserved."""
        assert thread_retriever._normalize_subject("Budget Discussion") == "budget discussion"
        assert thread_retriever._normalize_subject("Q4 Planning Meeting") == "q4 planning meeting"


class TestThreadGrouping:
    """Test grouping of emails into threads."""

    def test_group_by_normalized_subject(self, thread_retriever):
        """Test that emails are grouped by normalized subject."""
        chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email 1", token_count=10,
                meta={"subject": "Budget Discussion"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email 2", token_count=10,
                meta={"subject": "Re: Budget Discussion"}
            ),
            Chunk(
                id="3", doc_id="d3", text="Email 3", token_count=10,
                meta={"subject": "Meeting Notes"}
            ),
        ]

        threads = thread_retriever._group_by_thread(chunks)

        assert len(threads) == 2  # Two distinct threads
        assert "budget discussion" in threads
        assert "meeting notes" in threads
        assert len(threads["budget discussion"]) == 2
        assert len(threads["meeting notes"]) == 1

    def test_group_with_prefixes(self, thread_retriever):
        """Test grouping with various prefixes."""
        chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email 1", token_count=10,
                meta={"subject": "[Primo] Budget"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email 2", token_count=10,
                meta={"subject": "Re: [EXTERNAL] Budget"}
            ),
            Chunk(
                id="3", doc_id="d3", text="Email 3", token_count=10,
                meta={"subject": "Fwd: Re: Budget"}
            ),
        ]

        threads = thread_retriever._group_by_thread(chunks)

        assert len(threads) == 1  # All same thread
        assert "budget" in threads
        assert len(threads["budget"]) == 3

    def test_empty_subject_handling(self, thread_retriever):
        """Test handling of emails with empty subjects."""
        chunks = [
            Chunk(
                id="1", doc_id="d1", text="Email 1", token_count=10,
                meta={"subject": ""}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email 2", token_count=10,
                meta={"subject": "Budget"}
            ),
        ]

        threads = thread_retriever._group_by_thread(chunks)

        assert len(threads) == 2
        assert "" in threads
        assert "budget" in threads


class TestThreadScoring:
    """Test thread scoring and ranking."""

    def test_score_by_seed_count(self, thread_retriever):
        """Test that threads with more seed emails score higher."""
        threads = {
            "thread1": [
                Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={"date": "2025-11-20"}),
                Chunk(id="2", doc_id="d2", text="E2", token_count=10, meta={"date": "2025-11-20"}),
            ],
            "thread2": [
                Chunk(id="3", doc_id="d3", text="E3", token_count=10, meta={"date": "2025-11-20"}),
            ],
        }

        seed_emails = [
            Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={}),
            Chunk(id="2", doc_id="d2", text="E2", token_count=10, meta={}),
        ]

        scored = thread_retriever._score_threads(threads, seed_emails)

        # thread1 should be first (2 seed emails vs 0)
        assert scored[0] == "thread1"
        assert scored[1] == "thread2"

    def test_score_by_thread_size(self, thread_retriever):
        """Test thread size scoring (prefer moderate size)."""
        threads = {
            "small": [Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={"date": "2025-11-20"})],
            "optimal": [
                Chunk(id=str(i), doc_id=f"d{i}", text=f"E{i}", token_count=10, meta={"date": "2025-11-20"})
                for i in range(2, 7)  # 5 emails
            ],
            "large": [
                Chunk(id=str(i), doc_id=f"d{i}", text=f"E{i}", token_count=10, meta={"date": "2025-11-20"})
                for i in range(7, 27)  # 20 emails
            ],
        }

        seed_emails = []  # No seed emails, test size scoring only

        scored = thread_retriever._score_threads(threads, seed_emails)

        # Optimal size should score best
        # (Note: exact order depends on scoring weights)
        assert "optimal" in scored[:2]  # Should be in top 2

    def test_recency_scoring(self, thread_retriever):
        """Test that recent threads score higher."""
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

        threads = {
            "recent": [Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={"date": today})],
            "old": [Chunk(id="2", doc_id="d2", text="E2", token_count=10, meta={"date": week_ago})],
        }

        seed_emails = []

        scored = thread_retriever._score_threads(threads, seed_emails)

        # Recent thread should score higher
        assert scored[0] == "recent"


class TestRecencyCalculation:
    """Test recency score calculation."""

    def test_today_recency(self, thread_retriever):
        """Test that today's emails get high recency score."""
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chunks = [Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={"date": today})]

        score = thread_retriever._calculate_recency_score(chunks)
        assert score >= 0.9  # Should be close to 1.0

    def test_old_recency(self, thread_retriever):
        """Test that old emails get low recency score."""
        month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        chunks = [Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={"date": month_ago})]

        score = thread_retriever._calculate_recency_score(chunks)
        assert score <= 0.1  # Should be close to 0.0

    def test_no_date(self, thread_retriever):
        """Test handling of chunks with no date."""
        chunks = [Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={})]

        score = thread_retriever._calculate_recency_score(chunks)
        assert score == 0.0

    def test_invalid_date(self, thread_retriever):
        """Test handling of invalid date format."""
        chunks = [Chunk(id="1", doc_id="d1", text="E1", token_count=10, meta={"date": "invalid"})]

        score = thread_retriever._calculate_recency_score(chunks)
        assert score == 0.5  # Default for parse errors


class TestCompleteThreadRetrieval:
    """Test retrieval of complete threads from metadata."""

    def test_get_full_thread(self, thread_retriever, tmp_path):
        """Test loading complete thread from metadata file."""
        # Create temporary metadata file
        metadata_path = tmp_path / "outlook_eml_metadata.jsonl"
        metadata = [
            {
                "id": "1",
                "doc_id": "d1",
                "subject": "Budget Discussion",
                "text": "Email 1",
                "token_count": 10,
                "date": "2025-11-20 10:00:00"
            },
            {
                "id": "2",
                "doc_id": "d2",
                "subject": "Re: Budget Discussion",
                "text": "Email 2",
                "token_count": 10,
                "date": "2025-11-20 11:00:00"
            },
            {
                "id": "3",
                "doc_id": "d3",
                "subject": "Meeting Notes",
                "text": "Email 3",
                "token_count": 10,
                "date": "2025-11-20 12:00:00"
            },
        ]

        with open(metadata_path, 'w', encoding='utf-8') as f:
            for meta in metadata:
                f.write(json.dumps(meta) + '\n')

        # Mock project to return our temp path
        thread_retriever.project.get_metadata_path = Mock(return_value=metadata_path)

        # Get full thread for "budget discussion"
        thread_chunks = thread_retriever._get_full_thread("budget discussion", "outlook_eml")

        assert len(thread_chunks) == 2  # Only budget discussion emails
        assert thread_chunks[0].id == "1"
        assert thread_chunks[1].id == "2"

    def test_get_full_thread_missing_file(self, thread_retriever):
        """Test handling of missing metadata file."""
        thread_retriever.project.get_metadata_path = Mock(return_value=Path("/nonexistent.jsonl"))

        chunks = thread_retriever._get_full_thread("thread1", "outlook_eml")

        assert chunks == []


class TestEndToEndRetrieval:
    """Test complete retrieval workflow."""

    def test_retrieve_with_mocked_manager(self, thread_retriever):
        """Test retrieve method with mocked RetrievalManager."""
        # Mock seed emails
        seed_emails = [
            Chunk(
                id="1", doc_id="d1", text="Email 1", token_count=10,
                meta={"subject": "Budget", "date": "2025-11-20 10:00:00", "doc_type": "outlook_eml"}
            ),
            Chunk(
                id="2", doc_id="d2", text="Email 2", token_count=10,
                meta={"subject": "Re: Budget", "date": "2025-11-20 11:00:00", "doc_type": "outlook_eml"}
            ),
        ]

        thread_retriever.retrieval_manager.retrieve = Mock(return_value=seed_emails)
        thread_retriever._get_full_thread = Mock(return_value=seed_emails)

        chunks = thread_retriever.retrieve("budget discussion", top_threads=1)

        assert len(chunks) == 2
        # Should be sorted chronologically
        assert chunks[0].meta["date"] <= chunks[1].meta["date"]

    def test_retrieve_no_results(self, thread_retriever):
        """Test retrieve when no seed emails are found."""
        thread_retriever.retrieval_manager.retrieve = Mock(return_value=[])

        chunks = thread_retriever.retrieve("nonexistent query")

        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
