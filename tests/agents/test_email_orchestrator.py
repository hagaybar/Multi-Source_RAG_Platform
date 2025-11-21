"""
Unit tests for EmailOrchestratorAgent.

Tests cover:
- Complete retrieval orchestration
- Strategy execution for different intents
- Context assembly integration
- Metadata extraction
- Logging
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from scripts.agents.email_orchestrator import EmailOrchestratorAgent
from scripts.chunking.models import Chunk


@pytest.fixture
def mock_project():
    """Create a mock ProjectManager."""
    project = Mock()
    mock_task_paths = Mock()
    mock_task_paths.get_log_path = Mock(return_value="/tmp/test.log")
    project.get_task_paths = Mock(return_value=mock_task_paths)
    return project


@pytest.fixture
def sample_chunks():
    """Create sample email chunks."""
    now = datetime.now()
    return [
        Chunk(
            id="1", doc_id="d1", text="Budget discussion email", token_count=10,
            meta={
                "subject": "Budget Discussion",
                "sender": "alice@company.com",
                "sender_name": "Alice Johnson",
                "date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "doc_type": "outlook_eml"
            }
        ),
        Chunk(
            id="2", doc_id="d2", text="Budget reply", token_count=10,
            meta={
                "subject": "Re: Budget Discussion",
                "sender": "bob@company.com",
                "sender_name": "Bob Smith",
                "date": (now - timedelta(days=1, hours=-2)).strftime("%Y-%m-%d %H:%M:%S"),
                "doc_type": "outlook_eml"
            }
        ),
    ]


@pytest.fixture
def orchestrator(mock_project):
    """Create EmailOrchestratorAgent with mocked components."""
    with patch('scripts.agents.email_orchestrator.ThreadRetriever'), \
         patch('scripts.agents.email_orchestrator.TemporalRetriever'), \
         patch('scripts.agents.email_orchestrator.SenderRetriever'), \
         patch('scripts.agents.email_orchestrator.MultiAspectRetriever'), \
         patch('scripts.agents.email_orchestrator.LoggerManager.get_logger') as mock_logger:
        mock_logger.return_value = Mock()
        agent = EmailOrchestratorAgent(mock_project)
        return agent


class TestOrchestrationPipeline:
    """Test complete orchestration pipeline."""

    def test_retrieve_basic_flow(self, orchestrator, sample_chunks):
        """Test basic retrieve() workflow."""
        # Mock intent detection
        orchestrator.intent_detector.detect = Mock(return_value={
            "primary_intent": "sender_query",
            "confidence": 0.85,
            "metadata": {"sender": "Alice"},
            "secondary_signals": []
        })

        # Mock strategy selection
        orchestrator.strategy_selector.select_strategy = Mock(return_value={
            "primary": "sender_retrieval",
            "filters": [],
            "params": {"sender": "Alice"}
        })

        # Mock retrieval
        orchestrator.retrievers["sender_retrieval"].retrieve = Mock(return_value=sample_chunks)

        # Mock context assembly
        orchestrator.context_assembler.assemble = Mock(return_value="Assembled context")

        # Execute
        result = orchestrator.retrieve("What did Alice say?")

        # Verify all components called
        orchestrator.intent_detector.detect.assert_called_once_with("What did Alice say?")
        orchestrator.strategy_selector.select_strategy.assert_called_once()
        orchestrator.retrievers["sender_retrieval"].retrieve.assert_called_once()
        orchestrator.context_assembler.assemble.assert_called_once()

        # Verify result structure
        assert "chunks" in result
        assert "context" in result
        assert "intent" in result
        assert "strategy" in result
        assert "metadata" in result

    def test_retrieve_returns_correct_structure(self, orchestrator, sample_chunks):
        """Test that retrieve() returns complete result structure."""
        # Mock all components
        orchestrator.intent_detector.detect = Mock(return_value={
            "primary_intent": "factual_lookup",
            "confidence": 0.85,
            "metadata": {},
            "secondary_signals": []
        })

        orchestrator.strategy_selector.select_strategy = Mock(return_value={
            "primary": "multi_aspect",
            "filters": [],
            "params": {}
        })

        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=sample_chunks)
        orchestrator.context_assembler.assemble = Mock(return_value="Test context")

        result = orchestrator.retrieve("test query")

        # Verify result keys
        assert result["chunks"] == sample_chunks
        assert result["context"] == "Test context"
        assert result["intent"]["primary_intent"] == "factual_lookup"
        assert result["strategy"]["primary"] == "multi_aspect"
        assert isinstance(result["metadata"], dict)


class TestStrategyExecution:
    """Test execution of different retrieval strategies."""

    def test_execute_thread_retrieval(self, orchestrator, sample_chunks):
        """Test thread retrieval execution."""
        strategy = {
            "primary": "thread_retrieval",
            "filters": [],
            "params": {}
        }
        intent = {"primary_intent": "thread_summary"}

        orchestrator.retrievers["thread_retrieval"].retrieve = Mock(return_value=sample_chunks)

        chunks = orchestrator._execute_retrieval("query", strategy, intent, top_k=15)

        # Thread retrieval called with top_threads parameter
        orchestrator.retrievers["thread_retrieval"].retrieve.assert_called_once_with(
            "query",
            top_threads=2
        )
        assert chunks == sample_chunks

    def test_execute_sender_retrieval(self, orchestrator, sample_chunks):
        """Test sender retrieval execution."""
        strategy = {
            "primary": "sender_retrieval",
            "filters": [],
            "params": {"sender": "Alice"}
        }
        intent = {"primary_intent": "sender_query"}

        orchestrator.retrievers["sender_retrieval"].retrieve = Mock(return_value=sample_chunks)

        chunks = orchestrator._execute_retrieval("query", strategy, intent, top_k=15)

        # Sender retrieval called with intent_metadata
        orchestrator.retrievers["sender_retrieval"].retrieve.assert_called_once_with(
            "query",
            intent_metadata={"sender": "Alice"},
            top_k=15
        )
        assert chunks == sample_chunks

    def test_execute_temporal_retrieval(self, orchestrator, sample_chunks):
        """Test temporal retrieval execution."""
        strategy = {
            "primary": "temporal_retrieval",
            "filters": [],
            "params": {"time_range": "last_week"}
        }
        intent = {"primary_intent": "temporal_query"}

        orchestrator.retrievers["temporal_retrieval"].retrieve = Mock(return_value=sample_chunks)

        chunks = orchestrator._execute_retrieval("query", strategy, intent, top_k=15)

        orchestrator.retrievers["temporal_retrieval"].retrieve.assert_called_once_with(
            "query",
            intent_metadata={"time_range": "last_week"},
            top_k=15
        )
        assert chunks == sample_chunks

    def test_execute_multi_aspect_retrieval(self, orchestrator, sample_chunks):
        """Test multi-aspect retrieval execution."""
        strategy = {
            "primary": "multi_aspect",
            "filters": [],
            "params": {"sender": "Alice", "time_range": "last_week"}
        }
        intent = {
            "primary_intent": "sender_query",
            "metadata": {"sender": "Alice", "time_range": "last_week"},
            "secondary_signals": ["temporal_query"]
        }

        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=sample_chunks)

        chunks = orchestrator._execute_retrieval("query", strategy, intent, top_k=15)

        # Multi-aspect uses intent directly
        orchestrator.retrievers["multi_aspect"].retrieve.assert_called_once_with(
            "query",
            intent=intent,
            top_k=15
        )
        assert chunks == sample_chunks

    def test_execute_unknown_strategy_fallback(self, orchestrator, sample_chunks):
        """Test unknown strategy falls back to multi_aspect."""
        strategy = {
            "primary": "unknown_strategy",
            "filters": [],
            "params": {}
        }
        intent = {"primary_intent": "factual_lookup"}

        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=sample_chunks)

        chunks = orchestrator._execute_retrieval("query", strategy, intent, top_k=15)

        # Should fall back to multi_aspect
        orchestrator.retrievers["multi_aspect"].retrieve.assert_called_once()
        assert chunks == sample_chunks


class TestMetadataExtraction:
    """Test metadata extraction methods."""

    def test_build_metadata_basic(self, orchestrator, sample_chunks):
        """Test basic metadata building."""
        strategy = {
            "primary": "sender_retrieval",
            "filters": []
        }

        metadata = orchestrator._build_metadata(sample_chunks, strategy)

        assert metadata["chunk_count"] == 2
        assert metadata["strategy_used"] == "sender_retrieval"
        assert metadata["filters_applied"] == []

    def test_get_date_range(self, orchestrator, sample_chunks):
        """Test date range extraction."""
        date_range = orchestrator._get_date_range(sample_chunks)

        assert date_range is not None
        assert "start" in date_range
        assert "end" in date_range
        # First chunk is more recent
        assert date_range["start"] <= date_range["end"]

    def test_get_date_range_no_dates(self, orchestrator):
        """Test date range with no dates."""
        chunks = [
            Chunk(id="1", doc_id="d1", text="Email", token_count=10, meta={})
        ]

        date_range = orchestrator._get_date_range(chunks)

        assert date_range is None

    def test_get_unique_senders(self, orchestrator, sample_chunks):
        """Test unique sender extraction."""
        senders = orchestrator._get_unique_senders(sample_chunks)

        assert len(senders) == 2
        assert "Alice Johnson" in senders
        assert "Bob Smith" in senders
        # Should be sorted
        assert senders == sorted(senders)

    def test_get_unique_senders_no_senders(self, orchestrator):
        """Test unique senders with no sender_name."""
        chunks = [
            Chunk(id="1", doc_id="d1", text="Email", token_count=10, meta={})
        ]

        senders = orchestrator._get_unique_senders(chunks)

        assert senders == []

    def test_get_unique_subjects(self, orchestrator, sample_chunks):
        """Test unique subject extraction."""
        # Need to mock the thread retriever's normalization
        mock_thread_retriever = orchestrator.retrievers["thread_retrieval"]
        mock_thread_retriever._normalize_subject = Mock(side_effect=lambda s: s.lower().replace("re: ", ""))

        subjects = orchestrator._get_unique_subjects(sample_chunks)

        assert len(subjects) > 0
        # Should be normalized
        mock_thread_retriever._normalize_subject.assert_called()

    def test_get_unique_subjects_limit(self, orchestrator):
        """Test subject limit (max 5)."""
        # Create 10 chunks with different subjects
        chunks = [
            Chunk(
                id=str(i), doc_id=f"d{i}", text=f"Email {i}", token_count=10,
                meta={"subject": f"Subject {i}"}
            )
            for i in range(10)
        ]

        mock_thread_retriever = orchestrator.retrievers["thread_retrieval"]
        mock_thread_retriever._normalize_subject = Mock(side_effect=lambda s: s.lower())

        subjects = orchestrator._get_unique_subjects(chunks)

        # Should be limited to 5
        assert len(subjects) <= 5


class TestTopKParameter:
    """Test top_k parameter handling."""

    def test_top_k_passed_to_retrievers(self, orchestrator, sample_chunks):
        """Test top_k is passed to retrievers."""
        strategy = {
            "primary": "sender_retrieval",
            "filters": [],
            "params": {"sender": "Alice"}
        }
        intent = {"primary_intent": "sender_query"}

        orchestrator.retrievers["sender_retrieval"].retrieve = Mock(return_value=sample_chunks)

        orchestrator._execute_retrieval("query", strategy, intent, top_k=20)

        # Should pass top_k=20
        orchestrator.retrievers["sender_retrieval"].retrieve.assert_called_once_with(
            "query",
            intent_metadata={"sender": "Alice"},
            top_k=20
        )

    def test_default_top_k(self, orchestrator, sample_chunks):
        """Test default top_k=15 in retrieve()."""
        orchestrator.intent_detector.detect = Mock(return_value={
            "primary_intent": "factual_lookup",
            "confidence": 0.85,
            "metadata": {},
            "secondary_signals": []
        })

        orchestrator.strategy_selector.select_strategy = Mock(return_value={
            "primary": "multi_aspect",
            "filters": [],
            "params": {}
        })

        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=sample_chunks)
        orchestrator.context_assembler.assemble = Mock(return_value="context")

        # Don't specify top_k
        orchestrator.retrieve("query")

        # Should use default top_k=15
        call_args = orchestrator.retrievers["multi_aspect"].retrieve.call_args
        assert call_args[1]["top_k"] == 15


class TestMaxTokensParameter:
    """Test max_tokens parameter for context assembly."""

    def test_max_tokens_passed_to_assembler(self, orchestrator, sample_chunks):
        """Test max_tokens is passed to context assembler."""
        orchestrator.intent_detector.detect = Mock(return_value={
            "primary_intent": "factual_lookup",
            "confidence": 0.85,
            "metadata": {},
            "secondary_signals": []
        })

        orchestrator.strategy_selector.select_strategy = Mock(return_value={
            "primary": "multi_aspect",
            "filters": [],
            "params": {}
        })

        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=sample_chunks)
        orchestrator.context_assembler.assemble = Mock(return_value="context")

        # Specify max_tokens
        orchestrator.retrieve("query", max_tokens=6000)

        # Should pass max_tokens to assembler
        call_args = orchestrator.context_assembler.assemble.call_args
        assert call_args[1]["max_tokens"] == 6000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
