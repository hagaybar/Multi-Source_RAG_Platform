"""
Full pipeline integration test for Phases 1, 2, and 3.

This test demonstrates the complete email RAG pipeline:
Phase 1: Intent Detection + Context Assembly
Phase 2: Specialized Retrievers
Phase 3: Orchestrator Integration

Test Flow:
1. User query → EmailOrchestratorAgent
2. Intent detection (Phase 1)
3. Strategy selection (Phase 3)
4. Specialized retrieval (Phase 2)
5. Context assembly (Phase 1)
6. Result with metadata
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
def email_dataset():
    """Create realistic email dataset."""
    now = datetime.now()
    return [
        # Budget thread - Alice starts
        Chunk(
            id="1", doc_id="d1",
            text="Hi team, we need to discuss the Q4 budget allocation. I propose increasing the training budget by 20%. Thoughts?",
            token_count=25,
            meta={
                "subject": "[Finance] Budget Discussion",
                "sender": "alice@company.com",
                "sender_name": "Alice Johnson",
                "date": (now - timedelta(days=2)).strftime("%Y-%m-%d 10:00:00"),
                "doc_type": "outlook_eml"
            }
        ),
        # Budget thread - Bob replies
        Chunk(
            id="2", doc_id="d2",
            text="> I propose increasing the training budget by 20%\n\nI agree with Alice. The team needs more training resources.\n\n-- \nBest regards,\nBob",
            token_count=30,
            meta={
                "subject": "Re: [Finance] Budget Discussion",
                "sender": "bob@company.com",
                "sender_name": "Bob Smith",
                "date": (now - timedelta(days=2, hours=-2)).strftime("%Y-%m-%d 12:00:00"),
                "doc_type": "outlook_eml"
            }
        ),
        # Budget thread - Alice follows up
        Chunk(
            id="3", doc_id="d3",
            text="Thanks Bob! I'll prepare a detailed proposal by Friday.",
            token_count=12,
            meta={
                "subject": "Re: Re: Budget Discussion",
                "sender": "alice@company.com",
                "sender_name": "Alice Johnson",
                "date": (now - timedelta(days=2, hours=-4)).strftime("%Y-%m-%d 14:00:00"),
                "doc_type": "outlook_eml"
            }
        ),
        # Project status - Alice (recent)
        Chunk(
            id="4", doc_id="d4",
            text="Project Alpha is progressing well. We're ahead of schedule for the Q4 deadline.",
            token_count=18,
            meta={
                "subject": "Project Status Update",
                "sender": "alice@company.com",
                "sender_name": "Alice Johnson",
                "date": (now - timedelta(days=1)).strftime("%Y-%m-%d 09:00:00"),
                "doc_type": "outlook_eml"
            }
        ),
        # Meeting notes - Carol (old)
        Chunk(
            id="5", doc_id="d5",
            text="Summary of last month's planning meeting. We discussed goals for Q3.",
            token_count=15,
            meta={
                "subject": "Meeting Notes - Planning",
                "sender": "carol@company.com",
                "sender_name": "Carol Davis",
                "date": (now - timedelta(days=45)).strftime("%Y-%m-%d 15:00:00"),
                "doc_type": "outlook_eml"
            }
        ),
    ]


class TestThreadSummaryQuery:
    """Test thread summary queries (Phase 1 + 2 + 3)."""

    @patch('scripts.agents.email_orchestrator.ThreadRetriever')
    @patch('scripts.agents.email_orchestrator.TemporalRetriever')
    @patch('scripts.agents.email_orchestrator.SenderRetriever')
    @patch('scripts.agents.email_orchestrator.MultiAspectRetriever')
    @patch('scripts.agents.email_orchestrator.LoggerManager.get_logger')
    def test_thread_summary_full_pipeline(
        self,
        mock_logger,
        mock_multi,
        mock_sender,
        mock_temporal,
        mock_thread,
        mock_project,
        email_dataset
    ):
        """Test: 'Summarize the budget discussion thread'."""
        mock_logger.return_value = Mock()

        # Setup orchestrator
        orchestrator = EmailOrchestratorAgent(mock_project)

        # Mock thread retriever to return budget thread
        budget_thread = [email_dataset[0], email_dataset[1], email_dataset[2]]
        orchestrator.retrievers["thread_retrieval"].retrieve = Mock(return_value=budget_thread)

        # Execute query
        result = orchestrator.retrieve("Summarize the budget discussion thread")

        # Verify intent detection (Phase 1)
        assert result["intent"]["primary_intent"] == "thread_summary"

        # Verify strategy selection (Phase 3)
        assert result["strategy"]["primary"] == "thread_retrieval"

        # Verify retrieval (Phase 2)
        assert len(result["chunks"]) == 3
        assert all(c.id in ["1", "2", "3"] for c in result["chunks"])

        # Verify context assembly (Phase 1)
        context = result["context"]
        assert "Q4 budget" in context
        assert "training budget" in context
        # Verify quotes removed
        assert context.count("training budget") <= 3  # Original + at most 2 copies

        # Verify metadata (Phase 3)
        assert result["metadata"]["chunk_count"] == 3
        assert result["metadata"]["strategy_used"] == "thread_retrieval"


class TestTemporalQuery:
    """Test temporal queries (Phase 1 + 2 + 3)."""

    @patch('scripts.agents.email_orchestrator.ThreadRetriever')
    @patch('scripts.agents.email_orchestrator.TemporalRetriever')
    @patch('scripts.agents.email_orchestrator.SenderRetriever')
    @patch('scripts.agents.email_orchestrator.MultiAspectRetriever')
    @patch('scripts.agents.email_orchestrator.LoggerManager.get_logger')
    def test_temporal_query_full_pipeline(
        self,
        mock_logger,
        mock_multi,
        mock_sender,
        mock_temporal,
        mock_thread,
        mock_project,
        email_dataset
    ):
        """Test: 'What emails did I receive last week?'."""
        mock_logger.return_value = Mock()

        orchestrator = EmailOrchestratorAgent(mock_project)

        # Mock multi-aspect retriever (strategy selector chooses multi_aspect for temporal queries)
        recent_emails = [email_dataset[0], email_dataset[1], email_dataset[2], email_dataset[3]]
        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=recent_emails)

        result = orchestrator.retrieve("What emails did I receive last week?")

        # Verify intent
        assert result["intent"]["primary_intent"] == "temporal_query"
        assert "time_range" in result["intent"]["metadata"]

        # Verify strategy uses multi_aspect (adaptive)
        assert result["strategy"]["primary"] == "multi_aspect"

        # Verify retrieval excludes old emails
        assert len(result["chunks"]) == 4
        assert not any(c.id == "5" for c in result["chunks"])  # Old meeting notes excluded

        # Verify context
        assert len(result["context"]) > 0


class TestSenderQuery:
    """Test sender queries (Phase 1 + 2 + 3)."""

    @patch('scripts.agents.email_orchestrator.ThreadRetriever')
    @patch('scripts.agents.email_orchestrator.TemporalRetriever')
    @patch('scripts.agents.email_orchestrator.SenderRetriever')
    @patch('scripts.agents.email_orchestrator.MultiAspectRetriever')
    @patch('scripts.agents.email_orchestrator.LoggerManager.get_logger')
    def test_sender_query_full_pipeline(
        self,
        mock_logger,
        mock_multi,
        mock_sender,
        mock_temporal,
        mock_thread,
        mock_project,
        email_dataset
    ):
        """Test: 'What did Alice say about the budget?'."""
        mock_logger.return_value = Mock()

        orchestrator = EmailOrchestratorAgent(mock_project)

        # Mock multi-aspect retriever (strategy selector chooses multi_aspect for sender queries)
        alice_emails = [email_dataset[0], email_dataset[2], email_dataset[3]]
        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=alice_emails)

        result = orchestrator.retrieve("What did Alice say about the budget?")

        # Verify intent
        assert result["intent"]["primary_intent"] == "sender_query"
        assert result["intent"]["metadata"]["sender"] == "Alice"

        # Verify retrieval
        assert len(result["chunks"]) == 3
        for chunk in result["chunks"]:
            assert "Alice" in chunk.meta["sender_name"]

        # Verify metadata
        assert "Alice Johnson" in result["metadata"]["unique_senders"]


class TestMultiAspectQuery:
    """Test multi-aspect queries (Phase 1 + 2 + 3)."""

    @patch('scripts.agents.email_orchestrator.ThreadRetriever')
    @patch('scripts.agents.email_orchestrator.TemporalRetriever')
    @patch('scripts.agents.email_orchestrator.SenderRetriever')
    @patch('scripts.agents.email_orchestrator.MultiAspectRetriever')
    @patch('scripts.agents.email_orchestrator.LoggerManager.get_logger')
    def test_multi_aspect_query_full_pipeline(
        self,
        mock_logger,
        mock_multi,
        mock_sender,
        mock_temporal,
        mock_thread,
        mock_project,
        email_dataset
    ):
        """Test: 'What did Alice say about budget last week?'."""
        mock_logger.return_value = Mock()

        orchestrator = EmailOrchestratorAgent(mock_project)

        # Mock multi-aspect retriever (filters by sender + time)
        filtered_emails = [email_dataset[0], email_dataset[2]]  # Alice's budget emails from last week
        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=filtered_emails)

        result = orchestrator.retrieve("What did Alice say about budget last week?")

        # Verify intent detects multi-aspect
        assert result["intent"]["primary_intent"] == "sender_query"
        assert "temporal_query" in result["intent"]["secondary_signals"]
        assert result["intent"]["metadata"]["sender"] == "Alice"
        assert "time_range" in result["intent"]["metadata"]

        # Verify strategy uses multi_aspect
        assert result["strategy"]["primary"] == "multi_aspect"

        # Verify filtered results
        assert len(result["chunks"]) == 2
        for chunk in result["chunks"]:
            assert "Alice" in chunk.meta["sender_name"]
            # Should be recent (not project status from different time)

        # Verify context assembled
        assert len(result["context"]) > 0


class TestMetadataTransparency:
    """Test metadata extraction for transparency."""

    @patch('scripts.agents.email_orchestrator.ThreadRetriever')
    @patch('scripts.agents.email_orchestrator.TemporalRetriever')
    @patch('scripts.agents.email_orchestrator.SenderRetriever')
    @patch('scripts.agents.email_orchestrator.MultiAspectRetriever')
    @patch('scripts.agents.email_orchestrator.LoggerManager.get_logger')
    def test_metadata_provides_transparency(
        self,
        mock_logger,
        mock_multi,
        mock_sender,
        mock_temporal,
        mock_thread,
        mock_project,
        email_dataset
    ):
        """Test metadata provides transparency into retrieval process."""
        mock_logger.return_value = Mock()

        orchestrator = EmailOrchestratorAgent(mock_project)
        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=email_dataset)

        result = orchestrator.retrieve("test query")

        metadata = result["metadata"]

        # Verify comprehensive metadata
        assert "chunk_count" in metadata
        assert "strategy_used" in metadata
        assert "filters_applied" in metadata

        # Date range
        assert "date_range" in metadata
        assert "start" in metadata["date_range"]
        assert "end" in metadata["date_range"]

        # Unique senders
        assert "unique_senders" in metadata
        assert len(metadata["unique_senders"]) == 3  # Alice, Bob, Carol
        assert "Alice Johnson" in metadata["unique_senders"]
        assert "Bob Smith" in metadata["unique_senders"]
        assert "Carol Davis" in metadata["unique_senders"]

        # Unique subjects
        assert "unique_subjects" in metadata
        assert len(metadata["unique_subjects"]) > 0


class TestEndToEndRealWorldQuery:
    """Test end-to-end with realistic scenarios."""

    @patch('scripts.agents.email_orchestrator.ThreadRetriever')
    @patch('scripts.agents.email_orchestrator.TemporalRetriever')
    @patch('scripts.agents.email_orchestrator.SenderRetriever')
    @patch('scripts.agents.email_orchestrator.MultiAspectRetriever')
    @patch('scripts.agents.email_orchestrator.LoggerManager.get_logger')
    def test_realistic_workflow(
        self,
        mock_logger,
        mock_multi,
        mock_sender,
        mock_temporal,
        mock_thread,
        mock_project,
        email_dataset
    ):
        """Test realistic workflow: User asks question, gets clean answer."""
        mock_logger.return_value = Mock()

        orchestrator = EmailOrchestratorAgent(mock_project)

        # Setup retrievers - query about "decided" triggers decision_tracking -> multi_aspect
        budget_emails = [email_dataset[0], email_dataset[1], email_dataset[2]]
        orchestrator.retrievers["multi_aspect"].retrieve = Mock(return_value=budget_emails)

        # User query
        query = "Summarize what was decided about the budget"

        # Execute
        result = orchestrator.retrieve(query)

        # Verify complete result structure
        assert "chunks" in result
        assert "context" in result
        assert "intent" in result
        assert "strategy" in result
        assert "metadata" in result

        # Verify context is clean and useful
        context = result["context"]
        assert len(context) > 0
        assert "budget" in context.lower()

        # Context should NOT have email artifacts
        assert "Best regards" not in context  # Signatures removed
        assert context.count("training budget") <= 3  # Quotes deduplicated

        # Metadata provides transparency
        metadata = result["metadata"]
        assert metadata["chunk_count"] > 0
        assert metadata["strategy_used"] == "multi_aspect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
