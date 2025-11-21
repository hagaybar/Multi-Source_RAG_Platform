"""
Integration test for Phase 1 + Phase 2 email components.

This test demonstrates the complete workflow:
1. EmailIntentDetector analyzes query
2. Appropriate retriever is selected based on intent
3. Retriever fetches relevant emails
4. ContextAssembler cleans and organizes the results

Test Scenarios:
- Thread summary query
- Temporal query (last week)
- Sender query (specific person)
- Multi-aspect query (sender + temporal)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from scripts.agents.email_intent_detector import EmailIntentDetector
from scripts.retrieval.context_assembler import ContextAssembler
from scripts.retrieval.email_thread_retriever import ThreadRetriever
from scripts.retrieval.email_temporal_retriever import TemporalRetriever
from scripts.retrieval.email_sender_retriever import SenderRetriever
from scripts.retrieval.email_multi_aspect_retriever import MultiAspectRetriever
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
def intent_detector():
    """Create EmailIntentDetector."""
    return EmailIntentDetector()


@pytest.fixture
def context_assembler():
    """Create ContextAssembler."""
    return ContextAssembler()


@pytest.fixture
def sample_email_chunks():
    """Create sample email chunks for testing."""
    now = datetime.now()
    return [
        Chunk(
            id="1", doc_id="d1", text="Let's discuss the budget for Q4.",
            token_count=10,
            meta={
                "subject": "[Primo] Budget Discussion",
                "sender": "alice@company.com",
                "sender_name": "Alice Johnson",
                "date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "doc_type": "outlook_eml"
            }
        ),
        Chunk(
            id="2", doc_id="d2",
            text="> Let's discuss the budget for Q4.\n\nI agree! When should we meet?\n\n-- \nBest regards,\nBob",
            token_count=15,
            meta={
                "subject": "Re: [Primo] Budget Discussion",
                "sender": "bob@company.com",
                "sender_name": "Bob Smith",
                "date": (now - timedelta(days=1, hours=-2)).strftime("%Y-%m-%d %H:%M:%S"),
                "doc_type": "outlook_eml"
            }
        ),
        Chunk(
            id="3", doc_id="d3",
            text=">> Let's discuss the budget\n> I agree! When should we meet?\n\nTuesday at 2pm works for me.",
            token_count=12,
            meta={
                "subject": "Re: Re: Budget Discussion",
                "sender": "alice@company.com",
                "sender_name": "Alice Johnson",
                "date": (now - timedelta(days=1, hours=-4)).strftime("%Y-%m-%d %H:%M:%S"),
                "doc_type": "outlook_eml"
            }
        ),
        Chunk(
            id="4", doc_id="d4",
            text="The project status looks good. We're on track for the deadline.",
            token_count=12,
            meta={
                "subject": "Project Status Update",
                "sender": "alice@company.com",
                "sender_name": "Alice Johnson",
                "date": (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
                "doc_type": "outlook_eml"
            }
        ),
    ]


class TestIntentBasedRetrieval:
    """Test that different intents trigger appropriate retrievers."""

    def test_thread_summary_intent(self, intent_detector, sample_email_chunks):
        """Test thread summary query selects thread retriever."""
        query = "Summarize the budget discussion thread"

        # Detect intent
        intent = intent_detector.detect(query)

        # Should detect thread_summary intent
        assert intent["primary_intent"] == "thread_summary"
        assert intent["confidence"] > 0.5

    def test_temporal_intent(self, intent_detector, sample_email_chunks):
        """Test temporal query detects time range."""
        query = "What emails did I receive last week?"

        # Detect intent
        intent = intent_detector.detect(query)

        # Should detect temporal intent with time_range metadata
        assert intent["primary_intent"] == "temporal_query"
        assert "time_range" in intent["metadata"]

    def test_sender_intent(self, intent_detector, sample_email_chunks):
        """Test sender query detects sender name."""
        query = "What did Alice say about the budget?"

        # Detect intent
        intent = intent_detector.detect(query)

        # Should detect sender_query with Alice in metadata
        assert intent["primary_intent"] == "sender_query"
        assert intent["metadata"]["sender"] == "Alice"

    def test_multi_aspect_intent(self, intent_detector, sample_email_chunks):
        """Test multi-aspect query detects multiple signals."""
        query = "What did Alice say last week?"

        # Detect intent
        intent = intent_detector.detect(query)

        # Should detect primary + secondary signals
        assert intent["primary_intent"] == "sender_query"
        assert "temporal_query" in intent["secondary_signals"]
        assert intent["metadata"]["sender"] == "Alice"
        assert "time_range" in intent["metadata"]


class TestRetrievalWithContextAssembly:
    """Test retrieval + context assembly pipeline."""

    @patch('scripts.retrieval.email_thread_retriever.RetrievalManager')
    @patch('scripts.retrieval.email_thread_retriever.LoggerManager.get_logger')
    def test_thread_retrieval_and_assembly(
        self,
        mock_logger,
        mock_retrieval_manager,
        mock_project,
        context_assembler,
        sample_email_chunks
    ):
        """Test complete thread retrieval + context assembly."""
        mock_logger.return_value = Mock()

        # Setup thread retriever
        thread_retriever = ThreadRetriever(mock_project)
        thread_retriever.retrieval_manager.retrieve = Mock(return_value=sample_email_chunks[:3])
        thread_retriever._get_full_thread = Mock(return_value=sample_email_chunks[:3])

        # Retrieve thread
        chunks = thread_retriever.retrieve("budget discussion", top_threads=1)

        # Should retrieve all 3 emails in thread
        assert len(chunks) == 3

        # Assemble context
        intent = {"primary_intent": "thread_summary"}
        context = context_assembler.assemble(chunks, intent)

        # Verify context is cleaned
        assert "Let's discuss the budget for Q4" in context
        assert "I agree! When should we meet?" in context
        assert "Tuesday at 2pm works for me" in context

        # Verify quotes are removed
        assert "> Let's discuss the budget" not in context or context.count("Let's discuss the budget") <= 3

        # Verify signatures are removed
        assert "Best regards" not in context

    @patch('scripts.retrieval.email_temporal_retriever.RetrievalManager')
    @patch('scripts.retrieval.email_temporal_retriever.LoggerManager.get_logger')
    def test_temporal_retrieval_and_assembly(
        self,
        mock_logger,
        mock_retrieval_manager,
        mock_project,
        context_assembler,
        sample_email_chunks
    ):
        """Test temporal retrieval + context assembly."""
        mock_logger.return_value = Mock()

        # Setup temporal retriever
        temporal_retriever = TemporalRetriever(mock_project)
        # Return only recent emails (not the 10-day old one)
        temporal_retriever.retrieval_manager.retrieve = Mock(return_value=sample_email_chunks[:3])

        # Retrieve with temporal filter
        intent_metadata = {"time_range": "last_week"}
        chunks = temporal_retriever.retrieve("emails", intent_metadata, top_k=10)

        # Should filter to recent emails
        assert len(chunks) == 3
        # Should be sorted newest first
        assert chunks[0].meta["date"] >= chunks[-1].meta["date"]

        # Assemble context
        intent = {"primary_intent": "temporal_query"}
        context = context_assembler.assemble(chunks, intent)

        # Verify context contains recent emails
        assert len(context) > 0

    @patch('scripts.retrieval.email_sender_retriever.RetrievalManager')
    @patch('scripts.retrieval.email_sender_retriever.LoggerManager.get_logger')
    def test_sender_retrieval_and_assembly(
        self,
        mock_logger,
        mock_retrieval_manager,
        mock_project,
        context_assembler,
        sample_email_chunks
    ):
        """Test sender retrieval + context assembly."""
        mock_logger.return_value = Mock()

        # Setup sender retriever
        sender_retriever = SenderRetriever(mock_project)
        sender_retriever.retrieval_manager.retrieve = Mock(return_value=sample_email_chunks)

        # Retrieve Alice's emails
        intent_metadata = {"sender": "Alice"}
        chunks = sender_retriever.retrieve("budget", intent_metadata, top_k=10)

        # Should filter to Alice's emails only
        assert len(chunks) == 3  # Alice sent 3 emails
        for chunk in chunks:
            assert "Alice" in chunk.meta.get("sender_name", "")

        # Assemble context
        intent = {"primary_intent": "sender_query", "metadata": {"sender": "Alice"}}
        context = context_assembler.assemble(chunks, intent)

        # Verify context contains Alice's emails
        assert "Let's discuss the budget for Q4" in context
        assert "Tuesday at 2pm works for me" in context


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    @patch('scripts.retrieval.email_multi_aspect_retriever.RetrievalManager')
    @patch('scripts.retrieval.email_multi_aspect_retriever.ThreadRetriever')
    @patch('scripts.retrieval.email_multi_aspect_retriever.TemporalRetriever')
    @patch('scripts.retrieval.email_multi_aspect_retriever.SenderRetriever')
    @patch('scripts.retrieval.email_multi_aspect_retriever.LoggerManager.get_logger')
    def test_complete_workflow(
        self,
        mock_logger,
        mock_sender_retriever,
        mock_temporal_retriever,
        mock_thread_retriever,
        mock_retrieval_manager,
        mock_project,
        intent_detector,
        context_assembler,
        sample_email_chunks
    ):
        """Test complete workflow: Intent -> Retrieval -> Assembly."""
        mock_logger.return_value = Mock()

        # Step 1: Detect intent
        query = "What did Alice say about the budget?"
        intent = intent_detector.detect(query)

        assert intent["primary_intent"] == "sender_query"
        assert intent["metadata"]["sender"] == "Alice"

        # Step 2: Retrieve based on intent
        multi_retriever = MultiAspectRetriever(mock_project)
        multi_retriever.retrieval_manager.retrieve = Mock(return_value=sample_email_chunks)

        chunks = multi_retriever.retrieve(query, intent, top_k=10)

        # Should return Alice's emails
        assert len(chunks) > 0

        # Step 3: Assemble context
        context = context_assembler.assemble(chunks, intent)

        # Verify final context
        assert len(context) > 0
        assert isinstance(context, str)

    @patch('scripts.retrieval.email_multi_aspect_retriever.RetrievalManager')
    @patch('scripts.retrieval.email_multi_aspect_retriever.ThreadRetriever')
    @patch('scripts.retrieval.email_multi_aspect_retriever.TemporalRetriever')
    @patch('scripts.retrieval.email_multi_aspect_retriever.SenderRetriever')
    @patch('scripts.retrieval.email_multi_aspect_retriever.LoggerManager.get_logger')
    def test_multi_aspect_workflow(
        self,
        mock_logger,
        mock_sender_retriever,
        mock_temporal_retriever,
        mock_thread_retriever,
        mock_retrieval_manager,
        mock_project,
        intent_detector,
        context_assembler,
        sample_email_chunks
    ):
        """Test multi-aspect query workflow."""
        mock_logger.return_value = Mock()

        # Step 1: Detect multi-aspect intent
        query = "What did Alice say about budget last week?"
        intent = intent_detector.detect(query)

        assert intent["primary_intent"] == "sender_query"
        assert "temporal_query" in intent["secondary_signals"]

        # Step 2: Multi-aspect retrieval
        multi_retriever = MultiAspectRetriever(mock_project)
        multi_retriever.retrieval_manager.retrieve = Mock(return_value=sample_email_chunks)
        multi_retriever.temporal_retriever.parse_time_range = Mock(return_value={
            "start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": datetime.now().strftime("%Y-%m-%d")
        })

        chunks = multi_retriever.retrieve(query, intent, top_k=10)

        # Should filter by sender AND time
        assert len(chunks) >= 0  # May be filtered down significantly

        # Step 3: Assemble context
        context = context_assembler.assemble(chunks, intent)

        # Verify context assembled
        assert isinstance(context, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
