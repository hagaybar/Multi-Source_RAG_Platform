"""
Unit tests for DecisionExtractor.

Tests cover:
- LLM-based extraction
- Error handling
- Email formatting
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from scripts.agents.decision_extractor import DecisionExtractor
from scripts.chunking.models import Chunk


@pytest.fixture
def sample_emails():
    """Create sample email chunks with decisions."""
    return [
        Chunk(
            id="1", doc_id="d1",
            text="After reviewing the proposals, I've decided to approve the budget of $50,000.",
            token_count=15,
            meta={
                "sender_name": "Sarah Wilson",
                "date": "2025-01-15 10:00:00",
                "subject": "Budget Decision"
            }
        ),
        Chunk(
            id="2", doc_id="d2",
            text="We agreed to go with Vendor A for the migration project.",
            token_count=12,
            meta={
                "sender_name": "Bob Smith",
                "date": "2025-01-16 14:00:00",
                "subject": "Vendor Selection"
            }
        ),
    ]


class TestLLMExtraction:
    """Test LLM-based extraction."""

    def test_llm_extraction_success(self, sample_emails):
        """Test successful LLM extraction."""
        extractor = DecisionExtractor(use_llm=True)

        mock_response = """[
  {
    "decision": "Approved budget of $50,000",
    "made_by": "Sarah Wilson",
    "date": "2025-01-15",
    "source_email": "#1"
  },
  {
    "decision": "Selected Vendor A for migration",
    "made_by": "Team",
    "date": "2025-01-16",
    "source_email": "#2"
  }
]"""

        with patch('scripts.agents.decision_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = mock_response
            mock_completer_class.return_value = mock_completer

            decisions = extractor.extract(sample_emails)

            # Should call LLM
            mock_completer.get_completion.assert_called_once()

            # Should extract 2 decisions
            assert len(decisions) == 2
            assert decisions[0]["decision"] == "Approved budget of $50,000"
            assert decisions[0]["made_by"] == "Sarah Wilson"
            assert decisions[1]["decision"] == "Selected Vendor A for migration"

    def test_llm_extraction_markdown_json(self, sample_emails):
        """Test LLM extraction with markdown-wrapped JSON."""
        extractor = DecisionExtractor(use_llm=True)

        mock_response = """```json
[
  {
    "decision": "Approved vendor selection",
    "made_by": "Sarah",
    "date": "2025-01-15",
    "source_email": "#1"
  }
]
```"""

        with patch('scripts.agents.decision_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = mock_response
            mock_completer_class.return_value = mock_completer

            decisions = extractor.extract(sample_emails)

            # Should parse successfully
            assert len(decisions) == 1
            assert decisions[0]["decision"] == "Approved vendor selection"

    def test_llm_extraction_error_handling(self, sample_emails):
        """Test LLM extraction error handling."""
        extractor = DecisionExtractor(use_llm=True)

        with patch('scripts.agents.decision_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = "[ERROR] API error"
            mock_completer_class.return_value = mock_completer

            # Should return empty list on error
            decisions = extractor.extract(sample_emails)

            assert decisions == []

    def test_llm_extraction_invalid_json(self, sample_emails):
        """Test handling of invalid JSON from LLM."""
        extractor = DecisionExtractor(use_llm=True)

        with patch('scripts.agents.decision_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = "Not valid JSON"
            mock_completer_class.return_value = mock_completer

            # Should return empty list on JSON error
            decisions = extractor.extract(sample_emails)

            assert decisions == []

    def test_llm_disabled(self, sample_emails):
        """Test extraction with LLM disabled."""
        extractor = DecisionExtractor(use_llm=False)

        decisions = extractor.extract(sample_emails)

        # Should return empty list (no pattern fallback for decisions)
        assert decisions == []


class TestEmailFormatting:
    """Test email formatting for LLM prompts."""

    def test_format_emails(self, sample_emails):
        """Test email formatting."""
        extractor = DecisionExtractor(use_llm=True)

        formatted = extractor._format_emails(sample_emails)

        # Should contain numbered emails
        assert "Email #1" in formatted
        assert "Email #2" in formatted

        # Should contain sender names
        assert "Sarah Wilson" in formatted
        assert "Bob Smith" in formatted

        # Should contain email content
        assert "budget" in formatted
        assert "Vendor A" in formatted

    def test_format_long_email(self):
        """Test formatting of long email (should truncate)."""
        extractor = DecisionExtractor(use_llm=True)

        long_email = Chunk(
            id="1", doc_id="d1",
            text="A" * 1000,  # 1000 characters
            token_count=250,
            meta={"sender_name": "Alice", "date": "2025-01-15"}
        )

        formatted = extractor._format_emails([long_email])

        # Should truncate to 500 characters (+ formatting)
        assert len(formatted) < 700  # 500 + some overhead for formatting


class TestResultStructure:
    """Test decision result structure."""

    def test_result_structure(self):
        """Test that results have correct structure."""
        extractor = DecisionExtractor(use_llm=True)

        mock_response = """[
  {
    "decision": "Approved",
    "made_by": "Sarah",
    "date": "2025-01-15",
    "source_email": "#1"
  }
]"""

        email = Chunk(
            id="1", doc_id="d1",
            text="Decision made.",
            token_count=5,
            meta={"sender_name": "Sarah", "date": "2025-01-15"}
        )

        with patch('scripts.agents.decision_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = mock_response
            mock_completer_class.return_value = mock_completer

            decisions = extractor.extract([email])

            # Check structure
            assert len(decisions) == 1
            assert "decision" in decisions[0]
            assert "made_by" in decisions[0]
            assert "date" in decisions[0]
            assert "source_email" in decisions[0]

    def test_empty_emails(self):
        """Test extraction from empty email list."""
        extractor = DecisionExtractor(use_llm=True)

        with patch('scripts.agents.decision_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = "[]"
            mock_completer_class.return_value = mock_completer

            decisions = extractor.extract([])

            assert decisions == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
