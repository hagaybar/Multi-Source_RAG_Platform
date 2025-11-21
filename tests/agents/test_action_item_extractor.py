"""
Unit tests for ActionItemExtractor.

Tests cover:
- LLM-based extraction
- Pattern-based fallback
- Error handling
- Email formatting
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from scripts.agents.action_item_extractor import ActionItemExtractor
from scripts.chunking.models import Chunk


@pytest.fixture
def sample_emails():
    """Create sample email chunks."""
    return [
        Chunk(
            id="1", doc_id="d1",
            text="Hi team, please review the budget proposal by Friday. Can you also schedule a meeting?",
            token_count=20,
            meta={
                "sender_name": "Alice Johnson",
                "date": "2025-01-15 10:00:00",
                "subject": "Budget Review"
            }
        ),
        Chunk(
            id="2", doc_id="d2",
            text="TODO: Complete the migration plan. We need to finalize this before next week.",
            token_count=15,
            meta={
                "sender_name": "Bob Smith",
                "date": "2025-01-16 14:00:00",
                "subject": "Migration Plan"
            }
        ),
    ]


class TestLLMExtraction:
    """Test LLM-based extraction."""

    def test_llm_extraction_success(self, sample_emails):
        """Test successful LLM extraction."""
        extractor = ActionItemExtractor(use_llm=True)

        mock_response = """[
  {
    "task": "Review budget proposal",
    "deadline": "Friday",
    "assigned_to": null,
    "source_email": "#1"
  },
  {
    "task": "Schedule meeting",
    "deadline": null,
    "assigned_to": null,
    "source_email": "#1"
  },
  {
    "task": "Complete migration plan",
    "deadline": "next week",
    "assigned_to": null,
    "source_email": "#2"
  }
]"""

        with patch('scripts.agents.action_item_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = mock_response
            mock_completer_class.return_value = mock_completer

            action_items = extractor.extract(sample_emails)

            # Should call LLM
            mock_completer.get_completion.assert_called_once()

            # Should extract 3 action items
            assert len(action_items) == 3
            assert action_items[0]["task"] == "Review budget proposal"
            assert action_items[0]["deadline"] == "Friday"
            assert action_items[1]["task"] == "Schedule meeting"
            assert action_items[2]["task"] == "Complete migration plan"

    def test_llm_extraction_markdown_json(self, sample_emails):
        """Test LLM extraction with markdown-wrapped JSON."""
        extractor = ActionItemExtractor(use_llm=True)

        mock_response = """```json
[
  {
    "task": "Review proposal",
    "deadline": "Friday",
    "assigned_to": "Bob",
    "source_email": "#1"
  }
]
```"""

        with patch('scripts.agents.action_item_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = mock_response
            mock_completer_class.return_value = mock_completer

            action_items = extractor.extract(sample_emails)

            # Should parse successfully
            assert len(action_items) == 1
            assert action_items[0]["task"] == "Review proposal"

    def test_llm_extraction_error_handling(self, sample_emails):
        """Test LLM extraction error handling."""
        extractor = ActionItemExtractor(use_llm=True)

        with patch('scripts.agents.action_item_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = "[ERROR] API error"
            mock_completer_class.return_value = mock_completer

            # Should fall back to pattern matching
            action_items = extractor.extract(sample_emails)

            # Pattern matching should extract at least some items
            assert isinstance(action_items, list)

    def test_llm_extraction_invalid_json(self, sample_emails):
        """Test handling of invalid JSON from LLM."""
        extractor = ActionItemExtractor(use_llm=True)

        with patch('scripts.agents.action_item_extractor.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = "Not valid JSON"
            mock_completer_class.return_value = mock_completer

            # Should fall back to pattern matching
            action_items = extractor.extract(sample_emails)

            assert isinstance(action_items, list)


class TestPatternExtraction:
    """Test pattern-based extraction fallback."""

    def test_pattern_extraction_basic(self, sample_emails):
        """Test basic pattern extraction."""
        extractor = ActionItemExtractor(use_llm=False)

        action_items = extractor.extract(sample_emails)

        # Should extract at least the TODO item
        assert len(action_items) > 0

        # Check that one of them is the TODO item
        todo_items = [item for item in action_items if "migration plan" in item["task"].lower()]
        assert len(todo_items) > 0

    def test_pattern_extraction_please_pattern(self):
        """Test 'please' pattern extraction."""
        extractor = ActionItemExtractor(use_llm=False)

        email = Chunk(
            id="1", doc_id="d1",
            text="Please review the document and send feedback.",
            token_count=10,
            meta={"sender_name": "Alice", "date": "2025-01-15"}
        )

        action_items = extractor.extract([email])

        # Should extract the 'please' action
        assert len(action_items) > 0
        assert any("review" in item["task"].lower() for item in action_items)

    def test_pattern_extraction_deadline(self):
        """Test deadline extraction."""
        extractor = ActionItemExtractor(use_llm=False)

        email = Chunk(
            id="1", doc_id="d1",
            text="Please complete this task by January 15.",
            token_count=10,
            meta={"sender_name": "Alice", "date": "2025-01-15"}
        )

        action_items = extractor.extract([email])

        # Should extract deadline
        assert len(action_items) > 0
        # At least one item should have a deadline (January 15 matches pattern)
        assert any(item.get("deadline") for item in action_items)

    def test_pattern_extraction_empty_emails(self):
        """Test extraction from empty email list."""
        extractor = ActionItemExtractor(use_llm=False)

        action_items = extractor.extract([])

        assert action_items == []

    def test_pattern_extraction_no_action_items(self):
        """Test extraction from emails with no action items."""
        extractor = ActionItemExtractor(use_llm=False)

        email = Chunk(
            id="1", doc_id="d1",
            text="This is just a status update. Everything is going well.",
            token_count=10,
            meta={"sender_name": "Alice", "date": "2025-01-15"}
        )

        action_items = extractor.extract([email])

        # May extract nothing or very few false positives
        assert isinstance(action_items, list)


class TestEmailFormatting:
    """Test email formatting for LLM prompts."""

    def test_format_emails(self, sample_emails):
        """Test email formatting."""
        extractor = ActionItemExtractor(use_llm=False)

        formatted = extractor._format_emails(sample_emails)

        # Should contain numbered emails
        assert "Email #1" in formatted
        assert "Email #2" in formatted

        # Should contain sender names
        assert "Alice Johnson" in formatted
        assert "Bob Smith" in formatted

        # Should contain email content
        assert "budget proposal" in formatted
        assert "migration plan" in formatted

    def test_format_long_email(self):
        """Test formatting of long email (should truncate)."""
        extractor = ActionItemExtractor(use_llm=False)

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
    """Test action item result structure."""

    def test_result_has_required_fields(self, sample_emails):
        """Test that results have required fields."""
        extractor = ActionItemExtractor(use_llm=False)

        action_items = extractor.extract(sample_emails)

        if len(action_items) > 0:
            # Check first item has required fields
            item = action_items[0]
            assert "task" in item
            assert "deadline" in item
            assert "assigned_to" in item
            assert "source" in item


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
