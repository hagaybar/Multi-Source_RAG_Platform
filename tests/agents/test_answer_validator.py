"""
Unit tests for AnswerValidator.

Tests cover:
- Format checking for different intents
- Unsupported claims detection
- LLM-based contradiction detection
- Confidence scoring
"""

import pytest
from unittest.mock import Mock, patch
from scripts.agents.answer_validator import AnswerValidator


class TestFormatChecking:
    """Test format checking for different intent types."""

    def test_action_items_with_list_format(self):
        """Test action items query with proper list format."""
        validator = AnswerValidator(use_llm=False)

        answer = """Here are the action items:
- Complete budget report by Friday
- Schedule meeting
• Review vendor proposals"""
        context = "Email about tasks budget vendor"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_action_items_without_list_format(self):
        """Test action items query without list format."""
        validator = AnswerValidator(use_llm=False)

        answer = "You need to complete the budget report and schedule a meeting with Alice."
        context = "Email about tasks"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is False
        assert "no list format" in result["issues"][0]
        assert "bulleted list" in result["suggestions"][0]

    def test_action_items_numbered_list(self):
        """Test action items with numbered list."""
        validator = AnswerValidator(use_llm=False)

        answer = """Action items:
1. Complete budget report
2. Schedule meeting
3. Review proposals"""
        context = "Email about tasks: 1 budget 2 meeting 3 proposals"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is True

    def test_sender_query_with_sender_mentioned(self):
        """Test sender query with sender mentioned in answer."""
        validator = AnswerValidator(use_llm=False)

        answer = "Alice said the budget needs to increase by 20% for Q4."
        context = "From: Alice\nBudget increase needed by 20% for Q4."
        intent = {"primary_intent": "sender_query", "metadata": {"sender": "Alice"}}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is True

    def test_sender_query_without_sender_mentioned(self):
        """Test sender query without sender mentioned in answer."""
        validator = AnswerValidator(use_llm=False)

        answer = "The budget needs to increase by 20% for Q4."
        context = "From: Alice\nBudget increase needed."
        intent = {"primary_intent": "sender_query", "metadata": {"sender": "Alice"}}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is False
        assert "Alice" in result["issues"][0]
        assert "references what Alice said" in result["suggestions"][0]

    def test_sender_query_case_insensitive(self):
        """Test sender mention is case-insensitive."""
        validator = AnswerValidator(use_llm=False)

        answer = "ALICE mentioned the budget increase."
        context = "From: Alice\nBudget increase needed."
        intent = {"primary_intent": "sender_query", "metadata": {"sender": "Alice"}}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is True

    def test_decision_tracking_with_decision_language(self):
        """Test decision query with decision language."""
        validator = AnswerValidator(use_llm=False)

        answer = "The team decided to approve the vendor proposal."
        context = "Email about vendor selection"
        intent = {"primary_intent": "decision_tracking"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is True

    def test_decision_tracking_without_decision_language(self):
        """Test decision query without decision language."""
        validator = AnswerValidator(use_llm=False)

        answer = "The vendor proposal was discussed in the meeting."
        context = "Email about vendor selection"
        intent = {"primary_intent": "decision_tracking"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is False
        assert "no decision-related language" in result["issues"][0]

    def test_factual_lookup_no_format_requirements(self):
        """Test factual lookup has no specific format requirements."""
        validator = AnswerValidator(use_llm=False)

        answer = "The project is progressing well with no major issues."
        context = "Email about project status"
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        # Format check should pass (no requirements for factual_lookup)
        # Only unsupported claims might cause issues
        assert "no list format" not in str(result["issues"])


class TestUnsupportedClaims:
    """Test unsupported claims detection."""

    def test_numbers_in_both_answer_and_context(self):
        """Test numbers that appear in both answer and context."""
        validator = AnswerValidator(use_llm=False)

        answer = "The project has 150 tasks and needs $50,000."
        context = "Project status: 150 tasks remaining, budget $50,000."
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is True
        assert not any("Numbers not in context" in issue for issue in result["issues"])

    def test_unsupported_numbers(self):
        """Test numbers that don't appear in context."""
        validator = AnswerValidator(use_llm=False)

        answer = "The project has 150 tasks and needs $50,000 budget."
        context = "The project is ongoing."
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is False
        assert "unsupported claims" in result["issues"][0]
        # Check suggestions contain the specific numbers
        suggestions_text = " ".join(result["suggestions"])
        assert "150" in suggestions_text
        assert "50" in suggestions_text or "000" in suggestions_text  # From $50,000

    def test_proper_nouns_in_both(self):
        """Test proper nouns that appear in both answer and context."""
        validator = AnswerValidator(use_llm=False)

        answer = "Alice and Bob discussed the Primo migration with Carol."
        context = "Alice, Bob, and Carol had a meeting about Primo migration."
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        # Should be valid (all names are in context)
        assert result["is_valid"] is True

    def test_unsupported_names_threshold(self):
        """Test that a few missing names are allowed (false positive prevention)."""
        validator = AnswerValidator(use_llm=False)

        # 2 missing names should be allowed
        answer = "Alice and Bob discussed the project."
        context = "Project discussion happened."
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        # Should pass (only 2 missing names)
        assert not any("names not in context" in issue for issue in result["issues"])

    def test_many_unsupported_names(self):
        """Test that many missing names are flagged."""
        validator = AnswerValidator(use_llm=False)

        # 6 missing names should be flagged (threshold > 2)
        answer = "Alice, Bob, Carol, David, Emily, and Frank discussed with George."
        context = "Discussion happened."
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        # Should fail (too many missing names)
        assert result["is_valid"] is False
        # Check suggestions instead of issues (the detailed list is in suggestions)
        suggestions_text = " ".join(result["suggestions"])
        assert "potential names" in suggestions_text.lower() or "alice" in suggestions_text.lower()


class TestLLMContradictionDetection:
    """Test LLM-based contradiction detection."""

    def test_llm_disabled_by_default(self):
        """Test that LLM contradiction check is disabled by default."""
        validator = AnswerValidator(use_llm=False)

        answer = "The budget increased by 50%."
        context = "The budget decreased by 20%."
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        # Should NOT detect contradiction without LLM
        # (only heuristic checks run)
        assert "contradiction" not in str(result["issues"]).lower()

    def test_llm_contradiction_detected(self):
        """Test LLM detects contradiction."""
        validator = AnswerValidator(use_llm=True)

        with patch('scripts.agents.answer_validator.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = """{
  "has_contradiction": true,
  "detail": "Answer says budget increased by 50%, but context shows it decreased by 20%"
}"""
            mock_completer_class.return_value = mock_completer

            answer = "The budget increased by 50%."
            context = "The budget decreased by 20%."
            intent = {"primary_intent": "factual_lookup"}

            result = validator.validate(answer, context, intent)

            # Should detect contradiction
            assert result["is_valid"] is False
            assert "contradiction" in result["issues"][0].lower()
            assert "50%" in result["suggestions"][0] or "20%" in result["suggestions"][0]

    def test_llm_no_contradiction(self):
        """Test LLM finds no contradiction."""
        validator = AnswerValidator(use_llm=True)

        with patch('scripts.agents.answer_validator.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = """{
  "has_contradiction": false,
  "detail": ""
}"""
            mock_completer_class.return_value = mock_completer

            answer = "The budget increased by 20%."
            context = "Budget approved for 20% increase."
            intent = {"primary_intent": "factual_lookup"}

            result = validator.validate(answer, context, intent)

            # Should be valid (no contradiction)
            assert result["is_valid"] is True

    def test_llm_contradiction_check_failure(self):
        """Test handling of LLM API failure."""
        validator = AnswerValidator(use_llm=True)

        with patch('scripts.agents.answer_validator.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = "[ERROR] API error"
            mock_completer_class.return_value = mock_completer

            answer = "The budget increased."
            context = "Budget discussion."
            intent = {"primary_intent": "factual_lookup"}

            result = validator.validate(answer, context, intent)

            # Should gracefully handle error (not crash)
            # Still runs other checks
            assert "issues" in result
            assert "confidence" in result

    def test_llm_json_parsing_error(self):
        """Test handling of invalid JSON from LLM."""
        validator = AnswerValidator(use_llm=True)

        with patch('scripts.agents.answer_validator.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = "Not valid JSON"
            mock_completer_class.return_value = mock_completer

            answer = "The budget increased."
            context = "Budget discussion."
            intent = {"primary_intent": "factual_lookup"}

            result = validator.validate(answer, context, intent)

            # Should gracefully handle parsing error
            assert "issues" in result

    def test_llm_markdown_json(self):
        """Test handling of markdown-wrapped JSON."""
        validator = AnswerValidator(use_llm=True)

        with patch('scripts.agents.answer_validator.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = """```json
{
  "has_contradiction": true,
  "detail": "Conflicting information found"
}
```"""
            mock_completer_class.return_value = mock_completer

            answer = "Test answer"
            context = "Test context"
            intent = {"primary_intent": "factual_lookup"}

            result = validator.validate(answer, context, intent)

            # Should parse successfully
            assert result["is_valid"] is False
            assert "contradiction" in result["issues"][0].lower()


class TestConfidenceScoring:
    """Test confidence scoring mechanism."""

    def test_perfect_validation(self):
        """Test confidence is 1.0 for perfect validation."""
        validator = AnswerValidator(use_llm=False)

        answer = "- Complete budget report\n- Schedule meeting"
        context = "Tasks to do"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        assert result["confidence"] == 1.0

    def test_one_issue_reduces_confidence(self):
        """Test one issue reduces confidence by 0.2."""
        validator = AnswerValidator(use_llm=False)

        answer = "Complete budget report and schedule meeting"  # No list format
        context = "Tasks to do"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        assert result["confidence"] == 0.8

    def test_multiple_issues_reduce_confidence(self):
        """Test multiple issues reduce confidence proportionally."""
        validator = AnswerValidator(use_llm=False)

        # Missing list format + unsupported numbers
        answer = "Complete 150 tasks and get $50,000 budget"
        context = "Tasks to do"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        # 2 issues = 0.6 confidence
        assert result["confidence"] == 0.6

    def test_confidence_floor_at_zero(self):
        """Test confidence doesn't go below 0.0."""
        validator = AnswerValidator(use_llm=False)

        # Many issues
        answer = "Do 100, 200, 300, 400, 500, 600 tasks"  # 6 unsupported numbers + no list
        context = "Tasks"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        # Should be 0.0, not negative
        assert result["confidence"] >= 0.0


class TestValidationResult:
    """Test validation result structure."""

    def test_result_has_all_fields(self):
        """Test result has all required fields."""
        validator = AnswerValidator(use_llm=False)

        answer = "Test answer"
        context = "Test context"
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        assert "is_valid" in result
        assert "issues" in result
        assert "suggestions" in result
        assert "confidence" in result

    def test_valid_result_structure(self):
        """Test valid result has correct structure."""
        validator = AnswerValidator(use_llm=False)

        answer = "The project is progressing well."
        context = "Project status: progressing well."
        intent = {"primary_intent": "factual_lookup"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is True
        assert result["issues"] == []
        assert result["suggestions"] == []
        assert result["confidence"] == 1.0

    def test_invalid_result_structure(self):
        """Test invalid result has issues and suggestions."""
        validator = AnswerValidator(use_llm=False)

        answer = "Complete the tasks"
        context = "Tasks"
        intent = {"primary_intent": "action_items"}

        result = validator.validate(answer, context, intent)

        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert len(result["suggestions"]) > 0
        assert result["confidence"] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
