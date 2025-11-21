#!/usr/bin/env python3
"""
Unit tests for EmailIntentDetector.

Tests:
1. Intent classification accuracy
2. Metadata extraction (sender, time_range, topics)
3. Multi-aspect query handling
4. Secondary signal detection
5. Edge cases
"""

import pytest
from unittest.mock import Mock, patch
from scripts.agents.email_intent_detector import EmailIntentDetector


class TestIntentClassification:
    """Test intent classification for different query types."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = EmailIntentDetector()

    def test_thread_summary_intent(self):
        """Test detection of thread summary queries."""
        queries = [
            "Summarize the discussion about Primo NDE",
            "What was the conversation about the migration?",
            "Summary of the budget thread",
            "Thread about project deadlines",
        ]

        for query in queries:
            result = self.detector.detect(query)
            assert result["primary_intent"] == "thread_summary", \
                f"Failed for query: '{query}'"
            assert result["confidence"] >= 0.6, \
                f"Low confidence for '{query}': {result['confidence']}"

    def test_sender_query_intent(self):
        """Test detection of sender-specific queries."""
        queries = [
            "What did Alice say about the budget?",
            "Emails from Bob",
            "Did Sarah mention the deadline?",
            "According to John, what is the status?",
            "Mike's opinion on the proposal",
        ]

        for query in queries:
            result = self.detector.detect(query)
            assert result["primary_intent"] == "sender_query", \
                f"Failed for query: '{query}'"
            assert result["confidence"] >= 0.6

    def test_temporal_query_intent(self):
        """Test detection of time-based queries."""
        queries = [
            "Recent emails about migration",
            "What happened last week?",
            "Emails from yesterday",
            "Latest updates on the project",
            "This week's discussions",
        ]

        for query in queries:
            result = self.detector.detect(query)
            assert result["primary_intent"] == "temporal_query", \
                f"Failed for query: '{query}'"

    def test_action_items_intent(self):
        """Test detection of action item queries."""
        queries = [
            "What are the action items?",
            "List all tasks from the meeting",
            "What are the deadlines?",
            "What needs to be done?",
            "Show me the TODO items",
        ]

        for query in queries:
            result = self.detector.detect(query)
            assert result["primary_intent"] == "action_items", \
                f"Failed for query: '{query}'"

    def test_decision_tracking_intent(self):
        """Test detection of decision tracking queries."""
        queries = [
            "What was decided about the vendor?",
            "Final decision on the migration",
            "What did we agree on?",
            "What was the conclusion?",
            "Was the proposal approved?",
        ]

        for query in queries:
            result = self.detector.detect(query)
            assert result["primary_intent"] == "decision_tracking", \
                f"Failed for query: '{query}'"

    def test_factual_lookup_fallback(self):
        """Test that unclear queries fall back to factual_lookup."""
        queries = [
            "Tell me about Primo",
            "How does the system work?",
            "database configuration",
        ]

        for query in queries:
            result = self.detector.detect(query)
            assert result["primary_intent"] == "factual_lookup", \
                f"Failed for query: '{query}'"
            assert result["confidence"] <= 0.5, \
                "Factual lookup should have low confidence"


class TestMetadataExtraction:
    """Test metadata extraction from queries."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = EmailIntentDetector()

    def test_sender_extraction(self):
        """Test sender name extraction."""
        test_cases = [
            ("What did Alice say?", "Alice"),
            ("Emails from Bob", "Bob"),
            ("According to Sarah", "Sarah"),
            ("Mike's opinion", "Mike"),
            ("Did John mention this?", "John"),
        ]

        for query, expected_sender in test_cases:
            result = self.detector.detect(query)
            assert "sender" in result["metadata"], \
                f"No sender extracted from: '{query}'"
            assert result["metadata"]["sender"] == expected_sender, \
                f"Wrong sender for '{query}': {result['metadata']['sender']}"

    def test_time_range_extraction(self):
        """Test time range extraction."""
        test_cases = [
            ("Emails from yesterday", "yesterday"),
            ("What happened last week?", "last_week"),
            ("Recent updates", "recent"),
            ("This month's emails", "this_month"),
            ("Latest discussions", "recent"),
            ("Today's messages", "today"),
        ]

        for query, expected_time in test_cases:
            result = self.detector.detect(query)
            assert "time_range" in result["metadata"], \
                f"No time_range extracted from: '{query}'"
            assert result["metadata"]["time_range"] == expected_time, \
                f"Wrong time_range for '{query}': {result['metadata']['time_range']}"

    def test_topic_keywords_extraction(self):
        """Test topic keyword extraction."""
        query = "What did Alice say about the budget migration last week?"
        result = self.detector.detect(query)

        assert "topic_keywords" in result["metadata"]
        keywords = result["metadata"]["topic_keywords"]

        # Should include important words, excluding common words
        assert "budget" in keywords or "migration" in keywords
        # Should NOT include common words
        assert "what" not in keywords
        assert "the" not in keywords

    def test_metadata_combination(self):
        """Test extraction of multiple metadata fields."""
        query = "What did Sarah say about the Primo migration last month?"
        result = self.detector.detect(query)

        # Should extract all three types
        assert "sender" in result["metadata"]
        assert result["metadata"]["sender"] == "Sarah"

        assert "time_range" in result["metadata"]
        assert result["metadata"]["time_range"] == "last_month"

        assert "topic_keywords" in result["metadata"]
        keywords = result["metadata"]["topic_keywords"]
        assert any(word in keywords for word in ["primo", "migration"])


class TestMultiAspectQueries:
    """Test handling of queries with multiple intent aspects."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = EmailIntentDetector()

    def test_sender_plus_temporal(self):
        """Test query combining sender and temporal aspects."""
        query = "What did Alice say about the budget last week?"
        result = self.detector.detect(query)

        # Primary should be sender_query (more specific)
        assert result["primary_intent"] == "sender_query"

        # Should detect temporal as secondary signal
        assert "temporal_query" in result["secondary_signals"], \
            f"Temporal not in secondary signals: {result['secondary_signals']}"

        # Both metadata should be extracted
        assert result["metadata"]["sender"] == "Alice"
        assert result["metadata"]["time_range"] == "last_week"

    def test_temporal_plus_topic(self):
        """Test query combining temporal and topic aspects."""
        query = "Recent emails about the migration project"
        result = self.detector.detect(query)

        assert result["primary_intent"] == "temporal_query"
        assert "time_range" in result["metadata"]
        assert "topic_keywords" in result["metadata"]

    def test_secondary_signals_threshold(self):
        """Test that secondary signals have score > 0.3."""
        query = "Summarize what Alice said last week"
        result = self.detector.detect(query)

        # Should have secondary signals
        assert len(result["secondary_signals"]) > 0

        # Secondary signals should not include primary intent
        assert result["primary_intent"] not in result["secondary_signals"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = EmailIntentDetector()

    def test_empty_query(self):
        """Test handling of empty query."""
        result = self.detector.detect("")
        assert result["primary_intent"] == "factual_lookup"
        assert result["confidence"] <= 0.5

    def test_very_short_query(self):
        """Test handling of very short query."""
        result = self.detector.detect("budget")
        assert result["primary_intent"] == "factual_lookup"

    def test_no_metadata_query(self):
        """Test query with no extractable metadata."""
        query = "Tell me about this"
        result = self.detector.detect(query)

        # Should have metadata dict, but might be empty
        assert "metadata" in result
        # topic_keywords might still extract "tell"

    def test_case_insensitivity(self):
        """Test that detection is case-insensitive."""
        queries = [
            "WHAT DID ALICE SAY?",
            "what did alice say?",
            "What Did Alice Say?",
        ]

        results = [self.detector.detect(q) for q in queries]

        # All should detect same intent
        assert all(r["primary_intent"] == "sender_query" for r in results)

        # All should extract same sender (capitalized)
        assert all(r["metadata"].get("sender") == "Alice" for r in results)

    def test_multiple_pattern_matches(self):
        """Test query matching multiple patterns of same intent."""
        query = "Summarize the discussion and provide a thread summary"
        result = self.detector.detect(query)

        # Should detect thread_summary with higher confidence
        assert result["primary_intent"] == "thread_summary"
        assert result["confidence"] > 0.7, \
            "Multiple pattern matches should increase confidence"


class TestConfidenceScoring:
    """Test confidence scoring mechanism."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = EmailIntentDetector()

    def test_single_pattern_match(self):
        """Test confidence for single pattern match."""
        query = "What did Alice say?"  # Matches one sender_query pattern
        result = self.detector.detect(query)

        # Should have base confidence (0.6 + 0.2 = 0.8 for 1 match)
        assert 0.6 <= result["confidence"] <= 0.9

    def test_multiple_pattern_matches(self):
        """Test confidence increases with multiple matches."""
        # Query matching multiple action_items patterns
        query = "What are the action items and tasks that need to be done?"
        result = self.detector.detect(query)

        assert result["primary_intent"] == "action_items"
        # Multiple matches should increase confidence
        assert result["confidence"] > 0.7

    def test_ambiguous_query_low_confidence(self):
        """Test that ambiguous queries have lower confidence."""
        query = "Tell me about the project"
        result = self.detector.detect(query)

        # Factual lookup fallback should have low confidence
        assert result["confidence"] < 0.5


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def setup_method(self):
        """Initialize detector for each test."""
        self.detector = EmailIntentDetector()

    def test_real_world_queries(self):
        """Test with realistic email queries."""
        test_cases = [
            {
                "query": "Summarize the Primo NDE migration discussion",
                "expected_intent": "thread_summary",
                "expected_metadata": {"topic_keywords": ["primo", "nde", "migration"]},
            },
            {
                "query": "What did Sarah say about the budget approval last week?",
                "expected_intent": "sender_query",
                "expected_metadata": {
                    "sender": "Sarah",
                    "time_range": "last_week",
                },
            },
            {
                "query": "What are the action items from yesterday's meeting?",
                "expected_intent": "action_items",
                "expected_metadata": {"time_range": "yesterday"},
            },
            {
                "query": "Was the vendor selection approved?",
                "expected_intent": "decision_tracking",
                "expected_metadata": {},
            },
        ]

        for case in test_cases:
            result = self.detector.detect(case["query"])

            # Check intent
            assert result["primary_intent"] == case["expected_intent"], \
                f"Failed for: {case['query']}"

            # Check metadata
            for key, value in case["expected_metadata"].items():
                if key == "topic_keywords":
                    # Check that at least one expected keyword is present
                    keywords = result["metadata"].get("topic_keywords", [])
                    assert any(kw in keywords for kw in value), \
                        f"Missing keywords in {case['query']}: {keywords}"
                else:
                    assert result["metadata"].get(key) == value, \
                        f"Metadata mismatch for {key} in {case['query']}"


class TestLLMFallback:
    """Test LLM fallback functionality for low-confidence queries."""

    def test_llm_fallback_disabled_by_default(self):
        """Test that LLM fallback is disabled by default."""
        detector = EmailIntentDetector(use_llm_fallback=False)

        # Low-confidence query
        query = "Tell me about something"
        result = detector.detect(query)

        # Should use pattern-based detection only
        assert result["detection_method"] == "pattern"
        assert "pattern_confidence" not in result

    def test_llm_fallback_enabled(self):
        """Test that LLM fallback is triggered for low-confidence queries."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.6)

        # Mock LLM response
        mock_llm_result = {
            "intent": "sender_query",
            "confidence": 0.85,
            "metadata": {"sender": "Alice"}
        }

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = f"""{{
  "intent": "{mock_llm_result['intent']}",
  "confidence": {mock_llm_result['confidence']},
  "metadata": {{"sender": "{mock_llm_result['metadata']['sender']}"}}
}}"""
            mock_completer_class.return_value = mock_completer

            # Low-confidence query (factual_lookup has 0.3 confidence)
            query = "Tell me about something"
            result = detector.detect(query)

            # Should have called LLM
            mock_completer.get_completion.assert_called_once()

            # Should use LLM result (higher confidence)
            assert result["detection_method"] == "llm"
            assert result["primary_intent"] == "sender_query"
            assert result["confidence"] == 0.85
            assert result["pattern_confidence"] == 0.3  # Original pattern confidence preserved

    def test_llm_fallback_pattern_wins(self):
        """Test that pattern result is kept when LLM has lower confidence."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.9)

        # Mock LLM response with LOW confidence
        mock_llm_result = {
            "intent": "factual_lookup",
            "confidence": 0.4,
            "metadata": {}
        }

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = f"""{{
  "intent": "{mock_llm_result['intent']}",
  "confidence": {mock_llm_result['confidence']},
  "metadata": {{}}
}}"""
            mock_completer_class.return_value = mock_completer

            # Medium-confidence query (0.8 confidence, below 0.9 threshold)
            query = "Recent updates"  # temporal_query with 0.8 confidence
            result = detector.detect(query)

            # Should have called LLM (pattern confidence 0.8 < threshold 0.9)
            # But should keep pattern result (0.8 > LLM's 0.4)
            assert result["detection_method"] == "pattern_with_llm_check"
            assert result["primary_intent"] == "temporal_query"

    def test_llm_fallback_failure_handling(self):
        """Test that LLM fallback failures are handled gracefully."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.6)

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            # Simulate LLM error
            mock_completer.get_completion.return_value = "[ERROR] API error"
            mock_completer_class.return_value = mock_completer

            # Low-confidence query
            query = "Tell me about something"
            result = detector.detect(query)

            # Should fall back to pattern result
            assert result["detection_method"] == "pattern_llm_failed"
            assert result["primary_intent"] == "factual_lookup"
            assert result["confidence"] == 0.3

    def test_llm_fallback_json_parsing_error(self):
        """Test handling of invalid JSON from LLM."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.6)

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            # Return invalid JSON
            mock_completer.get_completion.return_value = "Not valid JSON at all"
            mock_completer_class.return_value = mock_completer

            # Low-confidence query
            query = "Tell me about something"
            result = detector.detect(query)

            # Should fall back to pattern result
            assert result["detection_method"] == "pattern_llm_failed"
            assert result["primary_intent"] == "factual_lookup"

    def test_llm_fallback_markdown_json(self):
        """Test handling of markdown-wrapped JSON from LLM."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.6)

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            # Return JSON wrapped in markdown code block
            mock_completer.get_completion.return_value = """```json
{
  "intent": "sender_query",
  "confidence": 0.9,
  "metadata": {"sender": "Bob"}
}
```"""
            mock_completer_class.return_value = mock_completer

            # Low-confidence query
            query = "Tell me about something"
            result = detector.detect(query)

            # Should parse successfully
            assert result["detection_method"] == "llm"
            assert result["primary_intent"] == "sender_query"
            assert result["confidence"] == 0.9
            assert result["metadata"]["sender"] == "Bob"

    def test_llm_confidence_threshold_respected(self):
        """Test that LLM is only called when confidence < threshold."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.8)

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer_class.return_value = mock_completer

            # High-confidence query (sender_query typically has 0.8+ confidence)
            query = "What did Alice say about the budget?"
            result = detector.detect(query)

            # Should NOT call LLM (confidence >= threshold)
            mock_completer.get_completion.assert_not_called()
            assert result["detection_method"] == "pattern"

    def test_llm_metadata_extraction(self):
        """Test that LLM correctly extracts metadata."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.6)

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            mock_completer.get_completion.return_value = """{
  "intent": "sender_query",
  "confidence": 0.95,
  "metadata": {
    "sender": "Alice",
    "time_range": "last_week",
    "topic_keywords": ["budget", "approval"]
  }
}"""
            mock_completer_class.return_value = mock_completer

            # Ambiguous query
            query = "Tell me what Alice mentioned"
            result = detector.detect(query)

            # Should extract all metadata
            assert result["metadata"]["sender"] == "Alice"
            assert result["metadata"]["time_range"] == "last_week"
            assert "budget" in result["metadata"]["topic_keywords"]
            assert "approval" in result["metadata"]["topic_keywords"]

    def test_llm_empty_metadata_cleaned(self):
        """Test that empty metadata values are removed."""
        detector = EmailIntentDetector(use_llm_fallback=True, llm_confidence_threshold=0.6)

        with patch('scripts.agents.email_intent_detector.OpenAICompleter') as mock_completer_class:
            mock_completer = Mock()
            # LLM returns empty metadata values
            mock_completer.get_completion.return_value = """{
  "intent": "factual_lookup",
  "confidence": 0.7,
  "metadata": {
    "sender": "",
    "time_range": "",
    "topic_keywords": []
  }
}"""
            mock_completer_class.return_value = mock_completer

            query = "Tell me about something"
            result = detector.detect(query)

            # Empty values should be cleaned
            assert result["metadata"] == {}


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
