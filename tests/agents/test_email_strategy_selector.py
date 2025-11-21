"""
Unit tests for EmailStrategySelector.

Tests cover:
- Strategy mapping for different intents
- Multi-aspect query detection
- Combined strategy handling
- Low confidence fallback
"""

import pytest

from scripts.agents.email_strategy_selector import EmailStrategySelector


@pytest.fixture
def selector():
    """Create EmailStrategySelector instance."""
    return EmailStrategySelector()


class TestBasicStrategySelection:
    """Test basic strategy selection for single-aspect queries."""

    def test_thread_summary_intent(self, selector):
        """Test thread_summary maps to thread_retrieval."""
        intent = {
            "primary_intent": "thread_summary",
            "confidence": 0.85,
            "metadata": {},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        assert strategy["primary"] == "thread_retrieval"
        assert strategy["filters"] == []

    def test_sender_query_intent(self, selector):
        """Test sender_query maps to sender_retrieval (single-aspect)."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.85,
            "metadata": {"sender": "Alice"},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        assert strategy["primary"] == "sender_retrieval"
        assert strategy["params"]["sender"] == "Alice"

    def test_temporal_query_intent(self, selector):
        """Test temporal_query maps to temporal_retrieval (single-aspect)."""
        intent = {
            "primary_intent": "temporal_query",
            "confidence": 0.85,
            "metadata": {"time_range": "last_week"},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        assert strategy["primary"] == "temporal_retrieval"
        assert strategy["params"]["time_range"] == "last_week"

    def test_factual_lookup_intent(self, selector):
        """Test factual_lookup maps to multi_aspect (adaptive)."""
        intent = {
            "primary_intent": "factual_lookup",
            "confidence": 0.85,
            "metadata": {},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        assert strategy["primary"] == "multi_aspect"


class TestMultiAspectDetection:
    """Test multi-aspect query detection."""

    def test_sender_plus_temporal(self, selector):
        """Test sender + temporal query uses multi_aspect."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.85,
            "metadata": {"sender": "Alice", "time_range": "last_week"},
            "secondary_signals": ["temporal_query"]
        }

        strategy = selector.select_strategy(intent)

        # Should use multi_aspect for combined query
        assert strategy["primary"] == "multi_aspect"
        assert strategy["params"]["sender"] == "Alice"
        assert strategy["params"]["time_range"] == "last_week"

    def test_secondary_signals_triggers_multi_aspect(self, selector):
        """Test that secondary signals trigger multi_aspect."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.85,
            "metadata": {"sender": "Alice"},
            "secondary_signals": ["temporal_query"]
        }

        strategy = selector.select_strategy(intent)

        assert strategy["primary"] == "multi_aspect"

    def test_multiple_metadata_aspects(self, selector):
        """Test multiple metadata fields trigger multi_aspect."""
        intent = {
            "primary_intent": "factual_lookup",
            "confidence": 0.85,
            "metadata": {
                "sender": "Alice",
                "time_range": "last_week",
                "topic_keywords": ["budget"]
            },
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        # 3 metadata aspects → multi_aspect
        assert strategy["primary"] == "multi_aspect"

    def test_inherently_multi_aspect_intents(self, selector):
        """Test that certain intents are inherently multi-aspect."""
        inherently_multi = ["aggregation_query", "action_items", "decision_tracking"]

        for intent_type in inherently_multi:
            intent = {
                "primary_intent": intent_type,
                "confidence": 0.85,
                "metadata": {},
                "secondary_signals": []
            }

            strategy = selector.select_strategy(intent)

            assert strategy["primary"] == "multi_aspect", \
                f"{intent_type} should use multi_aspect"


class TestLowConfidence:
    """Test low confidence handling."""

    def test_low_confidence_uses_multi_aspect(self, selector):
        """Test low confidence falls back to multi_aspect."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.3,  # Below MIN_CONFIDENCE (0.5)
            "metadata": {"sender": "Alice"},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        # Low confidence → adaptive multi_aspect
        assert strategy["primary"] == "multi_aspect"

    def test_borderline_confidence(self, selector):
        """Test confidence at threshold (0.5)."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.5,  # Exactly at threshold
            "metadata": {"sender": "Alice"},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        # At threshold → use specialized strategy
        assert strategy["primary"] == "sender_retrieval"

    def test_high_confidence(self, selector):
        """Test high confidence uses specialized strategy."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.95,
            "metadata": {"sender": "Alice"},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        assert strategy["primary"] == "sender_retrieval"


class TestMetadataPassthrough:
    """Test metadata is passed through to strategy params."""

    def test_metadata_passthrough(self, selector):
        """Test all metadata is included in params."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.85,
            "metadata": {
                "sender": "Alice",
                "time_range": "last_week",
                "topic_keywords": ["budget", "Q4"]
            },
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        # All metadata should be in params
        assert strategy["params"]["sender"] == "Alice"
        assert strategy["params"]["time_range"] == "last_week"
        assert strategy["params"]["topic_keywords"] == ["budget", "Q4"]

    def test_empty_metadata(self, selector):
        """Test empty metadata is handled correctly."""
        intent = {
            "primary_intent": "factual_lookup",
            "confidence": 0.85,
            "metadata": {},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        assert strategy["params"] == {}


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_intent(self, selector):
        """Test unknown intent falls back to multi_aspect."""
        intent = {
            "primary_intent": "unknown_intent_type",
            "confidence": 0.85,
            "metadata": {},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        # Unknown intent → multi_aspect (default from STRATEGY_MAP.get())
        assert strategy["primary"] == "multi_aspect"

    def test_missing_confidence(self, selector):
        """Test missing confidence defaults to 0.0."""
        intent = {
            "primary_intent": "sender_query",
            # No confidence field
            "metadata": {"sender": "Alice"},
            "secondary_signals": []
        }

        strategy = selector.select_strategy(intent)

        # 0.0 < MIN_CONFIDENCE → multi_aspect
        assert strategy["primary"] == "multi_aspect"

    def test_missing_secondary_signals(self, selector):
        """Test missing secondary_signals is handled."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.85,
            "metadata": {"sender": "Alice"}
            # No secondary_signals field
        }

        strategy = selector.select_strategy(intent)

        assert strategy["primary"] == "sender_retrieval"

    def test_missing_metadata(self, selector):
        """Test missing metadata is handled."""
        intent = {
            "primary_intent": "sender_query",
            "confidence": 0.85,
            "secondary_signals": []
            # No metadata field
        }

        strategy = selector.select_strategy(intent)

        assert strategy["params"] == {}


class TestFiltersField:
    """Test filters field in strategy."""

    def test_filters_always_empty(self, selector):
        """Test filters field is always empty (filters handled by multi_aspect)."""
        intents = [
            {
                "primary_intent": "sender_query",
                "confidence": 0.85,
                "metadata": {"sender": "Alice", "time_range": "last_week"},
                "secondary_signals": ["temporal_query"]
            },
            {
                "primary_intent": "thread_summary",
                "confidence": 0.85,
                "metadata": {},
                "secondary_signals": []
            }
        ]

        for intent in intents:
            strategy = selector.select_strategy(intent)
            assert strategy["filters"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
