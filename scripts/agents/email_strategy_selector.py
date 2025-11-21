"""
EmailStrategySelector - Selects retrieval strategy based on detected intent.

This selector maps intent types to retrieval strategies and supports
combined strategies (e.g., sender + temporal filtering).

Example:
    selector = EmailStrategySelector()
    intent = {
        "primary_intent": "sender_query",
        "metadata": {"sender": "Alice", "time_range": "last_week"},
        "secondary_signals": ["temporal_query"]
    }
    strategy = selector.select_strategy(intent)
    # Returns: {
    #     "primary": "multi_aspect",
    #     "filters": [],
    #     "params": {"sender": "Alice", "time_range": "last_week"}
    # }
"""

from typing import Dict, List, Optional


class EmailStrategySelector:
    """
    Selects retrieval strategy based on intent, supports combined strategies.

    Strategy Mapping:
    - thread_summary → thread_retrieval
    - sender_query → multi_aspect (if multi-aspect) or sender_retrieval
    - temporal_query → multi_aspect (if multi-aspect) or temporal_retrieval
    - aggregation_query → multi_aspect (needs multiple perspectives)
    - action_items → multi_aspect (may need temporal context)
    - decision_tracking → multi_aspect (may need thread context)
    - factual_lookup → multi_aspect (adaptive fallback)
    """

    # Strategy mapping for primary intents
    STRATEGY_MAP = {
        "thread_summary": "thread_retrieval",
        "sender_query": "sender_retrieval",
        "temporal_query": "temporal_retrieval",
        "aggregation_query": "multi_aspect",
        "action_items": "multi_aspect",
        "decision_tracking": "multi_aspect",
        "factual_lookup": "multi_aspect",
    }

    # Minimum confidence threshold for using specialized strategies
    MIN_CONFIDENCE = 0.5

    def select_strategy(self, intent: Dict) -> Dict:
        """
        Select retrieval strategy based on intent.

        Args:
            intent: Intent detection result from EmailIntentDetector
                {
                    "primary_intent": "sender_query",
                    "confidence": 0.85,
                    "metadata": {"sender": "Alice", "time_range": "last_week"},
                    "secondary_signals": ["temporal_query"]
                }

        Returns:
            Strategy dictionary:
                {
                    "primary": "multi_aspect",  # Primary retrieval strategy
                    "filters": [],  # Additional filters to apply
                    "params": {...}  # Parameters from intent metadata
                }
        """
        primary_intent = intent.get("primary_intent", "factual_lookup")
        confidence = intent.get("confidence", 0.0)
        secondary_signals = intent.get("secondary_signals", [])
        metadata = intent.get("metadata", {})

        # Low confidence → use adaptive multi_aspect strategy
        if confidence < self.MIN_CONFIDENCE:
            return {
                "primary": "multi_aspect",
                "filters": [],
                "params": metadata
            }

        # Check if this is a multi-aspect query
        is_multi_aspect = self._is_multi_aspect_query(primary_intent, secondary_signals, metadata)

        if is_multi_aspect:
            # Use multi-aspect retriever for complex queries
            return {
                "primary": "multi_aspect",
                "filters": [],
                "params": metadata
            }

        # Single-aspect query: use specialized retriever
        primary_strategy = self.STRATEGY_MAP.get(primary_intent, "multi_aspect")

        return {
            "primary": primary_strategy,
            "filters": [],
            "params": metadata
        }

    def _is_multi_aspect_query(
        self,
        primary_intent: str,
        secondary_signals: List[str],
        metadata: Dict
    ) -> bool:
        """
        Determine if query requires multi-aspect retrieval.

        A query is multi-aspect if:
        1. Has secondary signals (e.g., sender + temporal)
        2. Has multiple metadata fields (e.g., sender + time_range)
        3. Is inherently multi-aspect (aggregation_query, action_items, decision_tracking)

        Args:
            primary_intent: Primary intent type
            secondary_signals: List of secondary intent signals
            metadata: Extracted metadata (sender, time_range, etc.)

        Returns:
            True if multi-aspect retrieval needed, False otherwise
        """
        # Inherently multi-aspect intents
        if primary_intent in ["aggregation_query", "action_items", "decision_tracking"]:
            return True

        # Has secondary signals
        if len(secondary_signals) > 0:
            return True

        # Has multiple metadata aspects
        metadata_aspects = 0
        if metadata.get("sender"):
            metadata_aspects += 1
        if metadata.get("time_range"):
            metadata_aspects += 1
        if metadata.get("topic_keywords"):
            metadata_aspects += 1

        if metadata_aspects >= 2:
            return True

        return False
