"""
Email Intent Detector

Detects user intent from email queries with multi-aspect support.
Extracts metadata like sender names, time ranges, and topics.
"""

import re
from typing import Dict, List
from scripts.utils.logger import LoggerManager

logger = LoggerManager.get_logger("email_intent_detector")


class EmailIntentDetector:
    """
    Detects user intent from email queries with multi-aspect support.

    Supports intents:
    - thread_summary: Summarize email discussions
    - sender_query: Find emails from specific person
    - temporal_query: Find emails in time range
    - action_items: Extract tasks and deadlines
    - decision_tracking: Find decisions made
    - aggregation_query: Analysis queries (most/least/top/compare)
    - factual_lookup: Standard information retrieval
    """

    def __init__(self):
        """Initialize intent patterns."""
        self.patterns = {
            "thread_summary": [
                r"summarize.*(?:discussion|thread|conversation|exchange)",
                r"what.*(?:conversation|thread|exchange)",
                r"(?:thread|discussion|conversation)\s+about",
                r"summary of.*(?:emails|discussion|thread)",
            ],
            "sender_query": [
                r"what did (?!yesterday|today|last|this)(\w+) say",
                r"(\w+)'s (?:opinion|view|response|thoughts?|email)",
                r"emails from (?!yesterday|today|last|this|recent)(\w+)",
                r"did (?!yesterday|today|last|this)(\w+) mention",
                r"(\w+) said",
                r"according to (\w+)",
            ],
            "temporal_query": [
                r"\b(?:recent|latest|newest)\b(?!\s+action)",
                r"\blast (?:week|month|day|year)\b",
                r"\byesterday\b",
                r"\bthis (?:week|month|year)\b",
                r"\bin the past",
                r"\btoday\b",
            ],
            "action_items": [
                r"action items?",
                r"(?:what are|list|show).*\btasks?\b",
                r"(?:what are|list|show).*\bdeadlines?\b",
                r"\btodo\b",
                r"need to (?:do|complete)",
                r"what needs to be done",
            ],
            "decision_tracking": [
                r"what was decided",
                r"final decision",
                r"agree[d]? (?:on|to)",
                r"\bconclusion\b",
                r"decision about",
                r"approved",
            ],
            "aggregation_query": [
                r"\b(?:most|least|top|bottom)\s+(?:discussed|mentioned|common|frequent)",
                r"most.*(?:problem|issue|topic|question)",
                r"what are the (?:main|primary|key) (?:issues|topics|problems)",
                r"(?:how many|count|number of).*(?:emails|messages|discussions)",
                r"compare.*(?:discussion|emails|threads)",
                r"(?:frequently|commonly) (?:discussed|mentioned)",
                r"biggest (?:issue|problem|concern)",
            ],
        }

        # Priority weights for breaking ties (higher = more specific)
        self.intent_priorities = {
            "aggregation_query": 3,  # Most specific (requires analysis)
            "action_items": 3,
            "decision_tracking": 3,
            "thread_summary": 3,
            "sender_query": 2,  # Medium specificity (filter)
            "temporal_query": 1,  # Least specific (filter)
            "factual_lookup": 0,
        }

    def detect(self, query: str) -> Dict:
        """
        Detect intent with multi-aspect metadata extraction.

        Args:
            query: User query string

        Returns:
            {
                "primary_intent": "sender_query",
                "confidence": 0.85,
                "metadata": {
                    "sender": "Alice",
                    "time_range": "last_week",
                    "topic_keywords": ["budget"]
                },
                "secondary_signals": ["temporal_query"]
            }
        """
        # Score all intents
        intent_scores = self._score_patterns(query)

        # Get primary intent
        if not intent_scores or max(intent_scores.values()) == 0:
            primary_intent = "factual_lookup"
            confidence = 0.3
        else:
            # Use priority to break ties when scores are equal
            primary_intent = max(
                intent_scores.keys(),
                key=lambda intent: (
                    intent_scores[intent],
                    self.intent_priorities.get(intent, 0)
                )
            )
            confidence = min(intent_scores[primary_intent], 1.0)

        # Extract metadata
        metadata = self._extract_metadata(query)

        # Detect secondary signals (intents with score > 0.3 that aren't primary)
        secondary = [
            intent
            for intent, score in intent_scores.items()
            if score > 0.3 and intent != primary_intent
        ]

        result = {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "metadata": metadata,
            "secondary_signals": secondary,
        }

        logger.debug(
            f"Intent detected: {primary_intent} (confidence: {confidence:.2f})",
            extra={"intent_result": result}
        )

        return result

    def _score_patterns(self, query: str) -> Dict[str, float]:
        """
        Score query against all intent patterns.

        Returns:
            {"thread_summary": 0.8, "sender_query": 0.3, ...}
        """
        query_lower = query.lower()
        scores = {}

        for intent, patterns in self.patterns.items():
            score = 0.0
            matches = 0

            for pattern in patterns:
                if re.search(pattern, query_lower, re.I):
                    matches += 1

            # Score based on number of pattern matches
            if matches > 0:
                # More matches = higher confidence
                base_score = 0.6 + (matches * 0.2)
                score = min(base_score, 1.0)

            scores[intent] = score

        return scores

    def _extract_metadata(self, query: str) -> Dict:
        """
        Extract sender names, time ranges, topics from query.

        Returns:
            {
                "sender": "Alice",
                "time_range": "last_week",
                "topic_keywords": ["budget", "approval"]
            }
        """
        metadata = {}

        # Extract sender name
        sender_patterns = [
            r"(?:from|by|what did|did)\s+(\w+)",
            r"(\w+)'s (?:opinion|view|email|response)",
            r"according to (\w+)",
        ]

        # Temporal keywords to exclude from sender names
        temporal_keywords = {
            "yesterday", "today", "tomorrow", "recent", "latest",
            "last", "this", "next", "past", "week", "month", "year",
        }

        for pattern in sender_patterns:
            match = re.search(pattern, query, re.I)
            if match:
                # Get the captured name (first group)
                sender = match.group(1)
                # Skip if it's a temporal keyword
                if sender.lower() not in temporal_keywords:
                    # Capitalize first letter
                    metadata["sender"] = sender.capitalize()
                    break

        # Extract time range
        time_patterns = {
            "yesterday": r"\byesterday\b",
            "today": r"\btoday\b",
            "last_week": r"\blast week\b",
            "last_month": r"\blast month\b",
            "this_week": r"\bthis week\b",
            "this_month": r"\bthis month\b",
            "recent": r"\b(?:recent|latest|newest)\b",
        }

        for time_range, pattern in time_patterns.items():
            if re.search(pattern, query, re.I):
                metadata["time_range"] = time_range
                break

        # Extract topic keywords (simple approach: nouns/important words)
        # Remove common words
        common_words = {
            "what", "did", "say", "about", "the", "is", "was", "were",
            "from", "to", "in", "on", "at", "by", "for", "with", "a",
            "an", "and", "or", "but", "if", "then", "recent", "latest",
            "emails", "email", "discussion", "thread", "conversation",
        }

        words = query.lower().split()
        topic_keywords = [
            w.strip("?,!.")
            for w in words
            if w.strip("?,!.") not in common_words and len(w) > 2
        ]

        if topic_keywords:
            metadata["topic_keywords"] = topic_keywords

        return metadata


if __name__ == "__main__":
    # Quick test
    detector = EmailIntentDetector()

    test_queries = [
        "Summarize the discussion about Primo NDE",
        "What did Alice say about the budget last week?",
        "Recent emails about migration",
        "What are the action items from the project emails?",
        "What was decided about the vendor selection?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = detector.detect(query)
        print(f"  Intent: {result['primary_intent']} (conf: {result['confidence']:.2f})")
        print(f"  Metadata: {result['metadata']}")
        if result['secondary_signals']:
            print(f"  Secondary: {result['secondary_signals']}")
