"""
Email Intent Detector

Detects user intent from email queries with multi-aspect support.
Extracts metadata like sender names, time ranges, and topics.

Enhanced with optional LLM fallback for ambiguous queries.
"""

import re
import json
from typing import Dict, List, Optional
from scripts.utils.logger import LoggerManager
from scripts.api_clients.openai.completer import OpenAICompleter

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

    Features:
    - Pattern-based detection (fast, free)
    - Optional LLM fallback for ambiguous queries (accurate, costs ~$0.001/query)
    """

    def __init__(self, use_llm_fallback: bool = False, llm_confidence_threshold: float = 0.6):
        """
        Initialize intent detector.

        Args:
            use_llm_fallback: Enable LLM fallback for low-confidence detections
            llm_confidence_threshold: Confidence threshold for using LLM (default: 0.6)
        """
        self.use_llm_fallback = use_llm_fallback
        self.llm_confidence_threshold = llm_confidence_threshold
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

        # Add temporal_query to secondary if temporal constraint was extracted
        if "temporal_constraint" in metadata and "temporal_query" not in secondary and primary_intent != "temporal_query":
            secondary.append("temporal_query")

        # Add sender_query to secondary if sender was extracted
        if "sender" in metadata and "sender_query" not in secondary and primary_intent != "sender_query":
            secondary.append("sender_query")

        result = {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "metadata": metadata,
            "secondary_signals": secondary,
            "detection_method": "pattern"
        }

        # LLM fallback for low-confidence detections
        if self.use_llm_fallback and confidence < self.llm_confidence_threshold:
            logger.info(
                f"Low confidence ({confidence:.2f}), using LLM fallback",
                extra={"query": query, "pattern_confidence": confidence}
            )

            try:
                llm_result = self._llm_based_detection(query)

                # Use LLM result if it has higher confidence
                if llm_result["confidence"] > confidence:
                    logger.info(
                        f"LLM result used: {llm_result['primary_intent']} "
                        f"(confidence: {llm_result['confidence']:.2f})",
                        extra={"llm_result": llm_result}
                    )
                    result = llm_result
                    result["detection_method"] = "llm"
                    result["pattern_confidence"] = confidence  # Keep original for comparison
                else:
                    result["detection_method"] = "pattern_with_llm_check"

            except Exception as e:
                logger.warning(
                    f"LLM fallback failed: {e}",
                    extra={"error": str(e)}
                )
                result["detection_method"] = "pattern_llm_failed"

        logger.debug(
            f"Intent detected: {result['primary_intent']} (confidence: {result['confidence']:.2f}, "
            f"method: {result['detection_method']})",
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

        # Extract time range and temporal constraints
        time_patterns = {
            "yesterday": r"\byesterday\b",
            "today": r"\btoday\b",
            "last_week": r"\blast week\b",
            "last_month": r"\blast month\b",
            "this_week": r"\bthis week\b",
            "this_month": r"\bthis month\b",
            "recent": r"\b(?:recent|latest|newest)\b",
        }

        # Extract relative time expressions (e.g., "past 4 weeks", "last 3 months", "past four weeks")
        # Map word numbers to digits
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "twelve": 12, "twenty": 20, "thirty": 30
        }

        relative_time_pattern = r"(?:past|last)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|thirty)\s+(day|days|week|weeks|month|months|year|years)"
        relative_match = re.search(relative_time_pattern, query, re.I)

        if relative_match:
            value_str = relative_match.group(1).lower()
            value = word_to_num.get(value_str, int(value_str)) if value_str.isdigit() else word_to_num.get(value_str, 1)
            unit = relative_match.group(2).lower().rstrip('s')  # Normalize to singular

            # Convert to days for consistency
            days_map = {
                "day": 1,
                "week": 7,
                "month": 30,  # Approximation
                "year": 365
            }
            days_back = value * days_map.get(unit, 1)

            metadata["time_range"] = f"past_{value}_{unit}s"
            metadata["temporal_constraint"] = {
                "type": "relative",
                "value": value,
                "unit": unit,
                "days_back": days_back
            }
        else:
            # Try simple keyword patterns
            for time_range, pattern in time_patterns.items():
                if re.search(pattern, query, re.I):
                    metadata["time_range"] = time_range

                    # Add structured temporal constraint for known patterns
                    days_map = {
                        "yesterday": 1,
                        "today": 0,
                        "last_week": 7,
                        "last_month": 30,
                        "this_week": 7,
                        "this_month": 30,
                        "recent": 7  # Default to 1 week for "recent"
                    }

                    if time_range in days_map:
                        metadata["temporal_constraint"] = {
                            "type": "keyword",
                            "keyword": time_range,
                            "days_back": days_map[time_range]
                        }
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

    def _llm_based_detection(self, query: str) -> Dict:
        """
        Use LLM to classify intent for ambiguous queries.

        Args:
            query: User query string

        Returns:
            {
                "primary_intent": "sender_query",
                "confidence": 0.85,
                "metadata": {...},
                "secondary_signals": []
            }
        """
        # Create classification prompt
        prompt = f"""Classify this email query into one of these intents:

- thread_summary: User wants to summarize an email thread or discussion
- sender_query: User wants emails from a specific person
- temporal_query: User wants recent/time-based emails
- action_items: User wants tasks or deadlines
- decision_tracking: User wants decisions made
- aggregation_query: User wants analysis (most/least/top/compare/count)
- factual_lookup: User wants specific information

Also extract metadata:
- sender: name if mentioned (capitalize first letter, e.g., "Alice")
- time_range: if temporal words used (use: yesterday, today, last_week, last_month, this_week, this_month, or recent)
- topic_keywords: important topic words (lowercase, no common words)

Query: "{query}"

Return ONLY valid JSON (no markdown, no backticks):
{{
  "intent": "...",
  "confidence": 0.0-1.0,
  "metadata": {{"sender": "...", "time_range": "...", "topic_keywords": [...]}}
}}"""

        try:
            # Initialize LLM completer
            completer = OpenAICompleter(model_name="gpt-3.5-turbo")

            # Get completion
            response = completer.get_completion(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=200
            )

            logger.debug(f"LLM response: {response}", extra={"llm_response": response})

            # Handle error responses
            if response.startswith("[ERROR]"):
                raise ValueError(f"LLM returned error: {response}")

            # Parse JSON response
            # Remove markdown code blocks if present
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # Extract JSON from markdown code block
                lines = response_clean.split("\n")
                json_lines = [line for line in lines if line and not line.startswith("```")]
                response_clean = "\n".join(json_lines)

            result_data = json.loads(response_clean)

            # Validate required fields
            if "intent" not in result_data or "confidence" not in result_data:
                raise ValueError(f"LLM response missing required fields: {result_data}")

            # Convert to standard format
            metadata = result_data.get("metadata", {})

            # Clean metadata - remove empty values
            metadata = {k: v for k, v in metadata.items() if v}

            result = {
                "primary_intent": result_data["intent"],
                "confidence": float(result_data["confidence"]),
                "metadata": metadata,
                "secondary_signals": []  # LLM doesn't detect secondary signals
            }

            logger.info(
                f"LLM classified query as {result['primary_intent']} "
                f"(confidence: {result['confidence']:.2f})",
                extra={"llm_result": result}
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM JSON response: {e}",
                extra={"response": response}
            )
            raise ValueError(f"Invalid JSON from LLM: {e}")

        except Exception as e:
            logger.error(
                f"LLM-based detection failed: {e}",
                extra={"error": str(e)}
            )
            raise


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
