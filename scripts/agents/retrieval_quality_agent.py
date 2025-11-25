"""
Retrieval Quality Assessment Agent

This agent evaluates the quality of retrieved chunks before generation.
It uses both heuristic checks and LLM-based assessment to determine if
the retrieved context is sufficient to answer the query.

Quality Levels:
- HIGH: Can confidently answer the query
- MEDIUM: Some relevant info, but may be incomplete
- LOW: Insufficient or irrelevant information

Usage:
    agent = RetrievalQualityAgent()
    assessment = agent.assess(query, chunks)

    if assessment["quality"] == "HIGH":
        # Proceed with generation
    elif assessment["quality"] == "MEDIUM":
        # Consider reranking or clarification
    else:
        # Ask user for clarification
"""

from typing import List, Dict, Optional
import logging
from scripts.chunking.models import Chunk
from scripts.api_clients.openai.completer import OpenAICompleter


logger = logging.getLogger(__name__)


class RetrievalQualityAgent:
    """
    Assesses the quality of retrieved chunks for answering a query.

    Uses a two-stage approach:
    1. Fast heuristic checks (rerank scores, chunk count)
    2. LLM-based semantic assessment (can chunks answer query?)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_llm_assessment: bool = True,
        run_id: Optional[str] = None
    ):
        """
        Initialize quality assessment agent.

        Args:
            model: LLM model for assessment (default: gpt-4o-mini)
            use_llm_assessment: Whether to use LLM assessment (default: True)
            run_id: Optional run ID for logging
        """
        self.model = model
        self.use_llm_assessment = use_llm_assessment
        self.run_id = run_id

        # Initialize LLM client if needed
        if use_llm_assessment:
            self.llm = OpenAICompleter()

        logger.info(
            f"RetrievalQualityAgent initialized: model={model}, "
            f"llm_assessment={use_llm_assessment}"
        )

    def assess(
        self,
        query: str,
        chunks: List[Chunk],
        intent: Optional[Dict] = None
    ) -> Dict:
        """
        Assess retrieval quality for a query.

        Args:
            query: User query
            chunks: Retrieved chunks
            intent: Optional intent metadata

        Returns:
            {
                "quality": "HIGH" | "MEDIUM" | "LOW",
                "confidence": 0.0-1.0,
                "reasoning": "...",
                "heuristics": {...},
                "llm_assessment": {...} (if enabled),
                "recommendation": "proceed" | "rerank" | "clarify"
            }
        """
        logger.info(f"Assessing retrieval quality for query: '{query}'")

        # Step 1: Heuristic checks (fast)
        heuristics = self._heuristic_checks(chunks)
        logger.debug(f"Heuristic checks: {heuristics}")

        # Step 2: LLM assessment (if enabled)
        llm_assessment = None
        if self.use_llm_assessment and chunks:
            llm_assessment = self._llm_assessment(query, chunks)
            logger.debug(f"LLM assessment: {llm_assessment}")

        # Step 3: Combine assessments
        final_assessment = self._combine_assessments(
            heuristics,
            llm_assessment,
            intent
        )

        logger.info(
            f"Final assessment: quality={final_assessment['quality']}, "
            f"confidence={final_assessment['confidence']:.2f}, "
            f"recommendation={final_assessment['recommendation']}"
        )

        return final_assessment

    def _heuristic_checks(self, chunks: List[Chunk]) -> Dict:
        """
        Fast heuristic checks on retrieved chunks.

        Checks:
        - Chunk count (do we have enough?)
        - Rerank scores (are they relevant?)
        - Score distribution (diverse or focused?)

        Returns:
            Dictionary with heuristic metrics
        """
        if not chunks:
            return {
                "has_chunks": False,
                "chunk_count": 0,
                "avg_rerank_score": 0.0,
                "max_rerank_score": 0.0,
                "min_rerank_score": 0.0,
                "score_variance": 0.0,
                "heuristic_quality": "LOW",
                "issues": ["No chunks retrieved"]
            }

        # Extract rerank scores
        rerank_scores = []
        for chunk in chunks:
            score = chunk.meta.get("rerank_score")
            if score is not None:
                rerank_scores.append(score)

        # Calculate metrics
        chunk_count = len(chunks)
        has_rerank_scores = len(rerank_scores) > 0

        if has_rerank_scores:
            avg_score = sum(rerank_scores) / len(rerank_scores)
            max_score = max(rerank_scores)
            min_score = min(rerank_scores)

            # Variance (spread of scores)
            mean = avg_score
            variance = sum((s - mean) ** 2 for s in rerank_scores) / len(rerank_scores)
        else:
            avg_score = 0.5  # Neutral if no rerank scores
            max_score = 0.5
            min_score = 0.5
            variance = 0.0

        # Determine heuristic quality
        issues = []

        if chunk_count == 0:
            heuristic_quality = "LOW"
            issues.append("No chunks retrieved")
        elif chunk_count < 5:
            heuristic_quality = "MEDIUM"
            issues.append(f"Low chunk count ({chunk_count})")
        elif not has_rerank_scores:
            heuristic_quality = "MEDIUM"
            issues.append("No rerank scores available")
        elif max_score < 0.6:
            heuristic_quality = "LOW"
            issues.append(f"Low max rerank score ({max_score:.2f})")
        elif avg_score < 0.5:
            heuristic_quality = "LOW"
            issues.append(f"Low average rerank score ({avg_score:.2f})")
        elif avg_score > 0.7 and max_score > 0.8:
            heuristic_quality = "HIGH"
        elif avg_score > 0.5:
            heuristic_quality = "MEDIUM"
        else:
            heuristic_quality = "LOW"

        return {
            "has_chunks": chunk_count > 0,
            "chunk_count": chunk_count,
            "has_rerank_scores": has_rerank_scores,
            "avg_rerank_score": avg_score,
            "max_rerank_score": max_score,
            "min_rerank_score": min_score,
            "score_variance": variance,
            "heuristic_quality": heuristic_quality,
            "issues": issues
        }

    def _llm_assessment(self, query: str, chunks: List[Chunk]) -> Dict:
        """
        Use LLM to assess if chunks can answer the query.

        Args:
            query: User query
            chunks: Retrieved chunks

        Returns:
            {
                "can_answer": True/False,
                "confidence": 0.0-1.0,
                "reasoning": "...",
                "missing_info": "..." (optional)
            }
        """
        # Build assessment prompt
        prompt = self._build_assessment_prompt(query, chunks)

        try:
            # Call LLM
            response = self.llm.complete(
                prompt=prompt,
                model=self.model,
                temperature=0.0,  # Deterministic assessment
                max_tokens=300
            )

            # Parse response
            assessment = self._parse_llm_response(response)

            return assessment

        except Exception as e:
            logger.error(f"LLM assessment failed: {e}")
            return {
                "can_answer": True,  # Optimistic fallback
                "confidence": 0.5,
                "reasoning": "LLM assessment unavailable",
                "error": str(e)
            }

    def _build_assessment_prompt(self, query: str, chunks: List[Chunk]) -> str:
        """Build prompt for LLM assessment."""

        # Include top 5 chunks for assessment (keep prompt short)
        chunks_text = ""
        for i, chunk in enumerate(chunks[:5], 1):
            subject = chunk.meta.get("subject", "N/A")
            sender = chunk.meta.get("sender_name", "N/A")
            date = chunk.meta.get("date", "N/A")

            # Truncate long chunks
            text_preview = chunk.text[:400] + "..." if len(chunk.text) > 400 else chunk.text

            chunks_text += f"\n[Chunk {i}]\n"
            chunks_text += f"Subject: {subject}\n"
            chunks_text += f"From: {sender} ({date})\n"
            chunks_text += f"Content: {text_preview}\n"

        prompt = f"""You are a retrieval quality assessor. Your task is to determine if the retrieved email chunks contain sufficient information to answer the user's query.

Query: "{query}"

Retrieved Chunks (top 5):
{chunks_text}

Additional Context: {len(chunks)} total chunks retrieved.

Assessment Task:
1. Can these chunks answer the query?
2. How confident are you? (0-100%)
3. What information is present or missing?

Response Format:
CAN_ANSWER: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [One sentence explaining your assessment]
MISSING: [What key information is missing, if any]

Provide your assessment:"""

        return prompt

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM assessment response."""
        import re

        # Extract fields using regex
        can_answer_match = re.search(r'CAN_ANSWER:\s*(YES|NO)', response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\w+:|$)', response, re.DOTALL)
        missing_match = re.search(r'MISSING:\s*(.+?)(?=\n\w+:|$)', response, re.DOTALL)

        # Parse values
        can_answer = can_answer_match.group(1).upper() == "YES" if can_answer_match else True

        confidence_pct = int(confidence_match.group(1)) if confidence_match else 50
        confidence = confidence_pct / 100.0  # Normalize to 0-1

        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        missing = missing_match.group(1).strip() if missing_match else None

        return {
            "can_answer": can_answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "missing_info": missing
        }

    def _combine_assessments(
        self,
        heuristics: Dict,
        llm_assessment: Optional[Dict],
        intent: Optional[Dict]
    ) -> Dict:
        """
        Combine heuristic and LLM assessments into final quality rating.

        Strategy:
        - If heuristics are LOW and LLM agrees → LOW
        - If heuristics are HIGH and LLM agrees → HIGH
        - If they disagree → MEDIUM (cautious)
        - If no LLM assessment → rely on heuristics

        Args:
            heuristics: Heuristic check results
            llm_assessment: LLM assessment results (optional)
            intent: Query intent (optional)

        Returns:
            Final assessment with quality, confidence, recommendation
        """
        # Start with heuristic quality
        heuristic_quality = heuristics["heuristic_quality"]

        # Combine with LLM if available
        if llm_assessment:
            llm_can_answer = llm_assessment["can_answer"]
            llm_confidence = llm_assessment["confidence"]

            # Agreement cases
            if heuristic_quality == "HIGH" and llm_can_answer and llm_confidence > 0.7:
                final_quality = "HIGH"
                final_confidence = min(0.9, (heuristics["avg_rerank_score"] + llm_confidence) / 2)
                reasoning = f"Strong retrieval quality. {llm_assessment['reasoning']}"
                recommendation = "proceed"

            elif heuristic_quality == "LOW" and not llm_can_answer:
                final_quality = "LOW"
                final_confidence = 1.0 - llm_confidence
                reasoning = f"Insufficient information. {llm_assessment['reasoning']}"
                recommendation = "clarify"

            # Disagreement or MEDIUM cases
            else:
                final_quality = "MEDIUM"
                final_confidence = llm_confidence
                reasoning = llm_assessment["reasoning"]

                # Recommendation based on confidence
                if llm_confidence > 0.6:
                    recommendation = "proceed"
                else:
                    recommendation = "rerank"

        else:
            # No LLM assessment - rely on heuristics
            if heuristic_quality == "HIGH":
                final_quality = "HIGH"
                final_confidence = heuristics["avg_rerank_score"]
                reasoning = f"High rerank scores (avg: {heuristics['avg_rerank_score']:.2f})"
                recommendation = "proceed"

            elif heuristic_quality == "LOW":
                final_quality = "LOW"
                final_confidence = 0.3
                reasoning = ", ".join(heuristics["issues"])
                recommendation = "clarify"

            else:
                final_quality = "MEDIUM"
                final_confidence = 0.6
                reasoning = f"Moderate rerank scores (avg: {heuristics['avg_rerank_score']:.2f})"
                recommendation = "proceed"  # Cautiously proceed

        # Build final assessment
        assessment = {
            "quality": final_quality,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "heuristics": heuristics,
        }

        if llm_assessment:
            assessment["llm_assessment"] = llm_assessment

        return assessment
