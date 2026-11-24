"""
Query Analysis Agent - Analyzes query clarity, entities, and quality.

Part of Phase 6: Query Intelligence
Determines if queries are specific or vague, extracts entities,
and suggests reformulations for improvement.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional
from scripts.api_clients.openai.completer import OpenAICompleter
from scripts.utils.logger import LoggerManager


@dataclass
class QueryAnalysis:
    """Analysis of a query's clarity and quality."""

    # Core assessments
    is_specific: bool
    """Whether the query is specific (True) or vague/broad (False)"""

    intent_quality: float
    """How clear is the intent? 0.0 = very unclear, 1.0 = crystal clear"""

    entities: List[str]
    """Domain entities mentioned (e.g., ['Primo VE', 'NDE', 'CSL'])"""

    # Issues and improvements
    issues: List[str] = field(default_factory=list)
    """Problems with the query (e.g., ['too_vague', 'multiple_intents'])"""

    clarification_questions: List[str] = field(default_factory=list)
    """Questions to ask user for clarification (if vague)"""

    suggested_reformulations: List[str] = field(default_factory=list)
    """Alternative, more specific phrasings"""

    # Additional metadata
    reasoning: str = ""
    """Explanation of the analysis"""

    query_type: str = "general"
    """Type of query: 'factual', 'temporal', 'comparative', etc."""


class QueryAnalyzer:
    """
    Analyzes queries to determine clarity, extract entities, and suggest improvements.

    Uses LLM to assess:
    - Specificity (specific vs vague/broad)
    - Entity extraction (domain-specific terms)
    - Intent quality (how clear is what user wants)
    - Issues (vagueness, ambiguity, multiple intents)
    - Clarifications (questions to ask if vague)
    - Reformulations (rewrite as specific sub-queries)

    Example:
        analyzer = QueryAnalyzer()

        # Specific query
        analysis = analyzer.analyze("How do I configure facets in Primo VE?")
        # → is_specific=True, intent_quality=0.9, entities=['Primo VE', 'facets']

        # Vague query
        analysis = analyzer.analyze("What happened last month?")
        # → is_specific=False, intent_quality=0.3, issues=['too_vague', 'missing_context']
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        run_id: Optional[str] = None
    ):
        """
        Initialize Query Analyzer.

        Args:
            model: LLM model to use (default: gpt-4o-mini for speed/cost)
            run_id: Optional run ID for logging
        """
        self.model = model
        self.llm = OpenAICompleter(model_name=model)
        self.logger = LoggerManager.get_logger("query_analyzer", run_id=run_id)

        self.logger.info(f"QueryAnalyzer initialized with model={model}")

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query for clarity, entities, and quality.

        Args:
            query: The user's query string

        Returns:
            QueryAnalysis with assessment results
        """
        self.logger.info(f"Analyzing query: '{query}'")

        # Build analysis prompt
        prompt = self._build_analysis_prompt(query)

        # Get LLM analysis
        try:
            response = self.llm.get_completion(prompt, temperature=0.3, max_tokens=500)
            analysis_dict = self._parse_llm_response(response)

            # Build QueryAnalysis object
            analysis = QueryAnalysis(
                is_specific=analysis_dict.get("is_specific", True),
                intent_quality=float(analysis_dict.get("intent_quality", 0.5)),
                entities=analysis_dict.get("entities", []),
                issues=analysis_dict.get("issues", []),
                clarification_questions=analysis_dict.get("clarification_questions", []),
                suggested_reformulations=analysis_dict.get("suggested_reformulations", []),
                reasoning=analysis_dict.get("reasoning", ""),
                query_type=analysis_dict.get("query_type", "general")
            )

            self.logger.info(
                f"Query analysis complete: is_specific={analysis.is_specific}, "
                f"intent_quality={analysis.intent_quality:.2f}, "
                f"entities={len(analysis.entities)}, issues={len(analysis.issues)}"
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            # Return conservative default analysis
            return QueryAnalysis(
                is_specific=True,  # Assume specific to avoid unnecessary reformulation
                intent_quality=0.7,
                entities=[],
                issues=["analysis_failed"],
                reasoning=f"Analysis failed: {str(e)}"
            )

    def _build_analysis_prompt(self, query: str) -> str:
        """Build prompt for LLM query analysis."""
        return f"""Analyze this query for an email search system (Primo library system domain):

Query: "{query}"

Assess the following:

1. **Specificity**: Is this query specific and targeted, or vague and broad?
   - Specific: "How do I configure facets in Primo VE?"
   - Vague: "What happened last month?" or "Tell me about Primo"

2. **Entities**: Extract domain-specific entities mentioned:
   - Products: Primo, Primo VE, NDE, Alma, etc.
   - Features: facets, citation styles, CSL, search, etc.
   - Technologies: API, OAuth, SAML, etc.
   - Other relevant terms

3. **Intent Quality**: How clear is what the user wants? (0.0-1.0 scale)
   - 1.0 = Very clear ("How do I configure X?")
   - 0.5 = Somewhat clear ("What are the issues with X?")
   - 0.0 = Very unclear ("Tell me about X")

4. **Query Type**: What kind of question is this?
   - factual: Asking for specific information
   - temporal: Time-based (recent, last month, etc.)
   - comparative: Comparing options
   - troubleshooting: Reporting/diagnosing problems
   - exploratory: Broad exploration

5. **Issues**: Any problems with the query?
   - too_vague: Lacks specificity
   - too_broad: Covers too much
   - multiple_intents: Asking several things at once
   - missing_context: Needs temporal/spatial context
   - ambiguous: Unclear meaning

6. **Clarifications**: If vague, what questions would help clarify?
   - "Which specific aspect of X are you interested in?"
   - "What time period are you asking about?"

7. **Reformulations**: Rewrite as 2-4 more specific sub-queries
   - Break vague query into concrete questions
   - Make implicit context explicit

Return ONLY valid JSON (no markdown):
{{
  "is_specific": true/false,
  "intent_quality": 0.0-1.0,
  "entities": ["entity1", "entity2"],
  "query_type": "factual",
  "issues": ["issue1", "issue2"],
  "clarification_questions": ["question1?"],
  "suggested_reformulations": ["reformulated query 1", "reformulated query 2"],
  "reasoning": "Brief explanation of analysis"
}}"""

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response into dict."""
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            analysis_dict = json.loads(cleaned)

            return analysis_dict

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.error(f"Response was: {response[:200]}...")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
