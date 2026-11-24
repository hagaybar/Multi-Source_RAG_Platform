"""
Query Reformulation Agent - Reformulates vague queries into specific sub-queries.

Part of Phase 6: Query Intelligence
Takes vague/broad queries and breaks them into 2-4 specific sub-queries
for multi-query retrieval.
"""

import json
from typing import List, Optional
from scripts.agents.query_analyzer import QueryAnalyzer, QueryAnalysis
from scripts.api_clients.openai.completer import OpenAICompleter
from scripts.utils.logger import LoggerManager


class QueryReformulator:
    """
    Reformulates vague queries into specific sub-queries for better retrieval.

    Workflow:
    1. Analyze query using QueryAnalyzer
    2. If vague (is_specific=False), reformulate into 2-4 sub-queries
    3. If specific, return original query
    4. Use LLM to generate contextual sub-queries

    Example:
        reformulator = QueryReformulator()

        # Vague query
        sub_queries = reformulator.reformulate(
            "What happened with Primo VE last month?"
        )
        # Returns:
        # [
        #   "What new features were released in Primo VE in November 2025?",
        #   "What bugs or issues were reported with Primo VE in November 2025?",
        #   "What configuration changes were made in Primo VE in November 2025?"
        # ]

        # Specific query
        sub_queries = reformulator.reformulate(
            "How do I configure facets in Primo VE?"
        )
        # Returns: ["How do I configure facets in Primo VE?"]  # Original query
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        analyzer: Optional[QueryAnalyzer] = None,
        run_id: Optional[str] = None,
        vagueness_threshold: float = 0.7
    ):
        """
        Initialize Query Reformulator.

        Args:
            model: LLM model to use (default: gpt-4o-mini for speed/cost)
            analyzer: Optional QueryAnalyzer instance (creates new if None)
            run_id: Optional run ID for logging
            vagueness_threshold: Intent quality threshold below which to reformulate
                                 (default: 0.7 - reformulate if quality < 0.7)
        """
        self.model = model
        self.llm = OpenAICompleter(model_name=model)
        self.analyzer = analyzer or QueryAnalyzer(model=model, run_id=run_id)
        self.logger = LoggerManager.get_logger("query_reformulator", run_id=run_id)
        self.vagueness_threshold = vagueness_threshold

        self.logger.info(
            f"QueryReformulator initialized with model={model}, "
            f"vagueness_threshold={vagueness_threshold}"
        )

    def reformulate(
        self,
        query: str,
        context: Optional[dict] = None,
        force_reformulate: bool = False
    ) -> List[str]:
        """
        Reformulate query into specific sub-queries if vague.

        Args:
            query: The user's query string
            context: Optional context dict (e.g., {"current_date": "2025-11-25"})
            force_reformulate: If True, reformulate even if query is specific

        Returns:
            List of sub-queries. If query is specific, returns [original_query].
            If vague, returns 2-4 reformulated sub-queries.
        """
        self.logger.info(f"Reformulating query: '{query}'")

        # Step 1: Analyze query clarity
        analysis = self.analyzer.analyze(query)

        # Step 2: Decide if reformulation is needed
        needs_reformulation = (
            force_reformulate or
            not analysis.is_specific or
            analysis.intent_quality < self.vagueness_threshold
        )

        if not needs_reformulation:
            self.logger.info(
                f"Query is specific (quality={analysis.intent_quality:.2f}), "
                f"no reformulation needed"
            )
            return [query]

        # Step 3: Query is vague - reformulate
        self.logger.info(
            f"Query is vague (is_specific={analysis.is_specific}, "
            f"quality={analysis.intent_quality:.2f}), reformulating..."
        )

        # Use LLM-suggested reformulations if available
        if analysis.suggested_reformulations and len(analysis.suggested_reformulations) >= 2:
            sub_queries = analysis.suggested_reformulations[:4]  # Max 4
            self.logger.info(
                f"Using {len(sub_queries)} reformulations from QueryAnalyzer"
            )
            return sub_queries

        # Otherwise, generate reformulations with LLM
        sub_queries = self._generate_reformulations(query, analysis, context)

        self.logger.info(f"Generated {len(sub_queries)} sub-queries")
        return sub_queries

    def _generate_reformulations(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[dict] = None
    ) -> List[str]:
        """
        Generate reformulations using LLM.

        Args:
            query: Original query
            analysis: QueryAnalysis from analyzer
            context: Optional context dict

        Returns:
            List of 2-4 specific sub-queries
        """
        # Build context string
        context_str = ""
        if context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])

        # Build reformulation prompt
        prompt = self._build_reformulation_prompt(query, analysis, context_str)

        # Get LLM reformulations
        try:
            response = self.llm.get_completion(prompt, temperature=0.4, max_tokens=400)
            sub_queries = self._parse_reformulations(response)

            return sub_queries

        except Exception as e:
            self.logger.error(f"Reformulation failed: {e}", exc_info=True)
            # Fallback: return original query
            return [query]

    def _build_reformulation_prompt(
        self,
        query: str,
        analysis: QueryAnalysis,
        context_str: str
    ) -> str:
        """Build prompt for LLM reformulation."""
        return f"""Break this vague query into 2-4 specific, actionable sub-queries.

Original Query: "{query}"

Query Analysis:
- Type: {analysis.query_type}
- Intent Quality: {analysis.intent_quality:.2f}
- Entities: {', '.join(analysis.entities) if analysis.entities else 'None'}
- Issues: {', '.join(analysis.issues) if analysis.issues else 'None'}

{f"Context:\n{context_str}\n" if context_str else ""}
Guidelines for reformulation:
1. Each sub-query should be specific and focused
2. Cover different aspects of the original intent
3. Make temporal references explicit (use actual dates if "last month", etc.)
4. Include domain entities (Primo VE, NDE, CSL, etc.)
5. Focus on actionable questions (how, what, when, who)

Examples:

Original: "What happened with Primo VE last month?"
Sub-queries:
1. What new features were released in Primo VE in November 2025?
2. What bugs or issues were reported with Primo VE in November 2025?
3. What configuration changes were made in Primo VE in November 2025?

Original: "Tell me about NDE"
Sub-queries:
1. What is the NDE interface in Primo VE?
2. What are the key features of NDE?
3. What are common issues with NDE configuration?

Return ONLY a JSON array of sub-queries (no markdown):
["sub-query 1", "sub-query 2", "sub-query 3"]"""

    def _parse_reformulations(self, response: str) -> List[str]:
        """
        Parse LLM response into list of sub-queries.

        Args:
            response: LLM response (should be JSON array)

        Returns:
            List of sub-queries
        """
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

            # Parse JSON array
            sub_queries = json.loads(cleaned)

            if not isinstance(sub_queries, list):
                raise ValueError("Response is not a JSON array")

            # Validate sub-queries
            sub_queries = [sq.strip() for sq in sub_queries if isinstance(sq, str) and sq.strip()]

            # Ensure 2-4 sub-queries
            if len(sub_queries) < 2:
                self.logger.warning(f"Only {len(sub_queries)} sub-queries generated, expected 2+")
            if len(sub_queries) > 4:
                self.logger.warning(f"Truncating {len(sub_queries)} sub-queries to 4")
                sub_queries = sub_queries[:4]

            return sub_queries

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse reformulations as JSON: {e}")
            self.logger.error(f"Response was: {response[:200]}...")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
