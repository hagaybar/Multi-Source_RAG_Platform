"""
EmailOrchestratorAgent - Main orchestrator for email retrieval and context assembly.

This orchestrator coordinates the complete email RAG pipeline:
1. Intent detection (EmailIntentDetector)
2. Strategy selection (EmailStrategySelector)
3. Retrieval execution (specialized retrievers)
4. Context assembly (ContextAssembler)
5. Metadata extraction and logging

Example:
    orchestrator = EmailOrchestratorAgent(project)
    result = orchestrator.retrieve("What did Alice say about budget last week?")
    # Returns: {
    #     "chunks": [...],
    #     "context": "...",
    #     "intent": {...},
    #     "strategy": {...},
    #     "metadata": {...}
    # }
"""

from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import yaml

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.agents.email_intent_detector import EmailIntentDetector
from scripts.agents.email_strategy_selector import EmailStrategySelector
from scripts.retrieval.context_assembler import ContextAssembler
from scripts.retrieval.email_thread_retriever import ThreadRetriever
from scripts.retrieval.email_temporal_retriever import TemporalRetriever
from scripts.retrieval.email_sender_retriever import SenderRetriever
from scripts.retrieval.email_multi_aspect_retriever import MultiAspectRetriever
from scripts.retrieval.reranker import create_reranker
from scripts.agents.retrieval_quality_agent import RetrievalQualityAgent
from scripts.utils.logger import LoggerManager


class EmailOrchestratorAgent:
    """
    Main orchestrator for email retrieval and context assembly.

    Coordinates:
    - Intent detection
    - Strategy selection
    - Retrieval execution (including combined strategies)
    - Context assembly and cleaning
    - Logging and debugging
    """

    def __init__(self, project: ProjectManager, run_id: Optional[str] = None):
        """
        Initialize EmailOrchestratorAgent.

        Args:
            project: ProjectManager instance
            run_id: Optional run ID for logging
        """
        self.project = project
        self.run_id = run_id

        # Initialize logger first (needed by other components)
        self.logger = LoggerManager.get_logger(
            "email_orchestrator",
            task_paths=project.get_task_paths(),
            run_id=run_id
        )

        # Load reranking configuration
        self.reranking_config = self._load_reranking_config()

        # Initialize components
        self.intent_detector = EmailIntentDetector()
        self.strategy_selector = EmailStrategySelector()
        self.context_assembler = ContextAssembler()

        # Initialize retrievers
        self.retrievers = {
            "thread_retrieval": ThreadRetriever(project, run_id=run_id),
            "temporal_retrieval": TemporalRetriever(project, run_id=run_id),
            "sender_retrieval": SenderRetriever(project, run_id=run_id),
            "multi_aspect": MultiAspectRetriever(project, run_id=run_id),
        }

        # Initialize reranker if enabled
        self.reranker = None
        if self.reranking_config.get("enabled", False):
            try:
                reranker_type = self.reranking_config.get("type", "cross-encoder")

                # Prepare kwargs based on reranker type
                if reranker_type == "cross-encoder":
                    kwargs = {
                        "model_name": self.reranking_config.get("cross_encoder", {}).get(
                            "model_name",
                            "cross-encoder/ms-marco-MiniLM-L-12-v2"
                        )
                    }
                elif reranker_type == "llm":
                    kwargs = {
                        "model": self.reranking_config.get("llm", {}).get("model", "gpt-4o-mini"),
                        "batch_size": self.reranking_config.get("llm", {}).get("batch_size", 20)
                    }
                elif reranker_type == "cohere":
                    kwargs = {
                        "api_key": self.reranking_config.get("cohere", {}).get("api_key"),
                        "model": self.reranking_config.get("cohere", {}).get("model", "rerank-english-v2.0")
                    }
                else:
                    raise ValueError(f"Unknown reranker type: {reranker_type}")

                # Create reranker using factory
                self.reranker = create_reranker(reranker_type, **kwargs)
                self.logger.info(f"Reranker initialized: type={reranker_type}")

            except Exception as e:
                self.logger.warning(f"Failed to initialize reranker: {e}. Continuing without reranking.")
                self.reranker = None

        # Initialize quality assessment agent
        self.quality_agent = RetrievalQualityAgent(
            model="gpt-4o-mini",
            use_llm_assessment=True,
            run_id=run_id
        )

    def _load_reranking_config(self) -> Dict:
        """
        Load reranking configuration from YAML file.

        Returns:
            Dictionary with reranking config, or {"enabled": False} if file not found
        """
        config_path = Path(__file__).parent.parent.parent / "configs" / "reranking.yaml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("reranking", {"enabled": False})
            except Exception as e:
                self.logger.warning(f"Failed to load reranking config: {e}")
                return {"enabled": False}

        return {"enabled": False}

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        max_tokens: int = 4000
    ) -> Dict:
        """
        Main retrieval orchestration with context assembly.

        Args:
            query: User query
            top_k: Number of chunks to retrieve (None = auto-adjust based on intent)
            max_tokens: Maximum tokens for assembled context (default: 4000)

        Returns:
            {
                "chunks": [...],  # Cleaned, organized chunks
                "context": "...",  # Assembled context string
                "intent": {...},  # Detected intent
                "strategy": {...},  # Strategy used
                "metadata": {...}  # Additional info (chunk_count, senders, etc.)
            }
        """
        # Step 1: Detect intent
        intent = self.intent_detector.detect(query)
        self._log_intent(intent)

        # Step 2: Select strategy
        strategy = self.strategy_selector.select_strategy(intent)
        self._log_strategy(strategy)

        # Step 3: Adjust top_k based on intent if not explicitly provided
        if top_k is None:
            top_k = self._get_optimal_top_k(intent["primary_intent"])
            self.logger.info(
                f"Auto-adjusted top_k={top_k} for intent={intent['primary_intent']}",
                extra={"run_id": self.run_id, "intent": intent["primary_intent"], "top_k": top_k} if self.run_id else {}
            )

        self.logger.info(
            f"EmailOrchestrator: Starting retrieval for query='{query}'",
            extra={"run_id": self.run_id, "query": query, "top_k": top_k} if self.run_id else {}
        )

        # Step 3: Execute retrieval
        chunks = self._execute_retrieval(query, strategy, intent, top_k)
        self.logger.info(
            f"Retrieved {len(chunks)} chunks",
            extra={"run_id": self.run_id, "chunk_count": len(chunks)} if self.run_id else {}
        )

        # Step 4: Assess retrieval quality
        quality_assessment = self.quality_agent.assess(query, chunks, intent)
        self.logger.info(
            f"Quality assessment: {quality_assessment['quality']} "
            f"(confidence: {quality_assessment['confidence']:.2f})",
            extra={
                "run_id": self.run_id,
                "quality": quality_assessment["quality"],
                "confidence": quality_assessment["confidence"],
                "recommendation": quality_assessment["recommendation"]
            } if self.run_id else {}
        )

        # Step 5: Assemble clean context
        context = self.context_assembler.assemble(chunks, intent, max_tokens=max_tokens)
        self.logger.info(
            f"Assembled context: {len(context)} characters",
            extra={"run_id": self.run_id, "context_length": len(context)} if self.run_id else {}
        )

        # Step 6: Build metadata for transparency
        metadata = self._build_metadata(chunks, strategy)

        return {
            "chunks": chunks,
            "context": context,
            "intent": intent,
            "strategy": strategy,
            "metadata": metadata,
            "quality_assessment": quality_assessment
        }

    def _get_optimal_top_k(self, intent: str) -> int:
        """
        Determine optimal top_k based on intent type.

        Different intents require different amounts of context:
        - Thread summaries need more chunks (full thread)
        - Sender/temporal queries need moderate chunks
        - Factual lookups need fewer chunks (specific facts)

        Args:
            intent: Primary intent type

        Returns:
            Optimal top_k value (5-20)
        """
        # Intent-based top_k mapping
        top_k_map = {
            "thread_summary": 20,      # Need full thread context
            "aggregation_query": 20,   # Need broad coverage
            "temporal_query": 15,      # Time range might be broad
            "sender_query": 12,        # Sender might have multiple emails
            "decision_tracking": 12,   # Decisions scattered across emails
            "action_items": 12,        # Action items in multiple emails
            "factual_lookup": 10,      # Specific fact needs moderate context
        }

        optimal = top_k_map.get(intent, 10)  # Default: 10
        self.logger.debug(
            f"Optimal top_k for {intent}: {optimal}",
            extra={"run_id": self.run_id, "intent": intent, "top_k": optimal} if self.run_id else {}
        )
        return optimal

    def _execute_retrieval(
        self,
        query: str,
        strategy: Dict,
        intent: Dict,
        top_k: int
    ) -> List[Chunk]:
        """
        Execute retrieval strategy with optional reranking.

        Args:
            query: User query
            strategy: Selected strategy from EmailStrategySelector
            intent: Detected intent
            top_k: Number of chunks to retrieve (final count after reranking)

        Returns:
            List of retrieved chunks (reranked if reranking is enabled)
        """
        primary_strategy = strategy["primary"]
        params = strategy.get("params", {})

        # Determine initial_k for retrieval (wider net if reranking)
        if self.reranker:
            initial_k = self.reranking_config.get("retrieval", {}).get("initial_k", 100)
            self.logger.debug(
                f"Reranking enabled: retrieving initial_k={initial_k} candidates",
                extra={"run_id": self.run_id, "initial_k": initial_k} if self.run_id else {}
            )
        else:
            initial_k = top_k

        self.logger.debug(
            f"Executing retrieval: strategy={primary_strategy}, params={params}, initial_k={initial_k}",
            extra={"run_id": self.run_id, "strategy": primary_strategy, "params": params, "initial_k": initial_k} if self.run_id else {}
        )

        # Get retriever
        retriever = self.retrievers.get(primary_strategy)
        if not retriever:
            self.logger.warning(
                f"Unknown strategy '{primary_strategy}', falling back to multi_aspect",
                extra={"run_id": self.run_id} if self.run_id else {}
            )
            retriever = self.retrievers["multi_aspect"]

        # Execute retrieval based on strategy type
        if primary_strategy == "thread_retrieval":
            # Thread retrieval uses different parameters
            # Extract days_back from temporal_constraint if present
            days_back = None
            if "temporal_constraint" in params:
                days_back = params["temporal_constraint"].get("days_back")

            chunks = retriever.retrieve(query, top_threads=2, days_back=days_back)

        elif primary_strategy == "multi_aspect":
            # Multi-aspect uses intent directly
            chunks = retriever.retrieve(query, intent=intent, top_k=initial_k)

        elif primary_strategy in ["temporal_retrieval", "sender_retrieval"]:
            # Temporal and sender use intent_metadata
            chunks = retriever.retrieve(query, intent_metadata=params, top_k=initial_k)

        else:
            # Fallback: use multi_aspect
            chunks = self.retrievers["multi_aspect"].retrieve(query, intent=intent, top_k=initial_k)

        # Apply reranking if enabled and we have enough candidates
        if self.reranker and len(chunks) > top_k:
            self.logger.info(
                f"Applying reranking: {len(chunks)} candidates → top {top_k}",
                extra={"run_id": self.run_id, "candidates": len(chunks), "final_k": top_k} if self.run_id else {}
            )

            try:
                score_threshold = self.reranking_config.get("retrieval", {}).get("score_threshold")
                chunks = self.reranker.rerank(
                    query=query,
                    chunks=chunks,
                    top_k=top_k,
                    score_threshold=score_threshold
                )
                self.logger.info(
                    f"Reranking completed: {len(chunks)} chunks returned",
                    extra={"run_id": self.run_id, "reranked_count": len(chunks)} if self.run_id else {}
                )
            except Exception as e:
                self.logger.error(
                    f"Reranking failed: {e}. Returning top {top_k} chunks without reranking.",
                    extra={"run_id": self.run_id, "error": str(e)} if self.run_id else {},
                    exc_info=True
                )
                chunks = chunks[:top_k]

        elif self.reranker and len(chunks) <= top_k:
            self.logger.info(
                f"Skipping reranking: only {len(chunks)} chunks (≤ top_k={top_k})",
                extra={"run_id": self.run_id, "chunk_count": len(chunks)} if self.run_id else {}
            )

        return chunks

    def _build_metadata(self, chunks: List[Chunk], strategy: Dict) -> Dict:
        """
        Build metadata dictionary for transparency.

        Args:
            chunks: Retrieved chunks
            strategy: Strategy used

        Returns:
            Metadata dictionary
        """
        metadata = {
            "chunk_count": len(chunks),
            "strategy_used": strategy["primary"],
            "filters_applied": strategy.get("filters", []),
        }

        # Add date range if chunks have dates
        date_range = self._get_date_range(chunks)
        if date_range:
            metadata["date_range"] = date_range

        # Add unique senders
        unique_senders = self._get_unique_senders(chunks)
        if unique_senders:
            metadata["unique_senders"] = unique_senders

        # Add unique subjects
        unique_subjects = self._get_unique_subjects(chunks)
        if unique_subjects:
            metadata["unique_subjects"] = unique_subjects

        return metadata

    def _get_date_range(self, chunks: List[Chunk]) -> Optional[Dict[str, str]]:
        """
        Extract date range from chunks.

        Args:
            chunks: List of chunks

        Returns:
            {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} or None
        """
        dates = [c.meta.get("date", "") for c in chunks if c.meta.get("date")]
        if not dates:
            return None

        try:
            # Parse dates and find min/max
            parsed_dates = []
            for date_str in dates:
                date_part = date_str.split()[0] if ' ' in date_str else date_str
                parsed_dates.append(datetime.strptime(date_part, "%Y-%m-%d"))

            if parsed_dates:
                min_date = min(parsed_dates).strftime("%Y-%m-%d")
                max_date = max(parsed_dates).strftime("%Y-%m-%d")
                return {"start": min_date, "end": max_date}

        except Exception as e:
            self.logger.warning(
                f"Failed to parse date range: {e}",
                extra={"run_id": self.run_id} if self.run_id else {}
            )

        return None

    def _get_unique_senders(self, chunks: List[Chunk]) -> List[str]:
        """
        Extract unique senders from chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of unique sender names
        """
        senders = set()
        for chunk in chunks:
            sender_name = chunk.meta.get("sender_name")
            if sender_name:
                senders.add(sender_name)

        return sorted(list(senders))

    def _get_unique_subjects(self, chunks: List[Chunk]) -> List[str]:
        """
        Extract unique normalized subjects from chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of unique normalized subjects
        """
        # Use thread retriever's subject normalization
        thread_retriever = self.retrievers["thread_retrieval"]

        subjects = set()
        for chunk in chunks:
            subject = chunk.meta.get("subject")
            if subject:
                normalized = thread_retriever._normalize_subject(subject)
                if normalized:
                    subjects.add(normalized)

        return sorted(list(subjects))[:5]  # Limit to top 5 for readability

    def _log_intent(self, intent: Dict):
        """Log detected intent."""
        self.logger.info(
            f"Intent detected: {intent['primary_intent']} "
            f"(confidence: {intent['confidence']:.2f})",
            extra={
                "run_id": self.run_id,
                "intent": intent['primary_intent'],
                "confidence": intent['confidence'],
                "metadata": intent.get('metadata', {})
            } if self.run_id else {}
        )

    def _log_strategy(self, strategy: Dict):
        """Log selected strategy."""
        filters_str = f" + filters: {strategy['filters']}" if strategy.get('filters') else ""
        self.logger.info(
            f"Strategy selected: {strategy['primary']}{filters_str}",
            extra={
                "run_id": self.run_id,
                "strategy": strategy['primary'],
                "filters": strategy.get('filters', [])
            } if self.run_id else {}
        )
