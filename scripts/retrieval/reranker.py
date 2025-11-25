"""
Cross-Encoder Reranker for Improving Retrieval Relevance

This module implements reranking using cross-encoder models that evaluate
query-document pairs for relevance. Unlike bi-encoders (embeddings), cross-encoders
can capture complex interactions between query and document.

Usage:
    reranker = CrossEncoderReranker()
    reranked_chunks = reranker.rerank(query, chunks, top_k=15)
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
import logging
from scripts.chunking.models import Chunk


logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model for better relevance.

    Cross-encoders evaluate query-document pairs jointly, providing more accurate
    relevance scores than cosine similarity from embeddings.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name. Options:
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (default, good balance)
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (faster, slightly lower quality)
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, lower quality)
        """
        self.model_name = model_name
        logger.info(f"Loading cross-encoder model: {model_name}")

        try:
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info(f"Cross-encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 15,
        score_threshold: float = None
    ) -> List[Chunk]:
        """
        Rerank chunks by relevance to the query.

        Args:
            query: The search query
            chunks: List of chunks to rerank
            top_k: Number of top chunks to return
            score_threshold: Optional minimum score (0-1). Chunks below this are filtered out.

        Returns:
            List of reranked chunks (top_k), sorted by relevance score (highest first)
        """
        if not chunks:
            logger.warning("No chunks to rerank")
            return []

        if len(chunks) <= top_k:
            logger.info(f"Chunks ({len(chunks)}) <= top_k ({top_k}), reranking all")

        # Get relevance scores
        scores = self.get_relevance_scores(query, chunks)

        # Combine chunks with scores
        scored_chunks = list(zip(chunks, scores))

        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Apply score threshold if specified
        if score_threshold is not None:
            scored_chunks = [(c, s) for c, s in scored_chunks if s >= score_threshold]
            logger.info(f"After threshold {score_threshold}: {len(scored_chunks)} chunks remain")

        # Take top-k
        top_chunks = scored_chunks[:top_k]

        # Store rerank scores in chunk metadata
        reranked_chunks = []
        for chunk, score in top_chunks:
            # Create a copy to avoid modifying original
            chunk_copy = Chunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                token_count=chunk.token_count,
                meta=chunk.meta.copy() if chunk.meta else {}
            )
            chunk_copy.meta["rerank_score"] = float(score)
            reranked_chunks.append(chunk_copy)

        logger.info(
            f"Reranked {len(chunks)} chunks → {len(reranked_chunks)} chunks "
            f"(top_k={top_k}, scores: {top_chunks[0][1]:.3f} to {top_chunks[-1][1]:.3f})"
        )

        return reranked_chunks

    def get_relevance_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        """
        Get relevance scores for query-chunk pairs.

        Args:
            query: The search query
            chunks: List of chunks to score

        Returns:
            List of relevance scores (0-1, higher = more relevant)
        """
        if not chunks:
            return []

        # Prepare query-document pairs
        pairs = [[query, chunk.text] for chunk in chunks]

        # Get scores from cross-encoder
        # Returns logits that we convert to scores
        try:
            raw_scores = self.model.predict(pairs, show_progress_bar=False)

            # Normalize scores to 0-1 range using sigmoid-like scaling
            # Cross-encoder scores are typically in range [-10, 10]
            # We'll use a simple normalization
            import numpy as np

            # Convert to numpy if not already
            raw_scores = np.array(raw_scores)

            # Sigmoid normalization: 1 / (1 + exp(-x))
            normalized_scores = 1 / (1 + np.exp(-raw_scores))

            scores = normalized_scores.tolist()

            logger.debug(
                f"Scored {len(chunks)} chunks: "
                f"raw range [{raw_scores.min():.2f}, {raw_scores.max():.2f}], "
                f"normalized range [{min(scores):.3f}, {max(scores):.3f}]"
            )

            return scores

        except Exception as e:
            logger.error(f"Error scoring chunks: {e}")
            # Return neutral scores as fallback
            return [0.5] * len(chunks)


class CohereReranker:
    """
    Reranker using Cohere's Rerank API (requires API key and costs money).

    Alternative to CrossEncoderReranker. Typically higher quality but has cost.
    Cost: ~$1-2 per 1K rerank requests.
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Cohere rerank model name
        """
        try:
            import cohere
            self.client = cohere.Client(api_key)
            self.model = model
            logger.info(f"Cohere reranker initialized with model: {model}")
        except ImportError:
            raise ImportError(
                "Cohere package not installed. Install with: poetry add cohere"
            )

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 15
    ) -> List[Chunk]:
        """
        Rerank chunks using Cohere Rerank API.

        Args:
            query: The search query
            chunks: List of chunks to rerank
            top_k: Number of top chunks to return

        Returns:
            List of reranked chunks, sorted by relevance
        """
        if not chunks:
            return []

        # Prepare documents
        documents = [chunk.text for chunk in chunks]

        # Call Cohere rerank API
        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model=self.model
            )

            # Extract reranked results
            reranked_chunks = []
            for result in response.results:
                idx = result.index
                score = result.relevance_score

                chunk = chunks[idx]
                chunk_copy = Chunk(
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    meta=chunk.meta.copy() if chunk.meta else {}
                )
                chunk_copy.meta["rerank_score"] = float(score)
                reranked_chunks.append(chunk_copy)

            logger.info(f"Cohere reranked {len(chunks)} → {len(reranked_chunks)} chunks")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            # Return original chunks as fallback
            return chunks[:top_k]


class LLMReranker:
    """
    Reranks chunks using an LLM (gpt-4o-mini) for better semantic understanding.

    This approach uses the LLM to score relevance, which provides:
    - Better understanding of nuanced queries
    - Domain-specific knowledge
    - Explainable scoring

    Cost: ~$0.003 per query (100 candidates)
    Latency: ~3-8 seconds for 100 candidates (batched)
    """

    def __init__(self, model: str = "gpt-4o-mini", batch_size: int = 20):
        """
        Initialize LLM reranker.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            batch_size: How many chunks to score per API call (default: 20)
        """
        self.model = model
        self.batch_size = batch_size
        logger.info(f"LLM reranker initialized: model={model}, batch_size={batch_size}")

        # Initialize OpenAI client
        from scripts.api_clients.openai.completer import OpenAICompleter
        self.client = OpenAICompleter()

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 15,
        score_threshold: float = None
    ) -> List[Chunk]:
        """
        Rerank chunks using LLM.

        Args:
            query: The search query
            chunks: List of chunks to rerank
            top_k: Number of top chunks to return
            score_threshold: Optional minimum score (0-10)

        Returns:
            List of reranked chunks, sorted by relevance
        """
        if not chunks:
            logger.warning("No chunks to rerank")
            return []

        if len(chunks) <= top_k:
            logger.info(f"Chunks ({len(chunks)}) <= top_k ({top_k}), reranking all")

        # Get relevance scores
        scores = self.get_relevance_scores(query, chunks)

        # Combine chunks with scores
        scored_chunks = list(zip(chunks, scores))

        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Apply score threshold if specified
        if score_threshold is not None:
            scored_chunks = [(c, s) for c, s in scored_chunks if s >= score_threshold]
            logger.info(f"After threshold {score_threshold}: {len(scored_chunks)} chunks remain")

        # Take top-k
        top_chunks = scored_chunks[:top_k]

        # Store rerank scores in chunk metadata (normalize to 0-1)
        reranked_chunks = []
        for chunk, score in top_chunks:
            chunk_copy = Chunk(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                token_count=chunk.token_count,
                meta=chunk.meta.copy() if chunk.meta else {}
            )
            # Normalize 0-10 score to 0-1 range
            chunk_copy.meta["rerank_score"] = float(score) / 10.0
            chunk_copy.meta["rerank_score_raw"] = float(score)
            reranked_chunks.append(chunk_copy)

        if top_chunks:
            logger.info(
                f"Reranked {len(chunks)} chunks → {len(reranked_chunks)} chunks "
                f"(top_k={top_k}, scores: {top_chunks[0][1]:.1f} to {top_chunks[-1][1]:.1f})"
            )

        return reranked_chunks

    def get_relevance_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        """
        Get relevance scores using LLM in batches.

        Args:
            query: The search query
            chunks: List of chunks to score

        Returns:
            List of relevance scores (0-10, higher = more relevant)
        """
        if not chunks:
            return []

        all_scores = []

        # Process in batches to avoid token limits and reduce latency
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_scores = self._score_batch(query, batch)
            all_scores.extend(batch_scores)

            logger.debug(
                f"Scored batch {i//self.batch_size + 1}/{(len(chunks)-1)//self.batch_size + 1}: "
                f"{len(batch)} chunks"
            )

        return all_scores

    def _score_batch(self, query: str, chunks: List[Chunk]) -> List[float]:
        """Score a batch of chunks using LLM."""

        # Build scoring prompt
        prompt = self._build_scoring_prompt(query, chunks)

        try:
            # Call LLM
            response = self.client.complete(
                prompt=prompt,
                model=self.model,
                temperature=0.0,  # Deterministic scoring
                max_tokens=500,   # Enough for scores + brief reasoning
            )

            # Parse scores from response
            scores = self._parse_scores(response, len(chunks))

            return scores

        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            # Return neutral scores as fallback
            return [5.0] * len(chunks)

    def _build_scoring_prompt(self, query: str, chunks: List[Chunk]) -> str:
        """Build prompt for LLM scoring."""

        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            # Include email metadata for context
            subject = chunk.meta.get("subject", "N/A")
            sender = chunk.meta.get("sender_name", "N/A")
            date = chunk.meta.get("date", "N/A")

            # Truncate long chunks to save tokens
            text_preview = chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text

            chunks_text += f"\n[Chunk {i}]\n"
            chunks_text += f"Subject: {subject}\n"
            chunks_text += f"From: {sender} ({date})\n"
            chunks_text += f"Content: {text_preview}\n"

        prompt = f"""You are a relevance scoring assistant. Your task is to score how relevant each email chunk is to answering the user's query.

Query: "{query}"

Email Chunks:
{chunks_text}

Instructions:
1. Score each chunk from 0-10 based on relevance to the query:
   - 0-2: Not relevant (completely off-topic)
   - 3-5: Somewhat relevant (tangentially related)
   - 6-8: Relevant (contains useful information)
   - 9-10: Highly relevant (directly answers the query)

2. Consider:
   - Does the chunk contain information that helps answer the query?
   - Is it about the right topic, person, time period, or issue?
   - Does it provide specific details vs. general discussion?

3. Output format: One score per line
   Chunk 1: [score]
   Chunk 2: [score]
   ...

Provide only the scores, one per line, in order."""

        return prompt

    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse scores from LLM response."""
        import re

        scores = []

        # Extract scores using regex (looking for patterns like "Chunk 1: 7" or just "7")
        score_pattern = r'(?:Chunk\s+\d+:\s*)?(\d+(?:\.\d+)?)'
        matches = re.findall(score_pattern, response)

        for match in matches[:expected_count]:
            try:
                score = float(match)
                # Clamp to 0-10 range
                score = max(0.0, min(10.0, score))
                scores.append(score)
            except ValueError:
                scores.append(5.0)  # Neutral score if parsing fails

        # If we didn't get enough scores, pad with neutral
        while len(scores) < expected_count:
            scores.append(5.0)

        return scores[:expected_count]


def create_reranker(reranker_type: str = "cross-encoder", **kwargs):
    """
    Factory function to create a reranker instance.

    Args:
        reranker_type: Type of reranker ("cross-encoder", "llm", or "cohere")
        **kwargs: Additional arguments for the reranker

    Returns:
        Reranker instance
    """
    if reranker_type == "cross-encoder":
        model_name = kwargs.get("model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        return CrossEncoderReranker(model_name=model_name)

    elif reranker_type == "llm":
        model = kwargs.get("model", "gpt-4o-mini")
        batch_size = kwargs.get("batch_size", 20)
        return LLMReranker(model=model, batch_size=batch_size)

    elif reranker_type == "cohere":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Cohere reranker requires 'api_key' parameter")
        model = kwargs.get("model", "rerank-english-v2.0")
        return CohereReranker(api_key=api_key, model=model)

    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
