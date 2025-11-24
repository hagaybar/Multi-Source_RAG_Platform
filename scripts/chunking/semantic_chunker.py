#!/usr/bin/env python3
"""
Semantic Chunking for Email Data

Creates chunks based on topic boundaries rather than fixed sizes.
Uses embedding similarity between sentences to detect topic shifts.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ChunkBoundary:
    """Represents a detected chunk boundary."""
    position: int
    similarity_drop: float
    reason: str


class SemanticChunker:
    """
    Semantic boundary-based chunking for emails.

    Instead of splitting at fixed token counts, this chunker:
    1. Splits text into sentences
    2. Embeds each sentence
    3. Detects topic shifts via similarity drops
    4. Creates chunks at natural topic boundaries

    This preserves semantic coherence and prevents splitting
    related content across chunks.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.65,
        min_chunk_size: int = 50,
        max_chunk_size: int = 500,
        sentence_overlap: int = 1
    ):
        """
        Initialize semantic chunker.

        Args:
            model_name: SentenceTransformer model for embeddings
            similarity_threshold: Similarity below this = topic boundary
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk (hard limit)
            sentence_overlap: Number of sentences to overlap at boundaries
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

        print(f"SemanticChunker initialized with model: {model_name}")
        print(f"  Similarity threshold: {similarity_threshold}")
        print(f"  Chunk size range: {min_chunk_size}-{max_chunk_size} tokens")

    def chunk(self, text: str, metadata: dict = None) -> List[Tuple[str, int]]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata (for logging)

        Returns:
            List of (chunk_text, token_count) tuples
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self.split_sentences(text)

        if len(sentences) <= 1:
            # Single sentence - return as-is if within limits
            token_count = self.count_tokens(text)
            if token_count <= self.max_chunk_size:
                return [(text, token_count)]
            else:
                # Fall back to simple splitting for very long single sentence
                return self._fallback_split(text)

        # Embed sentences
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        # Detect boundaries
        boundaries = self.detect_boundaries(sentences, embeddings)

        # Create chunks respecting boundaries and size limits
        chunks = self.create_chunks(sentences, boundaries)

        return chunks

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with email-aware patterns.

        Handles:
        - Standard sentence boundaries (. ! ?)
        - Email-specific patterns (quoted text, signatures)
        - List items and bullet points
        - Preserves important whitespace
        """
        # Normalize excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Handle common email abbreviations (don't split on these)
        text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
        text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
        text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
        text = re.sub(r'\bProf\.', 'Prof<DOT>', text)
        text = re.sub(r'\be\.g\.', 'e<DOT>g<DOT>', text)
        text = re.sub(r'\bi\.e\.', 'i<DOT>e<DOT>', text)
        text = re.sub(r'\betc\.', 'etc<DOT>', text)

        # Split on sentence boundaries
        # Pattern: . ! ? followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        # Handle very short "sentences" (likely fragments)
        merged_sentences = []
        buffer = ""

        for sentence in sentences:
            if len(sentence.split()) < 3 and buffer:
                # Very short - merge with previous
                buffer += " " + sentence
            else:
                if buffer:
                    merged_sentences.append(buffer)
                buffer = sentence

        if buffer:
            merged_sentences.append(buffer)

        return merged_sentences

    def detect_boundaries(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[ChunkBoundary]:
        """
        Detect semantic boundaries between sentences.

        A boundary is detected when:
        - Cosine similarity drops below threshold
        - Significant topic shift indicated

        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings (N x D)

        Returns:
            List of boundary positions
        """
        boundaries = []

        # Calculate consecutive similarities
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i:i+1],
                embeddings[i+1:i+2]
            )[0][0]

            # Detect boundary
            if sim < self.threshold:
                boundaries.append(ChunkBoundary(
                    position=i + 1,  # Boundary after sentence i
                    similarity_drop=1.0 - sim,
                    reason=f"Topic shift (sim={sim:.3f} < {self.threshold})"
                ))

        return boundaries

    def create_chunks(
        self,
        sentences: List[str],
        boundaries: List[ChunkBoundary]
    ) -> List[Tuple[str, int]]:
        """
        Create chunks respecting semantic boundaries and size limits.

        Strategy:
        1. Start chunk at position 0
        2. Add sentences until:
           - Boundary detected AND min_size reached, OR
           - max_size would be exceeded
        3. Create chunk and start new one

        Args:
            sentences: List of sentences
            boundaries: Detected boundaries

        Returns:
            List of (chunk_text, token_count) tuples
        """
        chunks = []
        boundary_positions = set(b.position for b in boundaries)

        current_chunk_sentences = []
        current_chunk_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            # Check if adding this sentence would exceed max
            would_exceed_max = (current_chunk_tokens + sentence_tokens) > self.max_chunk_size

            # Check if we're at a boundary
            at_boundary = (i in boundary_positions)

            # Check if current chunk meets minimum
            meets_min = current_chunk_tokens >= self.min_chunk_size

            # Decide whether to break chunk
            should_break = False

            if would_exceed_max:
                # Hard limit - must break
                should_break = True
            elif at_boundary and meets_min:
                # Natural boundary and minimum met - break here
                should_break = True

            if should_break and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append((chunk_text, current_chunk_tokens))

                # Start new chunk with overlap
                if self.sentence_overlap > 0:
                    # Keep last N sentences for context
                    overlap_sentences = current_chunk_sentences[-self.sentence_overlap:]
                    current_chunk_sentences = overlap_sentences
                    current_chunk_tokens = sum(
                        self.count_tokens(s) for s in overlap_sentences
                    )
                else:
                    current_chunk_sentences = []
                    current_chunk_tokens = 0

            # Add current sentence to chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append((chunk_text, current_chunk_tokens))

        return chunks

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (simple word-based).

        More accurate would be to use tiktoken, but this is
        fast and sufficient for chunking decisions.
        """
        # Simple approximation: words * 1.3 (accounts for subword tokens)
        words = len(text.split())
        return int(words * 1.3)

    def _fallback_split(self, text: str) -> List[Tuple[str, int]]:
        """
        Fallback splitting for edge cases (very long single sentence).

        Splits on whitespace at approximately max_chunk_size.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = int(len(word.split()) * 1.3)

            if current_tokens + word_tokens > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, current_tokens))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(word)
            current_tokens += word_tokens

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, current_tokens))

        return chunks


def test_semantic_chunker():
    """Test semantic chunker on sample email text."""

    sample_email = """
    Hi team,

    I wanted to update you on the Q3 budget discussion. After reviewing the proposals,
    I think we should allocate an additional $50k to the marketing team. This will
    help us reach our customer acquisition targets for the quarter.

    On a completely different note, I wanted to remind everyone about the upcoming
    office move on September 15th. Please make sure to pack your personal belongings
    by September 10th. The facilities team will handle all office equipment.

    Finally, regarding the new Research Assistant feature in Primo VE - we've received
    excellent feedback from the beta testers. The AI-powered search is working well,
    though some users mentioned the guardrails are a bit too restrictive. We should
    discuss potential adjustments in next week's product meeting.

    Best regards,
    Alice
    """

    chunker = SemanticChunker(
        similarity_threshold=0.65,
        min_chunk_size=30,
        max_chunk_size=200
    )

    chunks = chunker.chunk(sample_email)

    print("\n" + "="*80)
    print("SEMANTIC CHUNKING TEST")
    print("="*80)
    print(f"\nInput: {len(sample_email)} chars, {len(sample_email.split())} words")
    print(f"Output: {len(chunks)} chunks\n")

    for i, (chunk_text, token_count) in enumerate(chunks, 1):
        print(f"Chunk {i} ({token_count} tokens):")
        print(f"  {chunk_text[:100]}...")
        print()


if __name__ == "__main__":
    test_semantic_chunker()
