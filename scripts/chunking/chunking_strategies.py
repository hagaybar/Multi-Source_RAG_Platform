# scripts/chunking/chunking_strategies.py
"""
This module contains the various strategies for chunking documents. Each
strategy is implemented as a separate function that takes the document text
and metadata, and returns a list of chunks.
"""

import re
import uuid
from typing import Any, Dict, List, Optional

from scripts.chunking.models import Chunk
from scripts.chunking.rules_v3 import ChunkRule, get_rule
from scripts.utils.logger import LoggerManager

import spacy

# Default logger - will be used if no project-specific logger is provided
_default_logger = LoggerManager.get_logger("chunking_strategies")

# --- regex patterns ----------------------------------------------------------
PARA_REGEX = re.compile(r"\n\s*\n")  # one or more blank lines
HEADING_REGEX = re.compile(r"\n(?=#+\s)") # Split on newlines followed by '#'
EMAIL_BLOCK_REGEX = re.compile(
    r"(\n\s*(?:From:|On .* wrote:))"
)  # email block separator with capturing group


# --- helpers -----------------------------------------------------------------
def _token_count(text: str) -> int:
    """Very rough token counter; will be replaced by real tokenizer later."""
    return len(text.split())


def build_chunk(text: str, meta: dict, token_count: int, doc_id: str) -> Chunk:
    chunk_id = uuid.uuid4().hex
    meta_copy = meta.copy()
    meta_copy["id"] = chunk_id
    return Chunk(
        doc_id=doc_id,
        text=text,
        meta=meta_copy,
        token_count=token_count,
        id=chunk_id,
    )


def merge_chunks_with_overlap(
    paragraphs: list[str], meta: dict, rule: ChunkRule, logger=None
) -> list[Chunk]:
    if logger is None:
        logger = _default_logger

    doc_id = meta.get('doc_id', 'unknown_doc_id')
    chunks = []
    buffer = []
    buffer_tokens = 0
    prev_tail_tokens: list[str] = []

    logger.debug(f"Using rule for '{meta['doc_type']}': {rule}")

    for para in paragraphs:
        para_tokens = _token_count(para)

        if buffer_tokens + para_tokens >= rule.max_tokens:
            chunk_tokens = " ".join(prev_tail_tokens + buffer).split()
            chunk_text = " ".join(chunk_tokens)
            if len(chunk_tokens) >= rule.min_tokens or meta.get("image_paths"):
                chunks.append(build_chunk(chunk_text, meta, len(chunk_tokens), doc_id))
            else:
                logger.debug(
                    f"[MERGE] Skipped chunk with only {len(chunk_tokens)} tokens and no image_paths"
                )

            prev_tail_tokens = chunk_tokens[-rule.overlap :] if rule.overlap else []
            buffer = []
            buffer_tokens = 0

        buffer.append(para)
        buffer_tokens += para_tokens

    # Final flush
    if buffer:
        chunk_tokens = " ".join(prev_tail_tokens + buffer).split()
        if chunk_tokens:
            chunk_text = " ".join(chunk_tokens)
            chunks.append(build_chunk(chunk_text, meta, len(chunk_tokens), doc_id))

    return chunks


# --- Chunking Strategies -----------------------------------------------------

def chunk_by_paragraph(cleaned_text: str, meta: dict, rule: ChunkRule, logger) -> list[Chunk]:
    """Chunk document by paragraphs."""
    items = [p.strip() for p in PARA_REGEX.split(cleaned_text.strip()) if p.strip()]
    return merge_chunks_with_overlap(items, meta, rule, logger)


def chunk_by_slide(cleaned_text: str, meta: dict, rule: ChunkRule, logger) -> list[Chunk]:
    """Chunk document by slides, with '---' as a separator."""
    items = [s.strip() for s in cleaned_text.strip().split("\n---\n") if s.strip()]
    return merge_chunks_with_overlap(items, meta, rule, logger)


def chunk_by_sheet(cleaned_text: str, meta: dict, rule: ChunkRule, logger) -> list[Chunk]:
    """Treat the entire sheet as a single chunk."""
    items = [cleaned_text.strip()] if cleaned_text.strip() else []
    return merge_chunks_with_overlap(items, meta, rule, logger)


def chunk_by_blank_line(cleaned_text: str, meta: dict, rule: ChunkRule, logger) -> list[Chunk]:
    """Chunk document by blank lines."""
    items = [b.strip() for b in cleaned_text.strip().split("\n\n") if b.strip()]
    return merge_chunks_with_overlap(items, meta, rule, logger)


def chunk_by_row(cleaned_text: str, meta: dict, rule: ChunkRule, logger) -> list[Chunk]:
    """Chunk a CSV-like document by rows."""
    items = [row.strip() for row in cleaned_text.strip().split('\n') if row.strip()]
    return merge_chunks_with_overlap(items, meta, rule, logger)


def chunk_by_email_block(cleaned_text: str, meta: dict, rule: ChunkRule, logger) -> list[Chunk]:
    """Chunk an email by semantic blocks."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(cleaned_text)
    items = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return merge_chunks_with_overlap(items, meta, rule, logger)


def parent_child_chunker(cleaned_text: str, meta: dict, rule: ChunkRule, logger) -> list[Chunk]:
    """
    Chunk a document into parent and child chunks based on headings.
    Each heading and its content becomes a parent chunk, and paragraphs
    within it become child chunks.
    """
    if logger is None:
        logger = _default_logger

    doc_id = meta.get('doc_id', 'unknown_doc_id')
    all_chunks = []

    # Split the document by headings to get parent chunks
    parent_sections = HEADING_REGEX.split(cleaned_text)

    for section in parent_sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n')
        heading = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()

        # Create the parent chunk
        parent_chunk_text = f"{heading}\n{content}"
        parent_token_count = _token_count(parent_chunk_text)

        # We can use a simplified rule for parent chunks, or the same rule
        # For now, let's assume parent chunks are not constrained by max_tokens
        # but this could be adjusted.
        parent_chunk = build_chunk(parent_chunk_text, meta, parent_token_count, doc_id)
        parent_chunk.title = heading
        all_chunks.append(parent_chunk)

        # Create child chunks from the content of the parent
        paragraphs = [p.strip() for p in PARA_REGEX.split(content) if p.strip()]

        for para in paragraphs:
            child_token_count = _token_count(para)
            if child_token_count >= rule.min_tokens:
                child_chunk = build_chunk(para, meta, child_token_count, doc_id)
                child_chunk.parent_id = parent_chunk.id
                all_chunks.append(child_chunk)
            else:
                logger.debug(
                    f"Skipped child chunk with {child_token_count} tokens, less than min_tokens={rule.min_tokens}"
                )

    return all_chunks
