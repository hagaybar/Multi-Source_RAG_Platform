# scripts/chunking/chunker_v3.py
"""
Chunker v3 – first slice: paragraph strategy.

For this initial step we:
1. Keep the public signature split(text, meta) -> list
2. Implement a *very* simple paragraph splitter: blank-line separates paragraphs
3. Return a plain list[str] so existing tests pass
"""

import re
import uuid
from typing import Any, Dict, List, Optional
from scripts.chunking.models import Chunk
from scripts.chunking.rules_v3 import ChunkRule
from scripts.chunking.rules_v3 import get_rule
from scripts.utils.email_utils import clean_email_text
import spacy

# # Logging setup
# Ensure you have a logger set up for your application
from scripts.utils.logger import LoggerManager

# Default logger - will be used if no project-specific logger is provided
_default_logger = LoggerManager.get_logger("chunker")

# --- regex patterns ----------------------------------------------------------
PARA_REGEX = re.compile(r"\n\s*\n")  # one or more blank lines
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
        # logger.debug(f"New paragraph: {para_tokens} tokens")
        # logger.debug(f"Buffer before merge check: {buffer_tokens} tokens")
        # logger.debug(f"[RULE] max_tokens: {rule.max_tokens}")

        if buffer_tokens + para_tokens >= rule.max_tokens:
            chunk_tokens = " ".join(prev_tail_tokens + buffer).split()
            chunk_text = " ".join(chunk_tokens)
            if len(chunk_tokens) >= rule.min_tokens or meta.get("image_paths"):
                chunks.append(build_chunk(chunk_text, meta, len(chunk_tokens), doc_id))
                logger.debug(
                    f"[MERGE] Created chunk with {len(chunk_tokens)} tokens "
                    f"(image-aware pass)"
                )
            else:
                logger.debug(
                    f"[MERGE] Skipped chunk with only {len(chunk_tokens)} tokens and no image_paths"
                )

            prev_tail_tokens = chunk_tokens[-rule.overlap :] if rule.overlap else []
            buffer = []
            buffer_tokens = 0
            # logger.debug(f"[MERGE] Tail tokens kept for overlap: {len(prev_tail_tokens)}")

        buffer.append(para)
        buffer_tokens += para_tokens
        # logger.debug(f"Added to buffer: {para_tokens} tokens, buffer now {buffer_tokens} tokens")

    # Final flush
    if buffer:
        chunk_tokens = " ".join(prev_tail_tokens + buffer).split()
        if chunk_tokens:
            chunk_text = " ".join(chunk_tokens)
            chunks.append(build_chunk(chunk_text, meta, len(chunk_tokens), doc_id))

            # logger.debug(f"[FINAL] Created final chunk with {len(chunk_tokens)} tokens")
        else:
            pass
            # logger.debug("[FINAL] Skipped empty buffer, no final chunk created")

    # logger.info(f"Total chunks created: {len(chunks)} for doc_id: {doc_id}")
    return chunks


def split(text: str, meta: dict, clean_options: dict = None, logger=None) -> list[Chunk]:
    if logger is None:
        logger = _default_logger
    # Validate doc_type presence in meta
    doc_type = meta.get('doc_type')
    if not doc_type:  # Covers None or empty string
        raise ValueError(
            "`doc_type` must be present in `meta` and non-empty to determine chunking strategy."
        )

    if clean_options is None:
        clean_options = {
            "remove_quoted_lines": True,
            "remove_reply_blocks": True,
            "remove_signature": True,
            "signature_delimiter": "-- ",
        }

    # Clean the email text using the provided options
    cleaned_text = clean_email_text(text, **clean_options)

    rule = get_rule(meta["doc_type"])

    if rule.strategy in ("by_paragraph", "paragraph"):
        items = [p.strip() for p in PARA_REGEX.split(cleaned_text.strip()) if p.strip()]
    elif rule.strategy in ("by_slide", "slide"):
        items = [s.strip() for s in cleaned_text.strip().split("\n---\n") if s.strip()]
    elif rule.strategy in ("split_on_sheets", "sheet", "sheets"):
        items = [cleaned_text.strip()] if cleaned_text.strip() else []
    elif rule.strategy in ("blank_line",):
        items = [b.strip() for b in cleaned_text.strip().split("\n\n") if b.strip()]
    elif rule.strategy == "split_on_rows":
        # Each line in the text is a row from the CSV
        items = [row.strip() for row in cleaned_text.strip().split('\n') if row.strip()]
    elif rule.strategy in ("by_email_block", "eml"):
        # Split the cleaned text into sentences using spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(cleaned_text)
        items = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        raise ValueError(f"Unsupported strategy: {rule.strategy}")

    # logger.debug(f"[SPLIT] Raw text length: {len(text)}")
    # logger.debug(f"[SPLIT] Using strategy: {rule.strategy}")
    # logger.debug(f"[SPLIT] Paragraph count: {len(items)}")
    # for i, item in enumerate(items):
    #     logger.debug(f"[SPLIT] Paragraph {i+1} ({_token_count(item)} tokens): {repr(item[:60])}...")

    return merge_chunks_with_overlap(items, meta, rule, logger)
