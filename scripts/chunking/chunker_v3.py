# scripts/chunking/chunker_v3.py
"""
Chunker v3 - Dispatcher

This module acts as a dispatcher for the various chunking strategies.
It imports the strategies from chunking_strategies.py and calls the
appropriate one based on the doc_type.
"""

from typing import Any, Dict, List
from scripts.chunking.models import Chunk
from scripts.chunking.rules_v3 import get_rule
from scripts.utils.email_utils import clean_email_text
from scripts.utils.logger import LoggerManager

from scripts.chunking.chunking_strategies import (
    chunk_by_paragraph,
    chunk_by_slide,
    chunk_by_sheet,
    chunk_by_blank_line,
    chunk_by_row,
    chunk_by_email_block,
    parent_child_chunker,
)

# Default logger - will be used if no project-specific logger is provided
_default_logger = LoggerManager.get_logger("chunker")

# Strategy registry mapping rule strategies to functions
STRATEGY_REGISTRY = {
    "by_paragraph": chunk_by_paragraph,
    "paragraph": chunk_by_paragraph,
    "by_slide": chunk_by_slide,
    "slide": chunk_by_slide,
    "split_on_sheets": chunk_by_sheet,
    "sheet": chunk_by_sheet,
    "sheets": chunk_by_sheet,
    "blank_line": chunk_by_blank_line,
    "split_on_rows": chunk_by_row,
    "by_email_block": chunk_by_email_block,
    "eml": chunk_by_email_block,
    "parent_child": parent_child_chunker,
}


def split(text: str, meta: dict, clean_options: dict = None, logger=None) -> list[Chunk]:
    if logger is None:
        logger = _default_logger

    doc_type = meta.get('doc_type')
    if not doc_type:
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

    cleaned_text = clean_email_text(text, **clean_options)
    rule = get_rule(doc_type)

    strategy_func = STRATEGY_REGISTRY.get(rule.strategy)
    if not strategy_func:
        raise ValueError(f"Unsupported strategy: {rule.strategy}")

    return strategy_func(cleaned_text, meta, rule, logger)
