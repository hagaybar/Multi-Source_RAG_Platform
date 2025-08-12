# tests/test_chunker_paragraph.py
from scripts.chunking.chunking_strategies import merge_chunks_with_overlap
from scripts.chunking.rules_v3 import ChunkRule

import re
import pytest


DOC_TEXT = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."


def test_force_two_chunks_with_overlap_merge():
    text = "Para1 " + "word " * 56 + "\n\n" + "Para2 " + "word " * 56
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    meta = {"doc_type": "docx"}

    rule = ChunkRule(strategy="by_paragraph", min_tokens=20, max_tokens=80, overlap=5)

    chunks = merge_chunks_with_overlap(paragraphs, meta, rule)

    print(f"Returned {len(chunks)} chunks.")
    for i, c in enumerate(chunks):
        print(f"Chunk {i + 1}: {len(c.text.split())} tokens")

    assert len(chunks) == 2
