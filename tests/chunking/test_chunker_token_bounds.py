from scripts.chunking.chunker_v3 import split
from scripts.chunking.chunking_strategies import merge_chunks_with_overlap
from scripts.chunking.rules_v3 import ChunkRule
import re


def test_split_enforces_max_tokens():
    para = "word " * 30  # 30 tokens per paragraph
    doc = "\n\n".join([para] * 10)  # 10 x 30 = 300 tokens
    meta = {"doc_type": "test_txt_small"}

    chunks = split(doc, meta)

    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c.token_count} tokens")

    # If no merging, we get 10 chunks of 30
    # If merging works, we should get fewer chunks of ~90–100 tokens
    assert len(chunks) < 10, "Chunks were not merged as expected"
    assert all(c.token_count <= 100 for c in chunks)
    assert all(c.token_count >= 30 for c in chunks)


def test_merge_short_paragraphs_retains_last_chunk():
    text = "ShortPara1 " + "word " * 20 + "\n\n" + "ShortPara2 " + "word " * 15
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    meta = {"doc_type": "pdf"}

    rule = ChunkRule(strategy="by_paragraph", min_tokens=40, max_tokens=100, overlap=0)

    chunks = merge_chunks_with_overlap(paragraphs, meta, rule)

    assert len(chunks) == 1
    token_count = len(chunks[0].text.split())
    assert token_count == 37  # 21 + 16 = 37 words
