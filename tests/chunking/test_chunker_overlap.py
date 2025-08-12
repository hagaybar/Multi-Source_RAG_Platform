from scripts.chunking.chunker_v3 import split
from scripts.chunking.rules_v3 import ChunkRule
from scripts.chunking.chunker_v3 import merge_chunks_with_overlap
import re


def test_split_debug_import():
    print(f">>> split() is from: {split.__module__}")


def test_overlap_tokens_are_preserved(monkeypatch):
    # Force rule: max 60 tokens per chunk, overlap 5

    monkeypatch.setattr(
        "scripts.chunking.chunker_v3.get_rule",
        lambda doc_type: ChunkRule(strategy="blank_line", min_tokens=10, max_tokens=60, overlap=5),
    )

    # 6 paragraphs of 20 tokens = 120 tokens total
    para = "word " * 20
    doc = "\n\n".join([para] * 6)
    meta = {"doc_type": "txt"}

    chunks = split(doc, meta)
    print(f"Got {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c.token_count} tokens")

    assert len(chunks) >= 2

    tokens_0 = chunks[0].text.split()
    tokens_1 = chunks[1].text.split()

    # Overlap: last 5 tokens of chunk 0 == first 5 tokens of chunk 1
    assert tokens_0[-5:] == tokens_1[:5], "Overlap tokens were not preserved"


def test_overlap_tokens_are_preserved():
    text = "Para1 " + "word " * 60 + "\n\n" + "Para2 " + "word " * 50
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    meta = {"doc_type": "docx"}

    rule = ChunkRule(strategy="by_paragraph", min_tokens=20, max_tokens=60, overlap=10)

    chunks = merge_chunks_with_overlap(paragraphs, meta, rule)

    assert len(chunks) == 2

    tokens_chunk1 = chunks[0].text.split()
    tokens_chunk2 = chunks[1].text.split()

    expected_overlap = tokens_chunk1[-rule.overlap :]
    actual_overlap = tokens_chunk2[: rule.overlap]

    assert expected_overlap == actual_overlap, (
        f"Expected overlap {expected_overlap}, got {actual_overlap}"
    )
