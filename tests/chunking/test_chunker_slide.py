from scripts.chunking.chunker_v3 import split
from scripts.chunking.models import Chunk
from scripts.chunking.rules_v3 import ChunkRule
import scripts.chunking.rules_v3 as rules_v3


def test_by_slide_strategy_merges_slides(monkeypatch):
    # Mock rule for pptx
    monkeypatch.setattr(
        "scripts.chunking.chunker_v3.get_rule",
        lambda doc_type: ChunkRule(
            strategy="by_slide",
            min_tokens=10,
            max_tokens=50,
            overlap=5,
        ),
    )

    # Simulate 4 slides of 15 tokens each = 60 tokens
    slide = "word " * 15
    slides = "\n---\n".join([slide] * 4)
    meta = {"doc_type": "pptx"}

    chunks = split(slides, meta)

    print(f"Generated {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c.token_count} tokens")

    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) >= 2  # expect 2 or more chunks due to max_tokens = 50

    # Check overlap
    if len(chunks) >= 2:
        assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]
