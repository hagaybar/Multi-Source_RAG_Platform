import tempfile
from pathlib import Path
import pytest

from scripts.ingestion import csv as csv_loader
from scripts.chunking import chunker_v3
from scripts.chunking.rules_v3 import get_rule
from scripts.chunking.models import Chunk


def count_tokens(text: str) -> int:
    return len(text.split())


def test_chunker_csv_split_on_rows():
    # ---- Generate realistic CSV content ----
    header = "H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14"

    def row(idx):
        return " ".join([f"R{idx}_C{j}" for j in range(1, 21)])

    row_count = 50  # 20 tokens per row
    rows = [header] + [row(i) for i in range(1, row_count + 1)]
    content = "\n".join(rows)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # ---- Load and chunk ----
        text_content, meta = csv_loader.load_csv(str(tmp_path))
        rule = get_rule("csv")
        chunks: list[Chunk] = chunker_v3.split(text_content, meta)

        assert len(chunks) >= 2, "Expected multiple chunks from long CSV"
        assert chunks[0].meta["doc_type"] == "csv"

        ROW_TOKENS = 20
        HEADER_TOKENS = len(header.split())

        # ---- Check token bounds per chunk ----
        for i, c in enumerate(chunks):
            allowed = rule.max_tokens + ROW_TOKENS  # Allow 1-row overflow
            assert c.token_count <= allowed, f"Chunk {i} exceeds max_tokens by more than one row"
            if i < len(chunks) - 1:
                assert c.token_count >= rule.min_tokens, f"Chunk {i} is below min_tokens"

        # ---- Overlap logic check (only if meaningful) ----
        if len(chunks) >= 2:
            chunk1_words = chunks[0].text.split()
            chunk2_words = chunks[1].text.split()

            # Skip header in chunk1 if present
            if chunk1_words[:HEADER_TOKENS] == header.split():
                chunk1_words = chunk1_words[HEADER_TOKENS:]

            overlap = rule.overlap or 0

            if len(chunk1_words) >= overlap:
                overlap_words_1 = chunk1_words[-overlap:]
                overlap_words_2 = chunk2_words[:overlap]

                assert overlap_words_1 == overlap_words_2, (
                    f"Token overlap mismatch.\n"
                    f"Chunk1 tail: {' '.join(overlap_words_1)}\n"
                    f"Chunk2 head: {' '.join(overlap_words_2)}"
                )

    finally:
        tmp_path.unlink()
