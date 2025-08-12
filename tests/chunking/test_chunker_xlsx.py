import pytest
from pathlib import Path
from scripts.chunking import chunker_v3
from scripts.chunking.models import Chunk

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "xlsx"
FIXTURE_FILE = FIXTURE_DIR / "demo.xlsx"


@pytest.mark.parametrize("file_path", [FIXTURE_FILE])
def test_chunker_xlsx_split_on_sheets(file_path):
    from scripts.ingestion.xlsx import XlsxIngestor

    ingestor = XlsxIngestor()
    segments = ingestor.ingest(str(file_path))
    assert segments, "No segments returned from XLSX ingestor"

    for i, (text, meta) in enumerate(segments):
        meta["doc_type"] = "xlsx"
        chunks: list[Chunk] = chunker_v3.split(text, meta)

        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.token_count >= 0 for c in chunks)

        total_tokens = sum(c.token_count for c in chunks)
        raw_tokens = len(text.split())

        assert total_tokens >= int(0.9 * raw_tokens), f"Too much token loss in sheet {i + 1}"

        print(f"✅ Sheet {i + 1}: {len(chunks)} chunks, {total_tokens} tokens total")
