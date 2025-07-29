import pytest
from pathlib import Path
from scripts.ingestion.xlsx import XlsxIngestor

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "xlsx"
FIXTURE_FILE = FIXTURE_DIR / "demo.xlsx"


@pytest.mark.parametrize("file_path", [FIXTURE_FILE])
def test_xlsx_ingestor_loads_sheets(file_path):
    ingestor = XlsxIngestor()
    segments = ingestor.ingest(str(file_path))

    assert isinstance(segments, list), "Returned result must be a list"
    assert len(segments) >= 2, "Expected at least 2 sheets (segments)"

    for text, meta in segments:
        assert isinstance(text, str)
        assert text.strip(), "Extracted text should not be empty"
        assert isinstance(meta, dict)
        assert "sheet_name" in meta
        assert meta["doc_type"] == "xlsx"
        assert meta["type"] == "sheet_content"
