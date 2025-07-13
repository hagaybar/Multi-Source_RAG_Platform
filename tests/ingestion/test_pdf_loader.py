import pytest
from pathlib import Path
from scripts.ingestion.pdf import load_pdf
from scripts.ingestion.models import UnsupportedFileError

# pytestmark = pytest.mark.legacy_chunker

# -------------------------------------------------------------------
# Fixture paths
# -------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pdf"
SIMPLE_PDF     = FIXTURES_DIR / "simple.pdf"
ENCRYPTED_PDF  = FIXTURES_DIR / "encrypted.pdf"
NO_TEXT_PDF    = FIXTURES_DIR / "no_text.pdf"

# -------------------------------------------------------------------
# Helper – create minimal PDFs if they’re missing
# -------------------------------------------------------------------
def _create_test_pdfs_if_not_exist() -> None:
    """Synthesise lightweight fixture PDFs so the tests run anywhere."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        # ReportLab isn’t installed in CI; skip on-the-fly creation
        return

    if not SIMPLE_PDF.exists():
        c = canvas.Canvas(str(SIMPLE_PDF))
        c.setTitle("Simple Test PDF")
        c.setAuthor("Test Author")
        c.drawString(100, 750, "This is page 1.")
        c.showPage()
        c.drawString(100, 750, "This is page 2.")
        c.showPage()
        c.save()

    if not ENCRYPTED_PDF.exists():
        c = canvas.Canvas(str(ENCRYPTED_PDF))
        c.drawString(100, 750, "This is an encrypted document.")
        c.setEncrypt("password")
        c.showPage()
        c.save()

    if not NO_TEXT_PDF.exists():
        c = canvas.Canvas(str(NO_TEXT_PDF))
        c.showPage()      # blank page → no extractable text
        c.save()


_create_test_pdfs_if_not_exist()

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
class TestPDFLoader:
    def test_load_simple_pdf_success(self):
        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not available.")

        segments = load_pdf(SIMPLE_PDF)

        assert isinstance(segments, list)
        text = "\n\n".join([seg[0] for seg in segments])
        meta = segments[0][1]

        # content
        assert text == "This is page 1.\n\nThis is page 2."

        assert meta["source_path"] == str(SIMPLE_PDF.resolve())
        assert meta["title"] == "Simple Test PDF"
        assert meta["author"] == "Test Author"
        assert meta["num_pages"] == 2
        assert "created" in meta and "modified" in meta

    def test_load_encrypted_pdf_raises_error(self):
        if not ENCRYPTED_PDF.exists():
            pytest.skip("encrypted.pdf fixture not available.")

        with pytest.raises(UnsupportedFileError, match="encrypted and requires a password"):
            load_pdf(ENCRYPTED_PDF)

    def test_load_no_text_pdf_raises_error(self):
        if not NO_TEXT_PDF.exists():
            pytest.skip("no_text.pdf fixture not available.")

        with pytest.raises(UnsupportedFileError, match="No extractable text"):
            load_pdf(NO_TEXT_PDF)

    def test_load_non_existent_pdf_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_pdf(Path("does_not_exist.pdf"))

    def test_load_corrupted_pdf_raises_error(self, tmp_path):
        corrupted = tmp_path / "corrupted.pdf"
        corrupted.write_text("Not a PDF.")

        with pytest.raises(UnsupportedFileError, match="Failed to parse PDF"):
            load_pdf(corrupted)