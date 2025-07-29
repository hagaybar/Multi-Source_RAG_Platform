from .csv import load_csv
from .docx_loader import load_docx
from .email_loader import load_eml
from .xlsx import XlsxIngestor  # Import XlsxIngestor class
from .pdf import load_pdf  # Add this import
from .pptx import PptxIngestor  # Import PptxIngestor


# Simple loader for .txt files
def load_txt(filepath: str) -> tuple[str, dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Basic metadata, IngestionManager will add/override doc_type and source_filepath
    return content, {}


LOADER_REGISTRY = {
    ".txt": load_txt,  # Added .txt loader
    ".csv": load_csv,
    ".docx": load_docx,
    ".eml": load_eml,
    ".pdf": load_pdf,  # Add this mapping
    ".pptx": PptxIngestor,  # Map .pptx to PptxIngestor class
    ".xlsx": XlsxIngestor,  # Map .xlsx to XlsxIngestor class
}
