# Ingestion Module

This directory holds the loaders and helper classes that turn raw files into `RawDoc` objects. It forms the first step of the pipeline before chunking and embedding.

## Key Components
- `__init__.py` – exposes `LOADER_REGISTRY` mapping file extensions to loader functions or ingestor classes. Includes a simple `load_txt` helper.
- `models.py` – defines the `RawDoc` dataclass, the abstract `AbstractIngestor` base class, and `UnsupportedFileError`.
- `manager.py` – `IngestionManager` walks through a path, dispatches each file to the appropriate loader and returns a list of `RawDoc` objects. Logging is handled via `scripts.utils.logger`.

## Loaders
- `csv.py` – `load_csv(path)` reads a CSV file and returns the concatenated text with metadata.
- `docx_loader.py` – `load_docx(path)` extracts paragraphs, tables and any embedded images from DOCX files. A legacy version exists in `docx_loader_old_0.py`.
- `email_loader.py` – `load_eml(path)` parses `.eml` files and extracts the plain text body.
- `pdf.py` – `load_pdf(path)` uses `pdfplumber` to extract per-page text and images, raising `UnsupportedFileError` for encrypted or corrupted PDFs.
- `pptx.py` – `PptxIngestor` splits PowerPoint slides into text segments (slide content and presenter notes) and saves images.
- `xlsx.py` – `XlsxIngestor` reads Excel workbooks and produces chunks of rows per sheet.

## How it fits together
`IngestionManager.ingest_path()` is typically called by the CLI or UI to process a directory of source files. The resulting `RawDoc` list is then fed to the chunking engine in `scripts/chunking` for further processing within the RAG workflow.
