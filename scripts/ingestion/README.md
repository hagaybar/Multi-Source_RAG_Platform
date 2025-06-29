# Ingestion Folder

The `scripts/ingestion` folder is responsible for loading and parsing documents from various file formats. This is the first step in the data processing pipeline for the RAG system, preparing raw content for subsequent chunking, embedding, and indexing.

This folder contains:

- `__init__.py`: Initializes the `LOADER_REGISTRY`. This registry maps file extensions (e.g., ".pdf", ".docx") to their corresponding loader functions or classes.
    - It currently includes:
        - `load_txt`: A simple function to read plain text files (`.txt`).
        - `load_csv`: A function to load and concatenate content from CSV files (`.csv`).
        - `load_docx`: A function to extract text from Microsoft Word files (`.docx`).
        - `load_eml`: A function to parse and extract text from email files (`.eml`).
        - `load_pdf`: A function to extract text from PDF files (`.pdf`).
        - `PptxIngestor`: A class-based ingestor for PowerPoint files (`.pptx`).
        - `XlsxIngestor`: A class-based ingestor for Microsoft Excel files (`.xlsx`).

- `models.py`: Defines the data structures and base classes for the ingestion process.
    - **`RawDoc` dataclass**: Represents a single piece of raw content extracted from a file before it's chunked. It contains:
        - `content: str`: The textual content.
        - `metadata: dict`: A dictionary of metadata about the content (e.g., source file, document type).
    - **`AbstractIngestor` (ABC)**: An abstract base class for creating ingestor classes. It defines an `ingest()` method that concrete ingestors must implement. This is useful for loaders that might produce multiple `RawDoc` instances from a single file (e.g., `PptxIngestor`, `XlsxIngestor`).
    - **`UnsupportedFileError` exception**: A custom exception raised when a file cannot be processed (e.g., corrupted, encrypted, or unsupported format).

- `manager.py`: Contains the `IngestionManager` class, which orchestrates the ingestion process.
    - **`IngestionManager` class**:
        - Its `ingest_path(path: str | pathlib.Path) -> List[RawDoc]` method takes a file or directory path.
        - It recursively searches for files (`rglob("*")`) within the given path.
        - For each file, it checks its suffix against the `LOADER_REGISTRY`.
        - If a loader is found, it invokes it. It handles both function-based loaders (e.g., `load_pdf`) and class-based ingestors (subclasses of `AbstractIngestor`, like `PptxIngestor` and `XlsxIngestor`).
        - For class-based ingestors, it instantiates the class and calls its `ingest()` method, which returns a list of `(text_segment, metadata)` tuples. Each tuple is then converted into a `RawDoc`.
        - For function-based loaders, it calls the function, expecting `(content, metadata)` to be returned, which is then wrapped in a `RawDoc`.
        - It populates `base_metadata` with `source_filepath` and `doc_type` (derived from the file extension) and merges it with metadata returned by the loader/ingestor.
        - It collects all `RawDoc` objects and returns them as a list.
        - Includes error handling for `UnsupportedFileError` and other exceptions during loading, logging warnings for problematic files.

- `csv.py`:
    - **`load_csv(file_path: str) -> tuple[str, dict]`**:
        - Loads a CSV (`.csv`) file.
        - Concatenates all rows into a single string, with cells joined by commas and rows by newlines.
        - Returns the full CSV text and a metadata dictionary: `{'doc_type': 'csv'}`.

- `docx_loader.py`:
    - **`load_docx(path: str | pathlib.Path) -> tuple[str, dict]`**:
        - Parses Microsoft Word (`.docx`) files using the `python-docx` library.
        - Extracts text from paragraphs and tables (tables are represented with " | " cell delimiters and structural markers).
        - Returns the extracted text and a metadata dictionary: `{"source": str(path), "content_type": "docx", "doc_type": "docx"}`.

- `email_loader.py`:
    - **`load_eml(path: str | Path) -> tuple[str, dict]`**:
        - Parses email (`.eml`) files using Python's built-in `email` module.
        - Prioritizes extracting the `text/plain` part of the email.
        - Returns the extracted plain text and a metadata dictionary: `{"source": str(path), "content_type": "email", "doc_type": "eml"}`.

- `pdf.py`:
    - **`load_pdf(path: str | Path) -> tuple[str, dict]`**:
        - Parses PDF (`.pdf`) files using the `pdfplumber` library.
        - Extracts text from each page and joins them with double newlines.
        - Raises `UnsupportedFileError` if the PDF is encrypted, corrupted, has no pages, or contains no extractable text.
        - Returns the extracted text and a metadata dictionary containing `source_path`, `title`, `author`, `created` (CreationDate), `modified` (ModDate), and `num_pages`.

- `pptx.py`:
    - **`PptxIngestor(AbstractIngestor)` class**:
        - Implements the `ingest(self, filepath: str) -> list[tuple[str, dict]]` method for PowerPoint (`.pptx`) files using the `python-pptx` library.
        - Extracts text from shapes on each slide and from presenter notes.
        - For each slide, it can produce separate text segments for slide content and presenter notes.
        - Returns a list of tuples, where each tuple is `(text_segment, metadata)`. The metadata includes `slide_number`, `type` ("slide_content" or "presenter_notes"), and `doc_type` ("pptx").
        - Raises `UnsupportedFileError` if the file is not a `.pptx` file or if errors occur during processing.

- `xlsx.py`:
    - **`XlsxIngestor(AbstractIngestor)` class**:
        - Implements the `ingest(self, filepath: str) -> list[tuple[str, dict]]` method for Microsoft Excel (`.xlsx`) files using the `openpyxl` library.
        - Reads data from each sheet, processing rows in chunks (default 50 rows per chunk).
        - For each chunk, it creates a text segment by joining cell values with tabs and rows with newlines.
        - Returns a list of `(text_segment, metadata)` tuples. The metadata for each segment includes `doc_type` ("xlsx"), `sheet_name`, `row_range` (e.g., "1-50"), and `source_filepath`.
        - Raises `UnsupportedFileError` if the file is not an `.xlsx` file or if errors occur during processing.

**Integration:**
The `IngestionManager` is the central component for ingesting documents. It relies on the `LOADER_REGISTRY` (defined in `__init__.py`) to select the appropriate loader or ingestor based on the file extension.
- **Function-based loaders** (like `load_pdf`, `load_docx`, `load_csv`, `load_eml`, `load_txt`) are directly called and are expected to return a tuple of `(content_string, metadata_dict)`.
- **Class-based ingestors** (like `PptxIngestor`, `XlsxIngestor`), which inherit from `AbstractIngestor`, are instantiated, and their `ingest` method is called. This method returns a list of `(content_string, metadata_dict)` tuples, allowing a single file to be processed into multiple `RawDoc` objects (e.g., one per slide note in PPTX, or one per chunk of rows in XLSX).

The `IngestionManager` standardizes the metadata by adding `source_filepath` and `doc_type` (derived from the file extension) and then merges it with the metadata provided by the specific loader/ingestor. The final output is a `List[RawDoc]`, where each `RawDoc` contains the text content and comprehensive metadata. This list is then typically passed to downstream components, such as those in `scripts/chunking/`, for further processing in the RAG pipeline.
