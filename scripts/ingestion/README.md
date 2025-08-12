# Ingestion Module

This module is responsible for the first stage of the RAG pipeline: ingesting raw files from a source directory and converting them into a standardized format. It can handle various file types, including documents, spreadsheets, presentations, and emails.

## Core Components

- **`manager.py`**: Contains the `IngestionManager`, the central orchestrator of the ingestion process. It recursively scans a given directory, identifies files with supported extensions, and dispatches them to the appropriate loader.
- **`models.py`**: Defines the core data structures for the ingestion process:
    - **`RawDoc`**: A dataclass that represents a single piece of content extracted from a file. It contains the `content` (text) and a `metadata` dictionary.
    - **`AbstractIngestor`**: An abstract base class that defines the common interface for all class-based ingestors.
    - **`UnsupportedFileError`**: A custom exception for handling unsupported or corrupted files.
- **`__init__.py`**: Exposes the `LOADER_REGISTRY`, a dictionary that maps file extensions (e.g., `.pdf`, `.docx`) to their corresponding loader functions or ingestor classes.

## Loaders and Ingestors

The ingestion module uses a combination of simple loader functions and more complex ingestor classes to handle different file formats.

### Function-Based Loaders

- **`csv.py` (`load_csv`)**: Reads a CSV file and concatenates all its rows into a single string.
- **`docx_loader.py` (`load_docx`)**: Extracts content from Microsoft Word documents. It processes paragraphs and tables, and also extracts any embedded images, saving them to a cache directory. It returns a separate `RawDoc` for each paragraph and table.
- **`email_loader.py` (`load_eml`)**: Parses `.eml` email files and extracts the plain text body.
- **`pdf.py` (`load_pdf`)**: Uses the `pdfplumber` library to extract text and images from each page of a PDF document. It creates a `RawDoc` for each page.

### Class-Based Ingestors

- **`pptx.py` (`PptxIngestor`)**: Ingests Microsoft PowerPoint presentations. It extracts the text and images from each slide, creating a `RawDoc` for the main slide content. It also extracts any presenter notes associated with a slide and creates a separate `RawDoc` for them.
- **`xlsx.py` (`XlsxIngestor`)**: Handles Microsoft Excel spreadsheets. It reads each sheet in the workbook and groups the rows into chunks of a predefined size, creating a `RawDoc` for each chunk.

## Workflow

The ingestion process is typically initiated by calling the `ingest_path` method of the `IngestionManager`. The workflow is as follows:

1. The `IngestionManager` recursively scans the specified input directory for files.
2. For each file, it checks the file extension against the `LOADER_REGISTRY` to find the appropriate loader or ingestor.
3. It invokes the loader/ingestor, which extracts the content and metadata from the file.
4. The loader/ingestor returns a list of `(content, metadata)` tuples, which the `IngestionManager` then converts into a list of `RawDoc` objects.
5. This list of `RawDoc` objects is then passed to the next stage of the pipeline: chunking.
