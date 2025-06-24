# Scripts Folder

The `scripts` folder contains Python scripts that implement the core logic of the project, particularly around document processing, embedding, and retrieval for a Retrieval Augmented Generation (RAG) system.

- `__init__.py`: Marks the `scripts` folder as a Python package.

The folder is organized into several subdirectories, each responsible for a specific aspect of the processing pipeline:

- **`agents/`**:
    - Contains scripts related to AI agents or agentic behavior within the RAG system.
    - `__init__.py`: Marks the `agents` folder as a Python package.
    - For more details, see `scripts/agents/README.md`.

- **`api_clients/`**:
    - Houses clients for interacting with external APIs, such as OpenAI.
    - **`openai/`**:
        - `batch_embedder.py`: Implements `BatchEmbedder` for submitting large embedding jobs to OpenAI's asynchronous `/v1/batches` API. It handles JSONL file preparation, batch submission, status polling, result downloading, and parsing.

- **`chunking/`**:
    - Contains modules for splitting documents into smaller, manageable chunks. This is a crucial step for RAG.
    - `__init__.py`: Marks the `chunking` folder as a Python package.
    - `models.py`: Defines `Chunk` and `Doc` dataclasses for representing text chunks and documents.
    - `rules_v3.py`: Defines `ChunkRule` dataclass and functions (`get_rule`, `get_all_rules`) to load and manage chunking rules from `configs/chunk_rules.yaml`. These rules specify strategy, token limits, and overlap.
    - `chunker_v3.py`: Implements the main `split` function for chunking text based on rules from `rules_v3.py`. It supports various strategies (e.g., "by_paragraph", "by_slide", "by_email_block"), uses `spacy` for sentence splitting in emails, and handles merging small chunks and overlaps.
    - For more details, see `scripts/chunking/README.md`.

- **`core/`**:
    - Contains core project management and configuration scripts.
    - `__init__.py`: Contains an older or alternative `ProjectManager` class definition.
    - `project_manager.py`: Defines the primary `ProjectManager` class, responsible for managing the RAG project workspace, including paths to configuration (`config.yml`), input/output directories, logs, FAISS indexes, and metadata files. It ensures these directories exist.
    - For more details, see `scripts/core/README.md`.

- **`embeddings/`**:
    - Contains scripts for generating and managing text embeddings.
    - `__init__.py`: Marks the `embeddings` folder as a Python package.
    - `base.py`: Defines an abstract base class `BaseEmbedder` with an `encode` method.
    - `bge_embedder.py`: Implements `BGEEmbedder`, a concrete embedder using SentenceTransformers (e.g., "BAAI/bge-large-en").
    - `litellm_embedder.py`: Implements `LiteLLMEmbedder` for generating embeddings via LiteLLM-compatible APIs (OpenAI, Ollama, etc.), using HTTP requests.
    - `embedder_registry.py`: Provides `get_embedder` function to fetch an embedder instance based on project configuration (e.g., "local" or "litellm").
    - `unified_embedder.py`: Implements `UnifiedEmbedder`, a comprehensive class for embedding chunks. It supports:
        - Deduplication of chunks based on content hashes.
        - Batch embedding using local models or OpenAI's async batch API (via `BatchEmbedder`).
        - Grouping chunks by document type.
        - Storing embeddings in FAISS indexes and metadata in JSONL files, organized by document type.
        - Loading chunks from TSV files.
    - For more details, see `scripts/embeddings/README.md`.

- **`index/`**:
    - Intended for scripts related to creating, managing, and querying an index of document embeddings.
    - `__init__.py`: Marks the `index` folder as a Python package.
    - For more details, see `scripts/index/README.md`.

- **`ingestion/`**:
    - Contains modules for loading and parsing various document formats.
    - `__init__.py`: Initializes a `LOADER_REGISTRY` mapping file extensions (e.g., ".pdf", ".docx", ".txt", ".xlsx") to their corresponding loader functions or classes.
    - `models.py`: Defines `RawDoc` dataclass (for content before chunking), `AbstractIngestor` base class, and `UnsupportedFileError` exception.
    - `manager.py`: Defines `IngestionManager` which orchestrates document ingestion. It recursively searches a path, uses `LOADER_REGISTRY` to find the appropriate loader for each file, and returns a list of `RawDoc` objects.
    - `csv.py`: `load_csv` function to load CSV content as a single string.
    - `docx_loader.py`: `load_docx` function to extract text from `.docx` files, including from tables, using `python-docx`.
    - `email_loader.py`: `load_eml` function to parse `.eml` files and extract plain text content.
    - `pdf.py`: `load_pdf` function to extract text from `.pdf` files using `pdfplumber`, handling encrypted or corrupted files.
    - `pptx.py`: `PptxIngestor` class (subclass of `AbstractIngestor`) to extract text from `.pptx` slides and presenter notes using `python-pptx`.
    - `xlsx.py`: `XlsxIngestor` class (subclass of `AbstractIngestor`) to extract data from `.xlsx` files, grouping rows from each sheet into text chunks using `openpyxl`.
    - For more details, see `scripts/ingestion/README.md`.

- **`prompting/`**:
    - Intended for scripts related to constructing prompts for the language model in the RAG system.
    - `__init__.py`: Marks the `prompting` folder as a Python package.

- **`retrieval/`**:
    - Contains scripts for retrieving relevant chunks from the index based on a query.
    - `__init__.py`: Marks the `retrieval` folder as a Python package.
    - `base.py`: Defines `BaseRetriever` abstract class and `FaissRetriever` for searching in a FAISS index and its associated metadata. It uses a shared embedder (from `scripts.api_clients.embedder`, though this path might need checking as `get_embedder` is in `scripts.embeddings.embedder_registry`) to encode queries.
    - `retrieval_manager.py`: Implements `RetrievalManager` which loads retrievers (currently `FaissRetriever`) for different document types and applies retrieval strategies (defined in `scripts.retrieval.strategies`) like "late_fusion".
    - For more details, see `scripts/retrieval/README_retrieval.md`.

- **`utils/`**:
    - Contains utility scripts and helper functions used across the project.
    - `__init__.py`: Marks the `utils` folder as a Python package.
    - `chunk_utils.py`: Provides `deduplicate_chunks` (based on content hashes) and `load_chunks` (from TSV files).
    - `config_loader.py`: Defines `ConfigLoader` for loading and accessing YAML configuration files with support for dot notation.
    - `create_demo_pptx.py`: A script to generate a demo `.pptx` file for testing.
    - `email_utils.py`: `clean_email_text` function to remove quoted lines, reply blocks, and signatures from email text.
    - `logger.py`: Implements `LoggerManager` for creating configured `logging.Logger` instances with console/file output, JSON/text formatting, and optional color. Also includes `JsonLogFormatter`.
    - `msg2email.py`: `msg_to_eml` function to convert Outlook `.msg` files to `.eml` format using `extract_msg`.

These scripts work together to form a pipeline: documents are ingested, chunked, converted to embeddings, indexed, and then retrieved to augment prompts for a language model.
