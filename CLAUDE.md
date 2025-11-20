# Multi-Source RAG Platform - Architecture & Best Practices

**Document Version:** 1.0
**Last Updated:** 2025-01-19
**Purpose:** Reference guide for Claude Code when working on this project

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [Pipeline Data Flow](#pipeline-data-flow)
4. [File Formats & Conventions](#file-formats--conventions)
5. [Coding Patterns & Best Practices](#coding-patterns--best-practices)
6. [Adding New Document Types](#adding-new-document-types)
7. [Common Pitfalls](#common-pitfalls)

---

## Project Overview

### What This Project Does

A **Retrieval-Augmented Generation (RAG)** platform that:
- Ingests multi-format documents (PDF, DOCX, PPTX, emails, etc.)
- Chunks them into semantically meaningful pieces
- Embeds chunks using local or OpenAI embedders
- Indexes embeddings in FAISS vector database
- Retrieves relevant chunks for user queries
- Generates answers using LLM with retrieved context

### Key Technologies

- **Vector DB**: FAISS (IndexFlatL2)
- **Embeddings**: OpenAI `text-embedding-3-large` or local BGE models
- **LLM**: OpenAI `gpt-4o` or other compatible models
- **UI**: Streamlit
- **Logging**: Structured JSON logs with per-run tracking
- **File Formats**: TSV (chunks), JSONL (metadata), FAISS (vectors)

---

## Core Architecture

### Project Structure

```
Multi-Source_RAG_Platform/
├── scripts/
│   ├── api_clients/          # OpenAI, other API clients
│   ├── agents/               # LLM-based agents (image insight, etc.)
│   ├── chunking/             # Text chunking logic (chunker_v3.py)
│   ├── connectors/           # Data source connectors (Outlook, etc.)
│   ├── core/                 # Project management (ProjectManager)
│   ├── embeddings/           # Embedding & indexing (UnifiedEmbedder)
│   ├── ingestion/            # Document loaders (PDF, DOCX, email, etc.)
│   ├── pipeline/             # Pipeline orchestration (PipelineRunner)
│   ├── prompting/            # Prompt building
│   ├── retrieval/            # Retrieval strategies (RetrievalManager)
│   ├── ui/                   # Streamlit UI components
│   └── utils/                # Utilities (logging, email cleaning, etc.)
├── configs/
│   └── chunk_rules.yaml      # Chunking rules per doc_type
├── data/
│   └── projects/             # Per-project data directories
│       └── <project_name>/
│           ├── input/
│           │   ├── raw/      # Original uploaded files
│           │   ├── cache/    # Extracted images
│           │   ├── chunks_<doc_type>.tsv  # Chunked data
│           │   └── enriched/ # Image-enriched chunks
│           ├── output/
│           │   ├── faiss/    # FAISS indices per doc_type
│           │   └── metadata/ # JSONL metadata per doc_type
│           ├── logs/
│           │   ├── app/      # Application-level logs
│           │   └── runs/     # Per-run logs and artifacts
│           └── config.yml    # Project configuration
├── docs/                     # Documentation
├── tests/                    # Test suite
└── pyproject.toml            # Dependencies (Poetry)
```

---

## Pipeline Data Flow

### Complete Pipeline Steps

```
1. INGEST   → Load raw files → Create RawDoc objects in memory
2. CHUNK    → Split RawDocs → Create Chunk objects + TSV files
3. ENRICH   → (Optional) Add image descriptions to chunks
4. EMBED    → Embed chunks → Create FAISS indices + metadata JSONL
5. INDEX_IMAGES → (Optional) Index image descriptions
6. RETRIEVE → Search FAISS → Return top-K relevant chunks
7. ASK      → Build prompt → Call LLM → Return answer
```

### Detailed Data Flow

#### STEP 1: INGEST

**Input:**
- Files in `<project_root>/input/raw/`
- Formats: PDF, DOCX, PPTX, XLSX, CSV, TXT, EML, MSG, MBOX

**Processing:**
```python
# Located in: scripts/pipeline/runner.py - step_ingest()
# 1. IngestionManager scans input/raw/ recursively
# 2. For each file, lookup loader in LOADER_REGISTRY by file extension
# 3. Call loader: returns (content, metadata) or list[(content, metadata)]
# 4. Create RawDoc objects:
RawDoc(
    content="Full text content",
    metadata={
        "source_filepath": "/path/to/file.pdf",
        "doc_type": "pdf",  # From file extension
        "content_hash": "sha256_hash",  # For deduplication
        "image_paths": ["img1.png"],  # If applicable
        "paragraph_number": 1  # If applicable
    }
)
# 5. Deduplicate by SHA256(content + image_paths)
```

**Output:**
- **In-Memory**: `self.raw_docs: list[RawDoc]`
- **Files Created**: None

**Key Files:**
- `scripts/ingestion/manager.py` - IngestionManager class
- `scripts/ingestion/__init__.py` - LOADER_REGISTRY
- `scripts/ingestion/pdf.py` - PDF loader
- `scripts/ingestion/docx_loader.py` - DOCX loader
- `scripts/ingestion/email_loader.py` - EML, MSG, MBOX loaders
- `scripts/ingestion/pptx.py` - PPTX loader
- `scripts/ingestion/xlsx.py` - XLSX loader

---

#### STEP 2: CHUNK

**Input:**
- **In-Memory**: `self.raw_docs: list[RawDoc]` from step_ingest()

**Processing:**
```python
# Located in: scripts/pipeline/runner.py - step_chunk()
# For each RawDoc:
#   1. Call chunker_v3.split(content, metadata)
#   2. Chunker splits on blank lines (paragraphs)
#   3. Merges paragraphs based on ChunkRule for doc_type
#   4. Rules loaded from configs/chunk_rules.yaml
#   5. Token-aware merging (min/max tokens per chunk)
#   6. Image-containing chunks bypass minimum token threshold
```

**ChunkRule Schema** (from `configs/chunk_rules.yaml`):
```yaml
pdf:
  strategy: by_paragraph
  min_tokens: 50
  max_tokens: 300
  overlap: 20

outlook_eml:
  strategy: by_email_block
  min_tokens: 20
  max_tokens: 300
  overlap: 5
```

**Output:**
- **In-Memory**: `self.chunks: list[Chunk]`
```python
Chunk(
    id="uuid_hex_string",
    doc_id="source_filepath",
    text="Merged paragraph text...",
    token_count=145,
    meta={
        "doc_type": "pdf",
        "source_filepath": "...",
        "image_paths": ["img1.png"],
        "page_number": 3,
        "content_hash": "sha256_hash"
    }
)
```

- **Files Created**: `<project_root>/input/chunks_<doc_type>.tsv`
  - Examples: `chunks_pdf.tsv`, `chunks_docx.tsv`, `chunks_outlook_eml.tsv`
  - **TSV Schema**:
    ```
    chunk_id<TAB>doc_id<TAB>text<TAB>token_count<TAB>meta_json
    ```
  - **Example Row**:
    ```
    a1b2c3...<TAB>file.pdf<TAB>The paragraph text here<TAB>145<TAB>{"doc_type":"pdf",...}
    ```

**Key Files:**
- `scripts/chunking/chunker_v3.py` - Main chunking logic
- `scripts/chunking/models.py` - Chunk, ChunkRule data models
- `configs/chunk_rules.yaml` - Per-doc_type chunking rules

---

#### STEP 3: EMBED

**Input:**
- **Memory Preferred**: `self.chunks` if available
- **Disk Fallback**: Load `chunks_<doc_type>.tsv` files from `input/`

**Processing:**
```python
# Located in: scripts/embeddings/unified_embedder.py - run()
# 1. Group chunks by doc_type
# 2. Deduplicate: check existing FAISS metadata for content_hash
# 3. Embed text using:
#    - Local embedder (BGE, sentence-transformers, etc.) OR
#    - OpenAI async batch API (text-embedding-3-large, 3072 dims)
# 4. Add embeddings to per-doc_type FAISS index
# 5. Append chunk metadata to per-doc_type JSONL file
```

**Output:**
- **FAISS Indices**: `<project_root>/output/faiss/<doc_type>.faiss`
  - Format: FAISS IndexFlatL2 (float32 vectors)
  - Dimension: 384-1536 (local models) or 3072 (OpenAI)
  - Examples: `pdf.faiss`, `docx.faiss`, `outlook_eml.faiss`

- **Metadata JSONL**: `<project_root>/output/metadata/<doc_type>_metadata.jsonl`
  - One line per chunk:
    ```json
    {
      "id": "chunk_uuid",
      "doc_type": "pdf",
      "source_filepath": "/path/to/file.pdf",
      "text": "Full chunk text content...",
      "token_count": 145,
      "image_paths": ["img1.png"],
      "page_number": 3,
      "content_hash": "sha256_hash"
    }
    ```

**Key Files:**
- `scripts/embeddings/unified_embedder.py` - UnifiedEmbedder class
- `scripts/embeddings/image_indexer.py` - ImageIndexer (for image chunks)
- `scripts/api_clients/openai/batch_embedder.py` - OpenAI batch embedding

---

#### STEP 4: RETRIEVE

**Input:**
- **FAISS Indices**: `output/faiss/<doc_type>.faiss`
- **Metadata**: `output/metadata/<doc_type>_metadata.jsonl`
- **Query**: User question string

**Processing:**
```python
# Located in: scripts/retrieval/retrieval_manager.py
# 1. Embed query using same embedder as chunks
# 2. Search FAISS indices (default: late_fusion strategy)
# 3. Combine text + image search results
# 4. Return top-K chunks with similarity scores
# 5. Enrich results with metadata from JSONL
```

**Output:**
- **In-Memory**: `self.retrieved_chunks: list[Chunk]`
  - Each chunk has `meta["similarity"]` score
  - Sorted by relevance

**Key Files:**
- `scripts/retrieval/retrieval_manager.py` - RetrievalManager
- `scripts/retrieval/strategies.py` - Retrieval strategies

---

#### STEP 5: ASK

**Input:**
- **In-Memory**: `self.retrieved_chunks` + user query

**Processing:**
```python
# Located in: scripts/prompting/prompt_builder.py
# 1. Build prompt: query + chunks as context
# 2. Call LLM (default: gpt-4o, temp=0.4, max_tokens=400)
# 3. Return answer text
```

**Output:**
- **In-Memory**: `self.last_answer: str`
- **Artifacts**: Saved to `logs/runs/<run_id>/`
  - `prompt.txt` - Full prompt sent to LLM
  - `response.txt` - LLM's answer
  - `metadata.json` - Retrieval parameters
  - `chunks.jsonl` - Retrieved chunks

**Key Files:**
- `scripts/prompting/prompt_builder.py` - PromptBuilder
- `scripts/api_clients/openai/completer.py` - OpenAICompleter

---

## File Formats & Conventions

### Raw Document Files

**Location**: `<project_root>/input/raw/<extension>/`

**Organization by Extension**:
```
input/raw/
├── pdf/
│   ├── document1.pdf
│   └── document2.pdf
├── docx/
│   └── report.docx
├── eml/
│   ├── email1.eml
│   └── email2.eml
└── outlook_eml/
    └── emails.outlook_eml  # JSONL format
```

**File Naming**:
- Original filenames preserved for uploaded files
- Generated files use descriptive names (e.g., `emails.outlook_eml`)

---

### Chunk TSV Files

**Location**: `<project_root>/input/chunks_<doc_type>.tsv`

**Format**: Tab-Separated Values (UTF-8 encoding)

**Schema**:
```
chunk_id<TAB>doc_id<TAB>text<TAB>token_count<TAB>meta_json
```

**Example**:
```tsv
a1b2c3d4e5f6	/path/to/file.pdf	The paragraph text here	145	{"doc_type":"pdf","source_filepath":"/path/to/file.pdf","page_number":3}
f6e5d4c3b2a1	/path/to/file.pdf	Another chunk of text	98	{"doc_type":"pdf","source_filepath":"/path/to/file.pdf","page_number":3}
```

**Notes**:
- `meta_json` is a JSON string (serialized dict)
- Each doc_type has its own TSV file (e.g., `chunks_pdf.tsv`, `chunks_outlook_eml.tsv`)

---

### Metadata JSONL Files

**Location**: `<project_root>/output/metadata/<doc_type>_metadata.jsonl`

**Format**: JSON Lines (one JSON object per line, UTF-8 encoding)

**Schema per Line**:
```json
{
  "id": "chunk_uuid",
  "doc_type": "pdf",
  "source_filepath": "/path/to/file.pdf",
  "text": "Full chunk text content...",
  "token_count": 145,
  "image_paths": ["cache/images/file_page3_img1.png"],
  "page_number": 3,
  "image_summaries": [],
  "content_hash": "sha256_hash"
}
```

**Notes**:
- One file per doc_type (e.g., `pdf_metadata.jsonl`, `outlook_eml_metadata.jsonl`)
- Paired with FAISS index (same doc_type)
- Used for retrieval to fetch full chunk text and metadata

---

### FAISS Index Files

**Location**: `<project_root>/output/faiss/<doc_type>.faiss`

**Format**: FAISS binary format (IndexFlatL2)

**Examples**:
- `pdf.faiss` - PDF document embeddings
- `docx.faiss` - DOCX document embeddings
- `outlook_eml.faiss` - Email embeddings

**Properties**:
- Vector dimension: 384-3072 (depends on embedder)
- Distance metric: L2 (Euclidean distance)
- Supports incremental updates (append new vectors)

**Paired Files**:
- Each `.faiss` file is paired with `metadata/<doc_type>_metadata.jsonl`
- Index position N maps to line N in JSONL file

---

### Image Files

**Location**: `<project_root>/input/cache/images/`

**Naming Convention**: `{doc_stem}_page{N}_img{M}.png`

**Examples**:
- `report_page1_img1.png` - First image from page 1 of report.pdf
- `presentation_slide3_img2.png` - Second image from slide 3 of presentation.pptx

**Storage**:
- Extracted during ingestion (PDF, DOCX, PPTX)
- Saved as PNG format (via PIL)
- Referenced in chunk metadata via `image_paths` array

---

## Coding Patterns & Best Practices

### 1. Adding New Document Loaders

**Pattern to Follow**:

```python
# File: scripts/ingestion/<format>_loader.py

from pathlib import Path
from typing import List, Tuple

def load_<format>(path: str | Path) -> Tuple[str, dict] | List[Tuple[str, dict]]:
    """
    Load <format> file and return content + metadata.

    For single-document formats (PDF, DOCX):
        Returns: (content, metadata)

    For multi-document formats (MBOX, JSONL):
        Returns: [(content1, metadata1), (content2, metadata2), ...]

    Args:
        path: Path to the file

    Returns:
        Tuple or List of tuples containing (text_content, metadata_dict)

    Metadata Requirements:
        - doc_type: str (e.g., "pdf", "outlook_eml")
        - source: str (file path)
        - content_type: str (e.g., "document", "email")
        - Additional fields as needed (subject, sender, page_number, etc.)
    """
    path = Path(path)

    # Read and parse file
    # ...

    # Build metadata
    metadata = {
        "source": str(path),
        "doc_type": "format_name",  # Must match chunking rule key
        "content_type": "document",  # or "email", "spreadsheet", etc.
        # Add format-specific fields
    }

    return (content, metadata)  # or list of tuples

# Register in scripts/ingestion/__init__.py
LOADER_REGISTRY = {
    ".format": load_<format>,
    # ...
}
```

**Key Requirements**:
1. **Return Type**:
   - Single document: `(str, dict)`
   - Multiple documents (like MBOX): `List[(str, dict)]`

2. **Metadata Fields**:
   - `doc_type` (REQUIRED): Must match a key in `configs/chunk_rules.yaml`
   - `source` or `source_filepath`: Path to original file
   - `content_type`: General category (document, email, etc.)
   - Additional fields as appropriate

3. **Text Cleaning**:
   - For emails: Use `scripts/utils/email_utils.clean_email_text()`
   - For HTML: Strip tags appropriately
   - Preserve paragraph structure (blank lines separate paragraphs)

4. **Error Handling**:
   - Raise descriptive exceptions (ImportError, RuntimeError, ValueError)
   - Include file path in error messages

---

### 2. Adding Chunking Rules

**File**: `configs/chunk_rules.yaml`

**Pattern**:
```yaml
<doc_type>:
  strategy: by_paragraph  # or by_email_block, by_page, etc.
  min_tokens: 50          # Minimum tokens per chunk
  max_tokens: 300         # Maximum tokens per chunk
  overlap: 20             # Token overlap between chunks
```

**Available Strategies**:
- `by_paragraph`: Split on blank lines, merge based on token limits
- `by_email_block`: Similar to paragraph but email-specific
- `by_page`: Keep page boundaries (for PDFs)

**Key Considerations**:
- `doc_type` must match the `doc_type` in loader metadata
- Chunks with `image_paths` bypass `min_tokens` requirement
- Overlap creates context continuity between chunks

---

### 3. Multi-Document File Formats (Like MBOX)

**Use Case**: When one file contains multiple documents (emails, records, etc.)

**Pattern**:
```python
def load_multi_doc_format(path: str | Path) -> List[Tuple[str, dict]]:
    """Load file containing multiple documents."""
    path = Path(path)
    documents = []

    # Open and parse file
    with open(path, 'r', encoding='utf-8') as f:
        for i, item in enumerate(parse_items(f)):
            # Extract content
            content = item.get_text()

            # Build metadata per item
            metadata = {
                "source": str(path),
                "doc_type": "format_name",
                "content_type": "email",  # or appropriate type
                "message_index": i,  # Sequential index
                # Add item-specific fields
                "subject": item.subject,
                "sender": item.sender,
                "date": item.date,
            }

            if content.strip():  # Only add non-empty items
                documents.append((content, metadata))

    return documents
```

**Examples in Codebase**:
- `scripts/ingestion/email_loader.py` - `load_mbox()` (mailbox format)
- `scripts/ingestion/email_loader.py` - `load_pst()` (Outlook PST format)

**Advantages**:
- Single file I/O operation
- Efficient for large datasets
- Standard pattern for batch document formats

---

### 4. Logging Best Practices

**Structured Logging**:
```python
from scripts.utils.logger import LoggerManager

# Get logger with subsystem name
logger = LoggerManager.get_logger("subsystem_name")

# Log with structured data
logger.info("Action description", extra={
    "action": "action_name",
    "param1": value1,
    "param2": value2
})

# Log errors with exception info
logger.error("Error description", extra={
    "action": "action_name",
    "error": str(e)
}, exc_info=True)
```

**Log Locations**:
- **App-level**: `logs/app/<subsystem>.log` (JSON format)
- **Per-run**: `logs/runs/<run_id>/app.log` (JSON format)
- **Artifacts**: `logs/runs/<run_id>/` (prompt.txt, response.txt, chunks.jsonl)

---

### 5. Image Handling

**Extraction**:
```python
from scripts.utils.image_utils import (
    save_image_pillow,
    generate_image_filename,
    record_image_metadata
)

# During ingestion:
img_filename = generate_image_filename(doc_id, page_number, img_index)
img_path = image_dir / img_filename
save_image_pillow(pil_image, img_path)

# Add to chunk metadata
record_image_metadata(chunk_metadata, img_path, project_root)
# This adds: metadata["image_paths"] = ["cache/images/file_page1_img1.png"]
```

**Image Directory**: `<project_root>/input/cache/images/`

**Image Metadata in Chunks**:
```python
chunk.meta = {
    "image_paths": ["cache/images/report_page1_img1.png"],
    # ... other metadata
}
```

**Image Enrichment** (Optional):
```python
# step_enrich() adds:
chunk.meta["image_summaries"] = [
    {
        "image_path": "cache/images/report_page1_img1.png",
        "description": "AI-generated description of the image"
    }
]
```

---

## Adding New Document Types

### Step-by-Step Guide

**Example**: Adding support for Outlook emails

#### 1. Create Connector/Loader

**File**: `scripts/ingestion/<format>_loader.py` or `scripts/connectors/<source>_connector.py`

```python
def load_outlook_emails(path: str | Path) -> List[Tuple[str, dict]]:
    """
    Load Outlook emails from JSONL file.

    Expected file format (JSONL):
    {"content": "email body", "metadata": {...}}
    {"content": "email body", "metadata": {...}}
    """
    import json
    path = Path(path)
    emails = []

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            content = data.get("content", "")
            metadata = data.get("metadata", {})

            # Ensure doc_type is set
            metadata["doc_type"] = "outlook_eml"
            metadata["message_index"] = i

            if content.strip():
                emails.append((content, metadata))

    return emails
```

#### 2. Register Loader

**File**: `scripts/ingestion/__init__.py`

```python
from .outlook_loader import load_outlook_emails

LOADER_REGISTRY = {
    # ... existing loaders
    ".outlook_eml": load_outlook_emails,  # Custom extension
}
```

#### 3. Add Chunking Rule

**File**: `configs/chunk_rules.yaml`

```yaml
outlook_eml:
  strategy: by_email_block
  min_tokens: 20
  max_tokens: 300
  overlap: 5
```

#### 4. Create Extraction Logic (if needed)

If extracting from external source (like Outlook client):

**File**: `scripts/connectors/outlook_connector.py`

```python
class OutlookConnector:
    def extract_emails(self) -> List[Tuple[str, dict]]:
        """Extract emails from Outlook and return in loader format."""
        emails = []
        # ... connect to Outlook, extract emails
        for email in outlook_emails:
            content = clean_email_text(email.body)
            metadata = {
                "doc_type": "outlook_eml",
                "content_type": "email",
                "subject": email.subject,
                "sender": email.sender,
                "date": email.date,
                # ...
            }
            emails.append((content, metadata))
        return emails

    def save_to_file(self, emails: List[Tuple[str, dict]], output_path: Path):
        """Save extracted emails as JSONL file."""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            for content, metadata in emails:
                json.dump({
                    "content": content,
                    "metadata": metadata
                }, f, ensure_ascii=False)
                f.write('\n')
```

#### 5. Create UI Components (if needed)

**File**: `scripts/ui/ui_<source>_manager.py`

```python
def render_extraction_ui(project_path: Path):
    """UI for extracting data from source."""
    if st.button("Extract Emails"):
        connector = OutlookConnector(config)
        emails = connector.extract_emails()

        # Save to raw directory
        raw_dir = project_path / "input" / "raw" / "outlook_eml"
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_file = raw_dir / "emails.outlook_eml"

        connector.save_to_file(emails, output_file)
        st.success(f"Saved {len(emails)} emails to {output_file}")
```

#### 6. Test the Flow

```bash
# 1. Extract emails (if applicable)
# 2. Run pipeline
ingest → chunk → embed

# 3. Verify files created:
# - input/raw/outlook_eml/emails.outlook_eml
# - input/chunks_outlook_eml.tsv
# - output/faiss/outlook_eml.faiss
# - output/metadata/outlook_eml_metadata.jsonl
```

---

## Common Pitfalls

### 1. ❌ Saving Each Document as Separate File

**Wrong**:
```python
# DON'T: Create one file per email
for i, email in enumerate(emails):
    file_path = raw_dir / f"email_{i:03d}.json"
    with open(file_path, 'w') as f:
        json.dump({"content": email.content, "metadata": email.metadata}, f)
```

**Right**:
```python
# DO: Save all emails in one JSONL file (like MBOX)
file_path = raw_dir / "emails.outlook_eml"
with open(file_path, 'w', encoding='utf-8') as f:
    for email in emails:
        json.dump({"content": email.content, "metadata": email.metadata}, f)
        f.write('\n')  # JSONL format
```

**Why**:
- Reduces I/O operations (1 file vs 100 files = 4 I/O ops vs 400 I/O ops)
- Follows existing pattern (MBOX, PST)
- Easier to manage and backup

---

### 2. ❌ Missing `doc_type` in Metadata

**Wrong**:
```python
metadata = {
    "source": str(path),
    "subject": "Email subject"
    # Missing doc_type!
}
```

**Right**:
```python
metadata = {
    "source": str(path),
    "doc_type": "outlook_eml",  # REQUIRED - matches chunk rule
    "subject": "Email subject"
}
```

**Why**: Chunking step looks up rules by `metadata["doc_type"]` - will fail without it.

---

### 3. ❌ Not Following Loader Return Type Convention

**Wrong**:
```python
def load_custom(path):
    # Returns just content, no metadata
    return file_content
```

**Right**:
```python
def load_custom(path):
    # Single document: return tuple
    return (content, metadata)

    # OR for multi-document:
    # return [(content1, meta1), (content2, meta2), ...]
```

**Why**: IngestionManager expects specific return types - breaks the pipeline otherwise.

---

### 4. ❌ Storing Data in Session State Instead of Files

**Wrong**:
```python
# DON'T: Store in Streamlit session state
st.session_state.extracted_emails = emails
# Pipeline can't access session state!
```

**Right**:
```python
# DO: Save to disk in input/raw/
raw_dir = project_path / "input" / "raw" / "outlook_eml"
raw_dir.mkdir(parents=True, exist_ok=True)
save_to_file(emails, raw_dir / "emails.outlook_eml")
# Now pipeline can load via step_ingest()
```

**Why**: Pipeline is file-based - reads from `input/raw/` directory.

---

### 5. ❌ Not Using Structured Logging

**Wrong**:
```python
print(f"Processed {count} items")  # Lost in output
```

**Right**:
```python
logger.info("Items processed", extra={
    "action": "process_items",
    "count": count,
    "source": "outlook"
})
```

**Why**: Structured logs are searchable, parseable, and go to proper log files.

---

## Quick Reference

### File Extension → Loader Mapping

| Extension | Loader | Returns | Location |
|-----------|--------|---------|----------|
| `.pdf` | `load_pdf()` | `list[(str, dict)]` | `ingestion/pdf.py` |
| `.docx` | `load_docx()` | `list[(str, dict)]` | `ingestion/docx_loader.py` |
| `.pptx` | `PptxIngestor` | `list[(str, dict)]` | `ingestion/pptx.py` |
| `.xlsx` | `XlsxIngestor` | `list[(str, dict)]` | `ingestion/xlsx.py` |
| `.txt` | `load_txt()` | `(str, dict)` | `ingestion/__init__.py` |
| `.csv` | `load_csv()` | `(str, dict)` | `ingestion/csv.py` |
| `.eml` | `load_eml()` | `(str, dict)` | `ingestion/email_loader.py` |
| `.msg` | `load_msg()` | `(str, dict)` | `ingestion/email_loader.py` |
| `.mbox` | `load_mbox()` | `list[(str, dict)]` | `ingestion/email_loader.py` |

### Pipeline Step Dependencies

| Step | Requires | Produces | Optional |
|------|----------|----------|----------|
| ingest | Raw files | `self.raw_docs` | No |
| chunk | `self.raw_docs` | `self.chunks` + TSV files | No |
| enrich | `self.chunks` or TSV files | Enriched chunks + TSV | Yes |
| embed | `self.chunks` or TSV files | FAISS + JSONL | No |
| index_images | Enriched TSV files | Image FAISS + JSONL | Yes |
| retrieve | FAISS + JSONL + query | `self.retrieved_chunks` | No |
| ask | `self.retrieved_chunks` + query | `self.last_answer` | No |

---

## Version History

- **v1.0** (2025-01-19): Initial documentation created during Outlook integration review

---

**For Questions or Updates:**
- Review recent commits for implementation examples
- Check `docs/` directory for feature-specific documentation
- Analyze existing loaders in `scripts/ingestion/` as templates

---

## Development Workflow

### Setup
- **Environment**: `poetry install`
- **Activate**: `poetry shell`

### Testing
- **Run Tests**: `pytest` (or run specific test scripts if needed)

### Running the App
- **Streamlit UI**: `streamlit run scripts/ui/ui.py`
- **CLI App**: `python -m app.cli` (or `poetry run app`)
