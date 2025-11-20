# Ingestion Loaders

This document describes the available data loaders for ingesting content.

## Email (`.eml`)

The email loader processes `.eml` files.

**Function:** `scripts.ingestion.email_loader.load_eml(path: str | pathlib.Path) -> tuple[str, dict]`

**Description:**
Extracts the plain text body from an email file.
Returns a tuple containing the text content and a metadata dictionary:
`{"source": "path/to/file.eml", "content_type": "email"}`.

**Usage:**
```python
from pathlib import Path
from scripts.ingestion.email_loader import load_eml

eml_path = Path("path/to/your/email.eml")
text, metadata = load_eml(eml_path)
print(f"Content: {text[:100]}...")
print(f"Metadata: {metadata}")
```

## DOCX (`.docx`)

The DOCX loader processes `.docx` files (Microsoft Word documents).

**Function:** `scripts.ingestion.docx_loader.load_docx(path: str | pathlib.Path) -> tuple[str, dict]`

**Description:**
Extracts plain text content from a DOCX file.
- Includes all paragraph text.
- Includes text inside tables in row-major order.
- Ignores images, comments, footnotes, and endnotes.
- Collapses consecutive whitespace to a single space and trims leading/trailing whitespace.

Returns a tuple containing the extracted text and a metadata dictionary:
`{"source": "path/to/file.docx", "content_type": "docx"}`.

**Usage:**
```python
from pathlib import Path
from scripts.ingestion.docx_loader import load_docx

docx_path = Path("path/to/your/document.docx")
text, metadata = load_docx(docx_path)
print(f"Content: {text[:100]}...")
print(f"Metadata: {metadata}")
```
