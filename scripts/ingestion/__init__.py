from .csv import load_csv
from .docx_loader import load_docx
from .email_loader import load_eml, load_msg, load_mbox
from .xlsx import XlsxIngestor  # Import XlsxIngestor class
from .pdf import load_pdf  # Add this import
from .pptx import PptxIngestor  # Import PptxIngestor


# Simple loader for .txt files
def load_txt(filepath: str) -> tuple[str, dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Basic metadata, IngestionManager will add/override doc_type and source_filepath
    return content, {}


# Loader for Outlook-extracted emails saved as JSONL
def load_outlook_eml_json(filepath: str) -> list[tuple[str, dict]]:
    """
    Load Outlook emails from JSONL file (one email per line).

    This follows the MBOX pattern: one file contains multiple emails.

    Expected JSONL format (one JSON object per line):
    {"content": "email body 1", "metadata": {...}}
    {"content": "email body 2", "metadata": {...}}
    {"content": "email body 3", "metadata": {...}}

    Returns:
        List of (content, metadata) tuples, one per email

    Example:
        >>> emails = load_outlook_eml_json("emails.outlook_eml")
        >>> len(emails)  # Number of emails in file
        150
        >>> content, meta = emails[0]
        >>> meta["subject"]
        'Meeting notes'
    """
    import json
    emails = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                data = json.loads(line)
                content = data.get("content", "")
                metadata = data.get("metadata", {})

                # Ensure doc_type is set (required for chunking)
                if "doc_type" not in metadata:
                    metadata["doc_type"] = "outlook_eml"

                # Add message index for tracking
                metadata["message_index"] = i

                if content.strip():  # Only add non-empty emails
                    emails.append((content, metadata))

            except json.JSONDecodeError as e:
                # Log error but continue processing other emails
                print(f"[WARN] Failed to parse line {i+1} in {filepath}: {e}")
                continue

    return emails


LOADER_REGISTRY = {
    ".txt": load_txt,  # Added .txt loader
    ".csv": load_csv,
    ".docx": load_docx,
    ".eml": load_eml,
    ".msg": load_msg,  # Outlook message files
    ".mbox": load_mbox,  # Mailbox format (bulk email)
    ".pdf": load_pdf,  # Add this mapping
    ".pptx": PptxIngestor,  # Map .pptx to PptxIngestor class
    ".xlsx": XlsxIngestor,  # Map .xlsx to XlsxIngestor class
    ".outlook_eml": load_outlook_eml_json,  # Outlook emails saved as JSON
}
