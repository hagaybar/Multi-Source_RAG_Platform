from pathlib import Path
import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException # Corrected import
from scripts.ingestion.models import UnsupportedFileError # RawDoc is not directly returned
# No hashlib needed as UID is not part of RawDoc

def load_pdf(path: str | Path) -> tuple[str, dict]: # Corrected return type
    if not isinstance(path, Path):
        path = Path(path)

    try:
        # Attempt to open the PDF. This can raise FileNotFoundError or PdfminerException (wrapping others).
        with pdfplumber.open(path) as pdf:
            # These operations might raise PDFPasswordIncorrect or PDFSyntaxError directly,
            # or other pdfminer errors if not caught by pdfplumber.open's wrapper.
            _ = len(pdf.pages)
            _ = pdf.metadata

            if not pdf.pages: # Check if there are any pages
                raise UnsupportedFileError(f"No pages found in PDF: {path}")

            page_texts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text: # Only append if text was extracted
                    page_texts.append(text.strip()) # Also strip whitespace from individual page texts

            if not page_texts: # No text extracted from any page
                raise UnsupportedFileError(f"No extractable text found in PDF: {path}")

            full_text = "\n\n".join(page_texts)

            # pdfplumber's metadata keys can be 'Title', 'Author', 'CreationDate', 'ModDate'
            # We use .get() to gracefully handle missing keys, returning None.
            # The dates are typically strings and are stored as such.
            metadata = {
                "source_path": str(path.resolve()), # Store absolute path as string
                "title": pdf.metadata.get("Title"),
                "author": pdf.metadata.get("Author"),
                "created": pdf.metadata.get("CreationDate"), # Store as string or None
                "modified": pdf.metadata.get("ModDate"),   # Store as string or None
                "num_pages": len(pdf.pages),
            }

            return full_text, metadata

    except FileNotFoundError as e: # If the file itself is not found
        raise # The test expects FileNotFoundError to be propagated.

    except PDFPasswordIncorrect as e: # If specifically password protected (potentially from pdf.pages etc.)
        raise UnsupportedFileError(f"PDF {path} is encrypted and requires a password.") from e

    except PDFSyntaxError as e: # If PDF is corrupted (potentially from pdf.pages etc.)
        raise UnsupportedFileError(f"Failed to parse PDF {path}, it might be corrupted: {e}") from e

    except PdfminerException as e: # Catch exceptions wrapped by pdfplumber.open()
        # Check the wrapped exception type
        if isinstance(e.args[0], PDFPasswordIncorrect):
            raise UnsupportedFileError(f"PDF {path} is encrypted and requires a password.") from e
        elif isinstance(e.args[0], PDFSyntaxError): # This covers general parsing errors
            raise UnsupportedFileError(f"Failed to parse PDF {path}, it might be corrupted: {e.args[0]}") from e
        else: # Other errors from pdfminer wrapped by PdfminerException
            raise UnsupportedFileError(f"An unexpected PDF processing error occurred with {path}: {e.args[0]}") from e

    except UnsupportedFileError: # If we raised it ourselves (e.g. no text, no pages)
        raise

    except Exception as e: # For any other truly unexpected errors
        raise UnsupportedFileError(f"An unexpected error occurred while processing PDF {path}: {e}") from e
