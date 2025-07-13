from pathlib import Path
from typing import List, Tuple

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException # Corrected import
from scripts.ingestion.models import UnsupportedFileError # RawDoc is not directly returned
# No hashlib needed as UID is not part of RawDoc

def _infer_project_root(file_path: Path) -> Path:
    parts = file_path.resolve().parts
    if "projects" in parts:
        idx = parts.index("projects")
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
    return file_path.parent


def _save_image(page, bbox, project_root: Path, filename: str) -> str:
    rel_dir = Path("input") / "cache" / "images"
    out_path = project_root / rel_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.crop(bbox).to_image(resolution=150).save(out_path, format="PNG")
    return str(rel_dir / filename)


def load_pdf(path: str | Path) -> List[Tuple[str, dict]]: # Corrected return type
    if not isinstance(path, Path):
        path = Path(path)

    try:
        with pdfplumber.open(path) as pdf:
            _ = len(pdf.pages)
            _ = pdf.metadata

            if not pdf.pages:
                raise UnsupportedFileError(f"No pages found in PDF: {path}")

            project_root = _infer_project_root(path)
            file_stem = path.stem

            base_doc_meta = {
                "source_path": str(path.resolve()),
                "title": pdf.metadata.get("Title"),
                "author": pdf.metadata.get("Author"),
                "created": pdf.metadata.get("CreationDate"),
                "modified": pdf.metadata.get("ModDate"),
                "num_pages": len(pdf.pages),
            }

            segments: List[Tuple[str, dict]] = []

            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                base_meta = {
                    **base_doc_meta,
                    "doc_type": "pdf",
                    "page_number": page_number,
                }

                images = page.images
                if images:
                    for img_idx, img in enumerate(images, start=1):
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        img_name = f"{file_stem}_page{page_number}_img{img_idx}.png"
                        rel_path = _save_image(page, bbox, project_root, img_name)
                        meta = base_meta.copy()
                        meta["image_path"] = rel_path
                        segments.append((text, meta))
                else:
                    segments.append((text, base_meta))

            if not segments or all(not seg[0].strip() for seg in segments):
                raise UnsupportedFileError(f"No extractable text found in PDF: {path}")

            return segments

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
