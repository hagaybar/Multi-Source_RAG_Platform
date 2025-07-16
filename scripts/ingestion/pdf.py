from pathlib import Path
from typing import List, Tuple

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException
from scripts.ingestion.models import UnsupportedFileError

from scripts.utils.image_utils import (
    infer_project_root,
    ensure_image_cache_dir,
    save_image_pillow,
    generate_image_filename
)
from PIL import Image

def load_pdf(path: str | Path) -> List[Tuple[str, dict]]:
    if not isinstance(path, Path):
        path = Path(path)

    try:
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                raise UnsupportedFileError(f"No pages found in PDF: {path}")

            project_root = infer_project_root(path)
            image_dir = ensure_image_cache_dir(project_root)
            doc_id = str(path)
            file_stem = path.stem

            base_doc_meta = {
                "source_path": str(path.resolve()),
                "doc_id": doc_id,
                "source_filepath": str(path),
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
                        try:
                            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                            cropped = page.crop(bbox).to_image(resolution=150)
                            pil_image = cropped.original

                            img_name = generate_image_filename(doc_id, page_number, img_idx)
                            img_path = image_dir / img_name

                            save_image_pillow(pil_image, img_path)

                            meta = base_meta.copy()
                            meta["image_path"] = str(img_path.relative_to(project_root))
                            segments.append((text or "[Image-only content]", meta))

                        except Exception as e:
                            print(f"[WARN] Failed to extract/save image on page {page_number}: {e}")
                            continue
                else:
                    if text:
                        segments.append((text, base_meta))

            if not segments or all(not seg[0].strip() for seg in segments):
                raise UnsupportedFileError(f"No extractable text or image found in PDF: {path}")

            return segments

    except FileNotFoundError:
        raise

    except PDFPasswordIncorrect as e:
        raise UnsupportedFileError(f"PDF {path} is encrypted and requires a password.") from e

    except PDFSyntaxError as e:
        raise UnsupportedFileError(f"Failed to parse PDF {path}, it might be corrupted: {e}") from e

    except PdfminerException as e:
        if isinstance(e.args[0], PDFPasswordIncorrect):
            raise UnsupportedFileError(f"PDF {path} is encrypted and requires a password.") from e
        elif isinstance(e.args[0], PDFSyntaxError):
            raise UnsupportedFileError(f"Failed to parse PDF {path}, it might be corrupted: {e.args[0]}") from e
        else:
            raise UnsupportedFileError(f"An unexpected PDF processing error occurred with {path}: {e.args[0]}") from e

    except UnsupportedFileError:
        raise

    except Exception as e:
        raise UnsupportedFileError(f"An unexpected error occurred while processing PDF {path}: {e}") from e
