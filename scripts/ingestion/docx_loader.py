from __future__ import annotations

import pathlib
from pathlib import Path
from typing import List, Tuple

from docx import Document
from docx.oxml.ns import qn

from scripts.utils.image_utils import (
    infer_project_root,
    ensure_image_cache_dir,
    save_image_blob,
    generate_image_filename
)


def load_docx(path: str | pathlib.Path) -> List[Tuple[str, dict]]:
    """Extract text and images from a .docx file.

    Returns a list of ``(text, metadata)`` tuples where ``metadata`` may contain
    ``image_path`` if an image was found in that paragraph.
    """
    if not isinstance(path, Path):
        path = Path(path)

    document = Document(path)
    project_root = infer_project_root(path)
    image_dir = ensure_image_cache_dir(project_root)

    segments: List[Tuple[str, dict]] = []
    file_stem = path.stem
    doc_id = str(path)

    for para_idx, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()
        base_meta = {
            "doc_type": "docx",
            "paragraph_number": para_idx,
            "source_filepath": str(path),
            "doc_id": doc_id,
        }

        img_count = 0
        image_found = False

        for run in paragraph.runs:
            blips = run._element.xpath(".//a:blip")
            for blip in blips:
                rId = blip.get(qn("r:embed"))
                image_part = document.part.related_parts.get(rId)
                if not image_part:
                    continue

                img_name = generate_image_filename(
                    doc_id=doc_id,
                    page_number=para_idx,
                    img_index=img_count,
                )
                img_path = image_dir / img_name
                save_image_blob(image_part.blob, img_path)

                meta = base_meta.copy()
                meta["image_path"] = str(img_path.relative_to(project_root))
                segments.append((text or "[Image-only content]", meta))
                img_count += 1
                image_found = True

        # If no image, but text exists → add as regular text chunk
        if not image_found and text:
            segments.append((text, base_meta))

    print(f"[INFO] Extracted {len([s for s in segments if 'image_path' in s[1]])} images from {path.name}")


    # Tables as additional segments
    for tbl_idx, table in enumerate(document.tables, start=1):
        rows = []
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                rows.append(" | ".join(row_cells))
        if rows:
            tbl_text = "\n".join(rows)
            meta = {
                "doc_type": "docx",
                "table_number": tbl_idx,
                "source_filepath": str(path),
                "doc_id": doc_id,
            }
            segments.append((tbl_text, meta))

    return segments
