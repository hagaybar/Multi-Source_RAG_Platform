from __future__ import annotations

import pathlib
from pathlib import Path
from typing import List, Tuple

from docx import Document
from docx.oxml.ns import qn

from scripts.utils.image_utils import (
    infer_project_root,
    record_image_metadata,
    get_project_image_dir,
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
    project_name = project_root.name
    image_dir = get_project_image_dir(project_name)
    print(f"[loader] Writing image to: {image_dir}")  # Should be inside data/projects/{project}/input/cache/images
    

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
        meta = base_meta.copy()

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

                record_image_metadata(meta, img_path, project_root)
                image_found = True
                img_count += 1

        if text or image_found:
            segments.append((text or "[Image-only content]", meta))

    print(f"[INFO] Extracted {sum('image_paths' in s[1] for s in segments)} image-attached chunks from {path.name}")


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
