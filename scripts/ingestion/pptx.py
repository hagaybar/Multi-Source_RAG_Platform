from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from scripts.ingestion.models import AbstractIngestor, UnsupportedFileError


def _infer_project_root(file_path: Path) -> Path:
    parts = file_path.resolve().parts
    if "projects" in parts:
        idx = parts.index("projects")
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
    return file_path.parent


class PptxIngestor(AbstractIngestor):
    """
    Ingestor for PPTX files.
    """

    def ingest(self, filepath: str) -> list[tuple[str, dict]]:
        """
        Ingests data from the given PPTX filepath.

        Args:
            filepath: Path to the PPTX file to ingest.

        Returns:
            A list of tuples, where each tuple contains the extracted text
            and associated metadata (slide number, type, doc_type).
        """
        if not filepath.endswith(".pptx"):
            raise UnsupportedFileError("File is not a .pptx file.")

        extracted_data = []
        try:
            prs = Presentation(filepath)
            file_path = Path(filepath)
            project_root = _infer_project_root(file_path)
            file_stem = file_path.stem
            for i, slide in enumerate(prs.slides):
                slide_number = i + 1
                text_on_slide = []

                image_counter = 0

                # Extract text from shapes and detect images
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text_on_slide.append(shape.text.strip())
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        image_counter += 1
                        img_name = f"{file_stem}_page{slide_number}_img{image_counter}.png"
                        rel_dir = Path("input") / "cache" / "images"
                        out_path = project_root / rel_dir / img_name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_path, "wb") as f:
                            f.write(shape.image.blob)
                        img_rel = str(rel_dir / img_name)
                        if text_on_slide:
                            slide_content = "\n".join(text_on_slide).strip()
                        else:
                            slide_content = ""
                        slide_meta = {
                            "slide_number": slide_number,
                            "type": "slide_content",
                            "doc_type": "pptx",
                            "image_path": img_rel,
                        }
                        extracted_data.append((slide_content, slide_meta))

                # Extract text from presenter notes
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes_text = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        notes_meta = {
                            "slide_number": slide_number,
                            "type": "presenter_notes",
                            "doc_type": "pptx"
                        }
                        # Format notes text consistently
                        formatted_notes = (
                            f"Presenter Notes (Slide {slide_number}):\n"
                            f"{notes_text}"
                        )
                        extracted_data.append((formatted_notes, notes_meta))

                # Combine all text from the slide itself when no images found
                if text_on_slide:
                    valid_texts_on_slide = [t for t in text_on_slide if t]
                    if valid_texts_on_slide:
                        slide_content = "\n".join(valid_texts_on_slide).strip()
                        if slide_content and image_counter == 0:
                            slide_meta = {
                                "slide_number": slide_number,
                                "type": "slide_content",
                                "doc_type": "pptx",
                            }
                            extracted_data.append((slide_content, slide_meta))

        except Exception as e:
            # Catch exceptions from python-pptx or other issues
            raise UnsupportedFileError(f"Error processing PPTX file {filepath}: {e}")

        return extracted_data
