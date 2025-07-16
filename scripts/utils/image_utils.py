import os
from pathlib import Path
from PIL import Image
import io
import hashlib
from scripts.utils.logger import LoggerManager

logger = LoggerManager.get_logger("image_utils")


def infer_project_root(doc_path: Path) -> Path:
    """
    Given the path to a raw input file, infer the project root.
    Assumes file lives in: data/projects/<project>/input/raw/
    """
    try:
        parts = doc_path.parts
        proj_index = parts.index("projects") + 1
        project_name = parts[proj_index]
        return Path(*parts[:proj_index + 1])
    except Exception as e:
        logger.warning(f"Failed to infer project root from path: {doc_path}")
        raise e


def ensure_image_cache_dir(project_root: Path) -> Path:
    """
    Create the images cache dir if missing. Returns the absolute path.
    """
    path = project_root / "input" / "cache" / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image_blob(image_bytes: bytes, output_path: Path) -> None:
    """
    Save raw image bytes to disk using PIL for robustness.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.save(output_path)
        logger.info(f"Saved image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        raise e


def save_image_pillow(image: Image.Image, output_path: Path) -> None:
    """
    Save an in-memory Pillow image (from e.g., pptx or pdfplumber) to disk.
    """
    try:
        image.save(output_path)
        logger.info(f"Saved image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Pillow image to {output_path}: {e}")
        raise e


def generate_image_filename(doc_id: str, page_number: int, img_index: int, ext: str = "png") -> str:
    """
    Create a consistent filename for saved image.
    """
    doc_base = Path(doc_id).stem.replace(" ", "_")
    return f"{doc_base}_page{page_number}_img{img_index}.{ext}"


def hash_image_content(image_bytes: bytes) -> str:
    """
    Generate a SHA256 hash of image content to avoid redundant saves (optional).
    """
    return hashlib.sha256(image_bytes).hexdigest()
