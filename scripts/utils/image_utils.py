import os
from pathlib import Path
from PIL import Image
import io
import hashlib
from scripts.utils.logger import LoggerManager

logger = LoggerManager.get_logger("image_utils")


def infer_project_root(doc_path: Path) -> Path:
    """
    Given a path like .../data/projects/<project>/input/raw/..., 
    return: Path("data/projects/<project>")
    """
    parts = doc_path.resolve().parts
    for i in range(len(parts) - 2):
        if parts[i] == "data" and parts[i + 1] == "projects":
            return Path(*parts[:i + 3])  # includes 'data/projects/<project>'
    raise ValueError(f"Could not infer project root from: {doc_path}")



def get_project_image_dir(project_name: str) -> Path:
    """
    Given a project name, returns the full path to its image cache directory:
    data/projects/<project_name>/input/cache/images

    Ensures the directory exists.
    """
    root = Path("data") / "projects" / project_name / "input" / "cache" / "images"
    root.mkdir(parents=True, exist_ok=True)
    return root


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


def record_image_metadata(meta: dict, image_path: Path, project_root: Path) -> None:
    """
    Adds a relative image path to the given chunk metadata dict.
    Ensures metadata uses 'image_paths' (a list) instead of single string.
    """
    image_path = image_path.resolve()
    rel_path = image_path.relative_to(project_root / "input")

    if "image_paths" not in meta:
        meta["image_paths"] = []
    meta["image_paths"].append(str(rel_path))
