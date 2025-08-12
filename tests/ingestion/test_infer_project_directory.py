from pathlib import Path
from scripts.ingestion.pptx import _infer_project_root

sample_path = Path("data/projects/demo-image-ingest/input/raw/ACQ/sample.pptx")
print(_infer_project_root(sample_path))
