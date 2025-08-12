import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from scripts.chunking.models import Chunk
from scripts.chunking.models import ImageChunk  # if used in your retrieval fusion


class RunLogger:
    def __init__(self, project_dir: Path, run_name: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = run_name or f"run_{timestamp}"
        self.base_dir = Path(project_dir) / "output" / "runs" / run_folder
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log_metadata(self, metadata: dict) -> None:
        path = self.base_dir / "run_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def log_chunks(self, chunks: List[Chunk]) -> None:
        path = self.base_dir / "retrieved_chunks.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                record = {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "meta": chunk.meta,
                }
                f.write(json.dumps(record) + "\n")

    def log_images(self, images: List[ImageChunk]) -> None:
        path = self.base_dir / "image_matches.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for img in images:
                record = {
                    "image_path": img.meta.get("image_path"),
                    "description": img.description,
                    "similarity": img.meta.get("similarity", None),
                    "source_chunk_id": img.meta.get("source_chunk_id"),
                }
                f.write(json.dumps(record) + "\n")

    def log_prompt(self, prompt: str) -> None:
        path = self.base_dir / "llm_prompt.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt)

    def log_response(self, answer: str) -> None:
        path = self.base_dir / "llm_response.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(answer)
