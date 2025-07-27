import json
import uuid
from pathlib import Path
from typing import List
import hashlib
import numpy as np
import faiss

from scripts.chunking.models import ImageChunk
from scripts.core.project_manager import ProjectManager
from scripts.embeddings.embedder_registry import get_embedder
from scripts.utils.logger import LoggerManager


class ImageIndexer:
    def __init__(self, project: ProjectManager):
        self.project = project
        self.output_dir = project.output_dir
        self.logger = LoggerManager.get_logger("image_indexer", log_file=project.get_log_path("embedder"))
        self.embedder = get_embedder(project)
        self.dim = self.embedder.encode(["test"])[0].__len__()

        self.index_path = self.output_dir / "image_index.faiss"
        self.meta_path = self.output_dir / "image_metadata.jsonl"

        self.logger.info(f"ImageIndexer initialized.")
        self.logger.info(f"Vector dimension: {self.dim}")
        self.logger.info(f"Index path: {self.index_path}")
        self.logger.info(f"Metadata path: {self.meta_path}")

    def run(self, image_chunks: List[ImageChunk]) -> None:
        if not image_chunks:
            self.logger.warning("No image chunks provided. Skipping indexing.")
            return

        self.logger.info(f"Indexing {len(image_chunks)} image chunks...")

        # Prepare descriptions
        texts = [chunk.description for chunk in image_chunks]
        ids = [chunk.id for chunk in image_chunks]

        # Embed
        vectors = self.embedder.encode(texts)
        emb_array = np.vstack(vectors).astype("float32")

        # Load or create FAISS index
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            self.logger.info("Loaded existing image FAISS index.")
        else:
            index = faiss.IndexFlatL2(self.dim)
            self.logger.info("Created new image FAISS index.")

        index.add(emb_array)
        faiss.write_index(index, str(self.index_path))
        self.logger.info(f"Added {len(vectors)} vectors to index.")

        # Append metadata
        # Append metadata
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for chunk in image_chunks:
                # Ensure deterministic image_hash (used for deduplication)
                if "image_hash" not in chunk.meta:
                    chunk.meta["image_hash"] = hashlib.sha256(chunk.description.strip().encode("utf-8")).hexdigest()

                record = {
                    "id": chunk.id,
                    "description": chunk.description,
                    **chunk.meta
                }
                f.write(json.dumps(record) + "\n")

        self.logger.info(f"Wrote {len(image_chunks)} metadata records to {self.meta_path}.")
