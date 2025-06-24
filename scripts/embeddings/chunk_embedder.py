import hashlib
import json
import csv
from pathlib import Path
from typing import List, Dict, Generator

import numpy as np
import faiss
import time

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.utils.logger import LoggerManager
from scripts.utils.chunk_utils import load_chunks
from scripts.embeddings.embedder_registry import get_embedder


class ChunkEmbedder:
    def __init__(self, project: ProjectManager):
        self.project = project
        self.logger = LoggerManager.get_logger("embedder", log_file=self.project.get_log_path("embedder"))
        self.embedder = get_embedder(project)
        self.dim = self.embedder.encode(["embedding-dim-probe"])[0].shape[0]

        self.skip_duplicates = self.project.config.get("embedding.skip_duplicates", True)
        self.embedding_mode = self.project.config.get("embedding.mode", "batch")
        self.batch_size = self.project.config.get("embedding.embed_batch_size", 100)
        if self.embedding_mode not in ("batch", "single"):
            raise ValueError("embedding.mode must be 'batch' or 'single'")

        self.logger.info(f"Embedding mode: {self.embedding_mode} | Batch size: {self.batch_size}")
        self.logger.info(f"Duplicate skipping is {'enabled' if self.skip_duplicates else 'disabled'}")

        self.chunks_path = self.project.get_chunks_path()

    def run(self, chunks: List[Chunk]) -> None:
        doc_type_map: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            doc_type = chunk.meta.get("doc_type", "default")
            doc_type_map.setdefault(doc_type, []).append(chunk)

        for doc_type, chunk_group in doc_type_map.items():
            self.logger.info(f"Embedding {len(chunk_group)} chunks for doc_type={doc_type}...")
            self._process_doc_type(doc_type, chunk_group)

    def run_from_file(self) -> None:
        chunks = load_chunks(self.chunks_path)
        self.run(chunks)

    def run_from_folder(self) -> None:
        chunk_files = list(self.project.input_dir.glob("chunks_*.tsv"))
        if not chunk_files:
            self.logger.warning("No chunks_*.tsv files found in input directory.")
            return

        for path in chunk_files:
            self.logger.info(f"Loading chunks from: {path.name}")
            chunks = load_chunks(path)
            self.run(chunks)

    def _process_doc_type(self, doc_type: str, chunks: List[Chunk]) -> None:
        index_path = self.project.get_faiss_path(doc_type)
        meta_path = self.project.get_metadata_path(doc_type)

        existing_ids = set()
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        meta = json.loads(line)
                        existing_ids.add(meta["id"])
        else:
            index = faiss.IndexFlatL2(self.dim)

        filtered_chunks = []
        for chunk in chunks:
            chunk_id = self._hash_text(chunk.text)
            if self.skip_duplicates and chunk_id in existing_ids:
                self.logger.debug(f"Skipping duplicate chunk {chunk_id[:8]}...")
                continue
            chunk.meta["id"] = chunk_id
            filtered_chunks.append(chunk)

        if not filtered_chunks:
            self.logger.info("No new chunks to embed.")
            return

        total_embeddings = []
        start_time = time.time()
        with open(meta_path, "a", encoding="utf-8") as f:
            for batch in self._batch_chunks(filtered_chunks, self.batch_size):
                texts = [chunk.text for chunk in batch]
                if self.embedding_mode == "batch":
                    embeddings = self.embedder.encode(texts)
                else:
                    embeddings = [self.embedder.encode([t])[0] for t in texts]

                emb_array = np.vstack(embeddings).astype("float32")
                index.add(emb_array)
                for chunk in batch:
                    f.write(json.dumps(chunk.meta) + "\n")
                total_embeddings.extend(embeddings)

        faiss.write_index(index, str(index_path))
        duration = time.time() - start_time
        self.logger.info(f"Embedded and indexed {len(total_embeddings)} vectors in {duration:.2f} seconds")

    def _batch_chunks(self, items: List[Chunk], batch_size: int) -> Generator[List[Chunk], None, None]:
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()