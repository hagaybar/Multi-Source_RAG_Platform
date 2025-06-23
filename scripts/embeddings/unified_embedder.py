import os
import hashlib
import json
import csv
import time
import pandas as pd
import numpy as np
import faiss
from typing import List, Optional, Union
from pathlib import Path
from scripts.api_clients.openai.batch_embedder import BatchEmbedder
from scripts.core.project_manager import ProjectManager
from scripts.chunking.models import Chunk
from scripts.utils.logger import LoggerManager
from scripts.utils.chunks_io import load_chunks
from scripts.embeddings.embedder_registry import get_embedder


class UnifiedEmbedder:
    """
    A general-purpose, agent-ready embedder with batch + async support for OpenAI, local, or custom clients.

    Supports:
    - Text deduplication
    - Chunked sync batching
    - OpenAI async batch jobs
    - FAISS + JSONL persistence
    """

    def __init__(self, project: ProjectManager):
        self.project = project
        self.config = self.project.config
        self.logger = LoggerManager.get_logger("embedder", log_file=self.project.get_log_path("embedder"))

        self.embedder = get_embedder(project)
        self.dim = self.embedder.encode(["probe"])[0].__len__()

        self.mode = self.config.get("embedding.mode", "batch")
        self.skip_duplicates = self.config.get("embedding.skip_duplicates", True)
        self.batch_size = self.config.get("embedding.embed_batch_size", 100)
        self.use_async_batch = self.config.get("embedding.use_async_batch", False)
        self.chunks_path = self.project.get_chunks_path()

        self.logger.info(f"Embedder initialized with mode: {self.mode} | batch_size: {self.batch_size}")

    def run_from_file(self) -> None:
        chunks = load_chunks(self.chunks_path)
        self.run(chunks)

    def run_from_folder(self) -> None:
        chunk_files = list(self.project.input_dir.glob("chunks_*.tsv"))
        if not chunk_files:
            self.logger.warning("No chunks_*.tsv files found.")
            return

        for path in chunk_files:
            self.logger.info(f"Embedding chunks from: {path.name}")
            chunks = load_chunks(path)
            self.run(chunks)

    def run(self, chunks: List[Chunk]) -> None:
        # Group by doc_type
        grouped = {}
        for chunk in chunks:
            doc_type = chunk.meta.get("doc_type", "default")
            grouped.setdefault(doc_type, []).append(chunk)

        for doc_type, chunk_group in grouped.items():
            if self.use_async_batch:
                self.run_async_batch(doc_type, chunk_group)
            else:
                self._embed_and_store(doc_type, chunk_group)

    def _embed_and_store(self, doc_type: str, chunks: List[Chunk]) -> None:
        faiss_path = self.project.get_faiss_path(doc_type)
        meta_path = self.project.get_metadata_path(doc_type)

        existing_ids = set()
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        meta = json.loads(line)
                        existing_ids.add(meta["id"])
        else:
            index = faiss.IndexFlatL2(self.dim)

        new_chunks = []
        for chunk in chunks:
            chunk_id = self._hash(chunk.text)
            if self.skip_duplicates and chunk_id in existing_ids:
                continue
            chunk.meta["id"] = chunk_id
            new_chunks.append(chunk)

        if not new_chunks:
            self.logger.info("No new chunks to embed.")
            return

        texts = [c.text for c in new_chunks]
        vectors = []

        start = time.time()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            if self.mode == "batch":
                batch_vecs = self.embedder.encode(batch)
            elif self.mode == "single":
                batch_vecs = [self.embedder.encode([t])[0] for t in batch]
            else:
                raise ValueError(f"Unknown embedding mode: {self.mode}")
            vectors.extend(batch_vecs)
        end = time.time()

        self.logger.info(f"Embedded {len(vectors)} vectors in {end-start:.2f}s")

        emb_array = np.vstack(vectors).astype("float32")
        index.add(emb_array)
        faiss.write_index(index, str(faiss_path))

        with open(meta_path, "a", encoding="utf-8") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk.meta) + "\\n")

    def run_async_batch(self, doc_type: str, chunks: List[Chunk]) -> None:
        # Only supports OpenAI for now
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Async batch mode requires OpenAI SDK")

        meta_path = self.project.get_metadata_path(doc_type)
        faiss_path = self.project.get_faiss_path(doc_type)

        chunk_texts = [c.text for c in chunks]
        ids = [f"{doc_type}-chunk-{i}" for i in range(len(chunk_texts))]
        df = pd.DataFrame({ "text": chunk_texts, "custom_id": ids })

        self.logger.info(f"Submitting async batch job to OpenAI: {len(df)} chunks")

        batch_embedder = BatchEmbedder(model="text-embedding-3-large", output_dir=self.project.output_dir)
        result_dict = batch_embedder.run(df["text"].tolist(), ids=ids)

        vectors = [result_dict[cid] for cid in df["custom_id"] if cid in result_dict]
        vec_array = np.array(vectors, dtype="float32")

        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
        else:
            index = faiss.IndexFlatL2(self.dim)

        index.add(vec_array)
        faiss.write_index(index, str(faiss_path))

        df.drop(columns=["custom_id"]).to_csv(meta_path, sep="\\t", mode="a", header=not os.path.exists(meta_path), index=False)
        self.logger.info(f"Async batch embedding complete. Stored {len(vectors)} vectors.")

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()