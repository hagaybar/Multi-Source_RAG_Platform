import hashlib
import json
import csv
from pathlib import Path
from typing import List, Dict
from glob import glob

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.utils.logger import LoggerManager
from scripts.utils.chunks_io import load_chunks

class ChunkEmbedder:
    # def __init__(self, project: ProjectManager, model_name: str = "BAAI/bge-large-en"):
    #     self.project = project
    #     self.model = SentenceTransformer(model_name)
    #     self.dim = self.model.get_sentence_embedding_dimension()
    #     self.logger = LoggerManager.get_logger("embedder", log_file=self.project.get_log_path("embedder"))

    def __init__(self, project: ProjectManager, model_name: str = "BAAI/bge-large-en"):
        self.project = project
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.logger = LoggerManager.get_logger("embedder", log_file=self.project.get_log_path("embedder"))
        
        self.skip_duplicates = self.project.config.get("embedding.skip_duplicates", True)
        self.logger.info(f"Duplicate skipping is {'enabled' if self.skip_duplicates else 'disabled'}")

        # Optional: Set default path for backward compatibility, but don't enforce its existence here
        self.chunks_path = self.project.get_chunks_path()



    def load_chunks_from_tsv(self) -> List[Chunk]:
        chunks = []
        chunk_path = self.project.input_dir / "chunks.tsv"

        if not chunk_path.exists():
            self.logger.error(f"chunks.tsv not found at {chunk_path}")
            return []

        with open(chunk_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                chunk_id = row.get("chunk_id")
                doc_id = row.get("doc_id")
                text = row.get("text", "")
                meta_json = row.get("meta_json", "{}")
                try:
                    meta = json.loads(meta_json)
                    meta["chunk_id"] = chunk_id
                    meta["doc_id"] = doc_id
                    token_count = len(text.split())
                    chunks.append(Chunk(text=text, meta=meta, token_count=token_count))
                except Exception as e:
                    self.logger.warning(f"Failed to load chunk row: {e}")

        self.logger.info(f"Loaded {len(chunks)} chunks from {chunk_path}")
        return chunks

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

        new_embeddings = []
        new_metadata = []

        for chunk in chunks:
            chunk_id = self._hash_text(chunk.text)
            if self.skip_duplicates and chunk_id in existing_ids:
                self.logger.debug(f"Skipping duplicate chunk {chunk_id[:8]}...")
                continue
            emb = self.model.encode(chunk.text)
            new_embeddings.append(emb)
            chunk.meta["id"] = chunk_id
            new_metadata.append(chunk.meta)

        if new_embeddings:
            emb_array = np.vstack(new_embeddings).astype("float32")
            index.add(emb_array)
            faiss.write_index(index, str(index_path))
            with open(meta_path, "a", encoding="utf-8") as f:
                for meta in new_metadata:
                    f.write(json.dumps(meta) + "\n")
            self.logger.info(f"Appended {len(new_embeddings)} new vectors to {index_path.name}")
        else:
            self.logger.info("No new chunks to embed.")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
