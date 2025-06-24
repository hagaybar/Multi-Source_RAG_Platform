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
import uuid

# CRITICAL: Add project root to Python path for imports
import sys
project_root = Path(__file__).parent.parent.parent  # Go up from scripts/embeddings/ to project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"DEBUG: Added to Python path: {project_root}")

from scripts.api_clients.openai.batch_embedder import BatchEmbedder
from scripts.core.project_manager import ProjectManager
from scripts.chunking.models import Chunk
from scripts.utils.logger import LoggerManager
from scripts.embeddings.embedder_registry import get_embedder
from scripts.utils.chunk_utils import load_chunks
from scripts.utils.chunk_utils import deduplicate_chunks




class UnifiedEmbedder:
    """
    A general-purpose, agent-ready embedder with batch + async support for OpenAI, local, or custom clients.

    Supports:
    - Text deduplication
    - Chunked sync batching
    - OpenAI async batch jobs
    - FAISS + JSONL persistence
    """
    def __init__(self, project: ProjectManager, runtime_config: Optional[dict] = None):
        print("=" * 80)
        print("DEBUG: UnifiedEmbedder.__init__() STARTING")
        print("=" * 80)

        self.project = project
        # Use runtime_config if provided, otherwise use project.config
        self.config = runtime_config if runtime_config is not None else project.config
        self.logger = LoggerManager.get_logger("embedder", log_file=project.get_log_path("embedder"))

        print(f"DEBUG: Full project config: {self.config}")
        print(f"DEBUG: Project config type: {type(self.config)}")

        # Get embedding config section
        embedding_config = self.config.get('embedding', {})

        # Read configuration values using direct dictionary access
        self.mode = embedding_config.get("mode", "batch")
        self.skip_duplicates = embedding_config.get("skip_duplicates", True)
        self.batch_size = embedding_config.get("embed_batch_size", 100)
        self.use_async_batch = embedding_config.get("use_async_batch", False)
        self.chunks_path = self.project.get_chunks_path()

        print(f"DEBUG: embedding.mode = {self.mode}")
        print(f"DEBUG: embedding.skip_duplicates = {self.skip_duplicates}")
        print(f"DEBUG: embedding.embed_batch_size = {self.batch_size}")
        print(f"DEBUG: embedding.use_async_batch = {self.use_async_batch}")
        print(f"DEBUG: chunks_path = {self.chunks_path}")

        self.logger.info(f"DEBUG: embedding.use_async_batch config value: {embedding_config.get('use_async_batch', 'NOT_SET')}")
        self.logger.info(f"DEBUG: self.use_async_batch final value: {self.use_async_batch}")

        # Handle embedder initialization based on mode
        if self.use_async_batch:
            print("DEBUG: ASYNC BATCH MODE DETECTED - Skipping local embedder")
            # For async batch, we use OpenAI so set the correct dimension
            self.embedder = None  # Won't be used for async batch
            self.dim = 3072  # text-embedding-3-large dimension
            self.logger.info("Using OpenAI async batch mode - local embedder bypassed")
            print("DEBUG: Set dim=3072 for OpenAI text-embedding-3-large")
        else:
            print("DEBUG: REGULAR MODE - Loading local embedder")
            # For regular mode, use the configured embedder
            self.embedder = get_embedder(project)
            self.dim = self.embedder.encode(["probe"])[0].__len__()
            self.logger.info(f"Embedder type: {type(self.embedder)} | Dimension: {self.dim}")
            print(f"DEBUG: Local embedder type: {type(self.embedder)}")
            print(f"DEBUG: Local embedder dimension: {self.dim}")

        self.logger.info(f"Embedder initialized with mode: {self.mode} | batch_size: {self.batch_size} | use_async_batch: {self.use_async_batch}")
        
        print("=" * 80)
        print("DEBUG: UnifiedEmbedder.__init__() COMPLETE")
        print(f"DEBUG: Final state - use_async_batch: {self.use_async_batch}")
        print("=" * 80)

    def run_from_file(self) -> None:
        print("DEBUG: run_from_file() called")
        chunks = load_chunks(self.chunks_path)
        print(f"DEBUG: Loaded {len(chunks)} chunks from file")
        self.run(chunks)

    def run_from_folder(self) -> None:
        print("DEBUG: run_from_folder() called")
        chunk_files = list(self.project.input_dir.glob("chunks_*.tsv"))
        print(f"DEBUG: Found chunk files: {[f.name for f in chunk_files]}")
        
        if not chunk_files:
            self.logger.warning("No chunks_*.tsv files found.")
            print("DEBUG: No chunk files found - returning")
            return

        for path in chunk_files:
            print(f"DEBUG: Processing chunk file: {path.name}")
            self.logger.info(f"Embedding chunks from: {path.name}")
            chunks = load_chunks(path)
            print(f"DEBUG: Loaded {len(chunks)} chunks from {path.name}")
            self.run(chunks)

    def run(self, chunks: List[Chunk]) -> None:
        print("\n" + "=" * 80)
        print("DEBUG: UnifiedEmbedder.run() ENTRY")
        print("=" * 80)
        
        print(f"DEBUG: run() called with {len(chunks)} chunks")
        print(f"DEBUG: use_async_batch = {self.use_async_batch}")
        print(f"DEBUG: use_async_batch type = {type(self.use_async_batch)}")
        
        self.logger.info(f"[DEBUG] run() called with {len(chunks)} chunks")
        self.logger.info(f"[DEBUG] use_async_batch = {self.use_async_batch}")
        
        # Group by doc_type
        grouped = {}
        for chunk in chunks:
            doc_type = chunk.meta.get("doc_type", "default")
            grouped.setdefault(doc_type, []).append(chunk)
        
        print(f"DEBUG: Grouped chunks: {[(k, len(v)) for k, v in grouped.items()]}")
        self.logger.info(f"[DEBUG] Grouped chunks: {[(k, len(v)) for k, v in grouped.items()]}")

        for doc_type, chunk_group in grouped.items():
            print(f"\nDEBUG: Processing doc_type: {doc_type}")
            print(f"DEBUG: Chunk group size: {len(chunk_group)}")
            print(f"DEBUG: use_async_batch check: {self.use_async_batch}")
            
            self.logger.info(f"[DEBUG] Processing doc_type: {doc_type}, use_async_batch: {self.use_async_batch}")
            
            if self.use_async_batch:
                print(f"DEBUG: CALLING run_async_batch for {doc_type}")
                self.logger.info(f"[DEBUG] Calling run_async_batch for {doc_type}")
                self.run_async_batch(doc_type, chunk_group)
            else:
                print(f"DEBUG: CALLING _embed_and_store for {doc_type}")
                self.logger.info(f"[DEBUG] Calling _embed_and_store for {doc_type}")
                if self.embedder is None:
                    error_msg = "Local embedder not initialized. Cannot run non-async batch mode."
                    print(f"ERROR: {error_msg}")
                    raise RuntimeError(error_msg)
                self._embed_and_store(doc_type, chunk_group)
        
        print("=" * 80)
        print("DEBUG: UnifiedEmbedder.run() COMPLETE")
        print("=" * 80)

    def _embed_and_store(self, doc_type: str, chunks: List[Chunk]) -> None:
        print(f"\nDEBUG: _embed_and_store() called for {doc_type} with {len(chunks)} chunks")
        
        faiss_path = self.project.get_faiss_path(doc_type)
        meta_path = self.project.get_metadata_path(doc_type)

        existing_ids = set()
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            meta = json.loads(line)
                            existing_ids.add(meta["id"])
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Skipping malformed JSON line: {line[:100]}...")
                            continue
        else:
            index = faiss.IndexFlatL2(self.dim)

        new_chunks = []
        for chunk in chunks:
            if self.skip_duplicates and chunk.id in existing_ids:
                continue
            new_chunks.append(chunk)

        if not new_chunks:
            self.logger.info("No new chunks to embed.")
            print("DEBUG: No new chunks to embed - returning")
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
        print(f"DEBUG: Embedded {len(vectors)} vectors in {end-start:.2f}s using LOCAL embedder")

        emb_array = np.vstack(vectors).astype("float32")
        index.add(emb_array)
        faiss.write_index(index, str(faiss_path))

        with open(meta_path, "a", encoding="utf-8") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk.meta) + "\n")

    def run_async_batch(self, doc_type: str, chunks: List[Chunk]) -> None:
        print("\n" + "=" * 100)
        print("DEBUG: run_async_batch() *** ENTRY POINT ***")
        print("=" * 100)
        
        print(f"DEBUG: run_async_batch() called with:")
        print(f"DEBUG:   - doc_type: {doc_type}")
        print(f"DEBUG:   - chunks: {len(chunks)}")
        print(f"DEBUG:   - use_async_batch: {self.use_async_batch}")
        
        self.logger.info(f"Running async batch for doc_type={doc_type}, chunks={len(chunks)}")

        # Only supports OpenAI for now
        try:
            from openai import OpenAI
            print("DEBUG: OpenAI import successful")
        except ImportError as e:
            error_msg = "Async batch mode requires OpenAI SDK"
            print(f"ERROR: {error_msg} - {e}")
            raise RuntimeError(error_msg)

        meta_path = self.project.get_metadata_path(doc_type)
        faiss_path = self.project.get_faiss_path(doc_type)
        
        print(f"DEBUG: meta_path: {meta_path}")
        print(f"DEBUG: faiss_path: {faiss_path}")

        # Filter duplicates
        print("DEBUG: Getting existing IDs for deduplication...")
        existing_ids = self._get_existing_ids(doc_type)
        print(f"DEBUG: Found {len(existing_ids)} existing IDs")
        
        new_chunks = []
        skipped_count = 0
        
        for chunk in chunks:
            if self.skip_duplicates and chunk.id in existing_ids:
                print(f"DEBUG: Skipping duplicate chunk: {chunk.id[:16]}...")
                skipped_count += 1
                continue
            new_chunks.append(chunk)


        print(f"DEBUG: After deduplication:")
        print(f"DEBUG:   - Original chunks: {len(chunks)}")
        print(f"DEBUG:   - Skipped duplicates: {skipped_count}")
        print(f"DEBUG:   - New chunks to process: {len(new_chunks)}")

        if not new_chunks:
            print("DEBUG: No new chunks to embed - EARLY RETURN")
            self.logger.info("No new chunks to embed.")
            return

        chunk_texts = [c.text for c in new_chunks]
        # Use chunk IDs as custom_ids for better tracking
        custom_ids = [chunk.id for chunk in new_chunks]


        print(f"DEBUG: Prepared for BatchEmbedder:")
        print(f"DEBUG:   - chunk_texts length: {len(chunk_texts)}")
        print(f"DEBUG:   - custom_ids length: {len(custom_ids)}")
        print(f"DEBUG:   - First few custom_ids: {custom_ids[:3]}")

        self.logger.info(f"Submitting async batch job to OpenAI: {len(chunk_texts)} chunks")

        # BatchEmbedder handles everything: JSONL creation, submission, waiting, download
        print("DEBUG: Creating BatchEmbedder instance...")
        batch_embedder = BatchEmbedder(
            model="text-embedding-3-large", 
            output_dir=self.project.output_dir, 
            logger=self.logger
        )
        print(f"DEBUG: BatchEmbedder created: {batch_embedder}")
        
        # This will automatically create JSONL, submit to OpenAI, wait for completion, and return results
        print("DEBUG: About to call BatchEmbedder.run() - THIS SHOULD CALL OPENAI!")
        print("DEBUG: If you don't see OpenAI API calls after this, the issue is in BatchEmbedder")
        
        try:
            result_dict = batch_embedder.run(chunk_texts, ids=custom_ids)
            print(f"DEBUG: BatchEmbedder.run() returned: {type(result_dict)} with {len(result_dict)} items")
        except Exception as e:
            print(f"ERROR: BatchEmbedder.run() failed: {e}")
            print(f"ERROR: Exception type: {type(e)}")
            raise

        # Extract vectors in the same order as chunks
        vectors = []
        successful_chunks = []
        
        for chunk in new_chunks:
            chunk_id = chunk.id
            if chunk_id in result_dict:
                vectors.append(result_dict[chunk_id])
                successful_chunks.append(chunk)
            else:
                self.logger.warning(f"No embedding returned for chunk ID: {chunk_id}")
                print(f"DEBUG: WARNING - No embedding for chunk ID: {chunk_id}")

        if not vectors:
            error_msg = "No embeddings were successfully processed!"
            print(f"ERROR: {error_msg}")
            self.logger.error(error_msg)
            return

        print(f"DEBUG: Successfully processed {len(vectors)} embeddings")

        # Convert to numpy array and add to FAISS
        vec_array = np.array(vectors, dtype="float32")
        print(f"DEBUG: Created numpy array with shape: {vec_array.shape}")
        
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
            print("DEBUG: Loaded existing FAISS index")
        else:
            index = faiss.IndexFlatL2(self.dim)
            print(f"DEBUG: Created new FAISS index with dimension {self.dim}")

        index.add(vec_array)
        faiss.write_index(index, str(faiss_path))
        print(f"DEBUG: Added {len(vectors)} vectors to FAISS and saved to {faiss_path}")

        # Store metadata in proper JSONL format
        with open(meta_path, "a", encoding="utf-8") as f:
            for chunk in successful_chunks:
                f.write(json.dumps(chunk.meta) + "\n")

        print(f"DEBUG: Saved metadata for {len(successful_chunks)} chunks to {meta_path}")
        self.logger.info(f"Async batch embedding complete. Stored {len(vectors)} vectors and metadata.")
        
        print("=" * 100)
        print("DEBUG: run_async_batch() *** COMPLETE ***")
        print("=" * 100)

    def _get_existing_ids(self, doc_type: str) -> set:
        """Get existing chunk IDs to avoid duplicates."""
        existing_ids = set()
        meta_path = self.project.get_metadata_path(doc_type)
        
        print(f"DEBUG: _get_existing_ids() for doc_type: {doc_type}")
        print(f"DEBUG: Checking meta_path: {meta_path}")
        print(f"DEBUG: Meta file exists: {meta_path.exists()}")
        
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    try:
                        meta = json.loads(line)
                        existing_ids.add(meta["id"])
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping malformed JSON line: {line[:100]}...")
                        print(f"DEBUG: WARNING - Malformed JSON line {line_count}: {line[:50]}...")
                        continue
                print(f"DEBUG: Processed {line_count} lines from metadata file")
        
        print(f"DEBUG: Found {len(existing_ids)} existing IDs")
        return existing_ids

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()