# ─────────────────────────────────────────────
# 🔧 Standard Library Imports
# ─────────────────────────────────────────────
from datetime import datetime
from time import perf_counter
import json
import csv
import uuid
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Callable, Iterator

# ─────────────────────────────────────────────
# 🌐 OpenAI / External API Clients
# ─────────────────────────────────────────────
from scripts.api_clients.openai.completer import OpenAICompleter

# ─────────────────────────────────────────────
# 🧠 Agents
# ─────────────────────────────────────────────
from scripts.agents.image_insight_agent import ImageInsightAgent

# ─────────────────────────────────────────────
# 🧩 Chunking System
# ─────────────────────────────────────────────
from scripts.chunking.chunker_v3 import split as chunk_text
from scripts.chunking.models import Chunk, ImageChunk

# ─────────────────────────────────────────────
# 📁 Project & Ingestion
# ─────────────────────────────────────────────
from scripts.core.project_manager import ProjectManager
from scripts.ingestion.manager import IngestionManager
from scripts.ingestion.models import RawDoc

# ─────────────────────────────────────────────
# 🔍 Embeddings & Indexing
# ─────────────────────────────────────────────
from scripts.embeddings.unified_embedder import UnifiedEmbedder
from scripts.embeddings.image_indexer import ImageIndexer

# ─────────────────────────────────────────────
# 🤖 Retrieval & Prompting
# ─────────────────────────────────────────────
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.prompting.prompt_builder import PromptBuilder

# ─────────────────────────────────────────────
# 🧰 Utilities
# ─────────────────────────────────────────────
from scripts.utils.logger import LoggerManager
from scripts.utils.chunk_utils import load_chunks
from scripts.utils.run_logger import RunLogger
from scripts.utils.task_paths import TaskPaths
from scripts.utils.logger_context import with_context


# ─────────────────────────────────────────────
# 🚨 Error Policy Configuration
# ─────────────────────────────────────────────

# Error handling policies
ERROR_POLICY_FAIL_FAST = "fail_fast"
ERROR_POLICY_SOFT_FAIL = "soft_fail"

# Default error threshold for soft-fail steps
# Fail if: errors > 0 and successes == 0, OR error_rate >= threshold
DEFAULT_ERROR_THRESHOLD = 0.2  # 20%

# Step error policies
STEP_ERROR_POLICIES = {
    # Fail-fast: any error stops the step immediately
    "retrieve": ERROR_POLICY_FAIL_FAST,
    "ask": ERROR_POLICY_FAIL_FAST,
    
    # Soft-fail: errors are tolerated up to threshold
    "ingest": ERROR_POLICY_SOFT_FAIL,
    "chunk": ERROR_POLICY_SOFT_FAIL,
    "enrich": ERROR_POLICY_SOFT_FAIL,
    "embed": ERROR_POLICY_SOFT_FAIL,
    "index_images": ERROR_POLICY_SOFT_FAIL,
}


class PipelineRunner:
    """
    Orchestrates sequential execution of modular pipeline steps
    (ingest, chunk, enrich, embed, index, retrieve, ask).

    Logging design (aligned with plan_for_fixing_logs.txt):
      • App-level logs → logs/app/pipeline.log (JSON)
      • Per-run logs  → logs/runs/<run_id>/app.log (JSON, with auto context)
      • Run artifacts → logs/runs/<run_id>/* (prompt/response/chunks/images/metadata)
    """

    def __init__(self, project: ProjectManager, config: dict, run_id: str | None = None):
        self.project = project
        self.config = config
        self.steps: list[tuple[str, dict]] = []

        # Optional external run_id (keeps backward compatibility)
        self.run_id = run_id

        # Centralized app/per-run logger (no artifacts)
        paths = TaskPaths()
        self.logger = LoggerManager.get_logger(
            name="pipeline",            # stable subsystem name → logs/app/pipeline.log
            task_paths=paths,
            run_id=self.run_id,          # None => app log; value => logs/runs/<run_id>/app.log
            use_json=True,
        )

        # Per-run helpers (created lazily when a run-scoped step starts)
        self._run_logger: RunLogger | None = None  # artifacts writer
        self._run_id: str | None = None            # materialized run id (folder name)
        self.run_log = None                        # contextual per-run logger

        # Pipeline state
        self.raw_docs: list[RawDoc] = []  # ← Store output of ingest
        self.seen_hashes: set[str] = set()  # ← Optional deduplication base
        self.chunks: list[Chunk] = []
        self.retrieved_chunks = []
        self.last_answer = None
        self._model_name = (
            self.config.get("model_name")
            or self.config.get("llm", {}).get("model")
            or "gpt-4o"
        )

    # ─────────────────────────────────────────────
    # Logging helpers
    # ─────────────────────────────────────────────
    def _ensure_run_logging(self):
        """Create RunLogger (artifacts) + per-run structured logger if missing.
        This is idempotent and safe to call at the start of any run-scoped step.
        """
        if self._run_logger is not None:
            return

        # Artifacts writer → creates logs/runs/<run_id>/
        rl = RunLogger(self.project.root_dir)
        self._run_logger = rl
        self._run_id = rl.base_dir.name  # run folder name is the canonical run_id

        # Structured JSON logger bound to the run → logs/runs/<run_id>/app.log
        base = LoggerManager.get_logger(
            name="pipeline",
            task_paths=TaskPaths(),
            run_id=self._run_id,
            use_json=True,
        )
        # Auto-inject run context (run_id, component) into every line
        self.run_log = with_context(base, run_id=self._run_id, component="pipeline")

        # Optional: small breadcrumb that a run started
        self.run_log.info("run.init")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────
    def set_model(self, model_name: str):
        """Set the LLM model for all subsequent calls."""
        self._model_name = model_name

    def get_model(self) -> str:
        """Get the currently set LLM model."""
        return self._model_name

    # ─────────────────────────────────────────────
    # Error Policy Helpers
    # ─────────────────────────────────────────────
    def _should_fail_step(self, step_name: str, error_count: int, success_count: int) -> bool:
        """
        Determine if a step should fail based on its error policy and counts.
        
        Args:
            step_name: Name of the pipeline step
            error_count: Number of errors encountered
            success_count: Number of successful operations
            
        Returns:
            True if the step should fail and stop pipeline execution
        """
        policy = STEP_ERROR_POLICIES.get(step_name, ERROR_POLICY_FAIL_FAST)
        
        if policy == ERROR_POLICY_FAIL_FAST:
            # Any error causes failure
            return error_count > 0
            
        elif policy == ERROR_POLICY_SOFT_FAIL:
            # Fail if: no successes and any errors, OR error rate >= threshold
            if error_count > 0 and success_count == 0:
                return True
                
            total_operations = error_count + success_count
            if total_operations > 0:
                error_rate = error_count / total_operations
                return error_rate >= DEFAULT_ERROR_THRESHOLD
                
        return False

    def add_step(self, name: str, **kwargs) -> None:
        """
        Adds a step to the pipeline by name, with optional keyword arguments.
        The step must have a corresponding method: `step_<name>()`.
        """
        method_name = f"step_{name}"
        if not hasattr(self, method_name):
            raise ValueError(
                f"Step '{name}' is not implemented (missing method: {method_name})"
            )

        self.steps.append((name, kwargs))
        self.logger.info("step.added", extra={"extra_data": {"name": name, "kwargs": kwargs}})

    def run_steps(self) -> Iterator[str]:
        """
        Executes all configured pipeline steps in order.
        Yields human-readable progress messages for UI or CLI.
        """
        self.logger.info("pipeline.start")
        yield "🚀 Starting pipeline execution..."

        for name, kwargs in self.steps:
            method_name = f"step_{name}"
            yield f"▶️ Running step: {name}"
            self.logger.info("step.run", extra={"extra_data": {"name": name, "kwargs": kwargs}})

            try:
                step_fn: Callable = getattr(self, method_name)
                if not callable(step_fn):
                    raise AttributeError(f"'{method_name}' is not callable.")

                result = step_fn(**kwargs)

                if isinstance(result, Iterator):
                    yield from result
                else:
                    yield f"✅ Step '{name}' completed."

                self.logger.info("step.ok", extra={"extra_data": {"name": name}})

            except Exception as e:
                self.logger.error("step.fail", extra={"extra_data": {"name": name}}, exc_info=True)
                yield f"❌ Step '{name}' failed: {e}"
                raise

        self.logger.info("pipeline.end")
        yield "🏁 Pipeline finished."

    def clear_steps(self) -> None:
        """
        Clears all steps from the pipeline.
        Useful before re-running or resetting the workflow.
        """
        self.steps.clear()
        self.logger.info("steps.cleared")

    # ----------------------------#
    #           Steps             #
    # ----------------------------#

    def step_ingest(self, path: Path = None, **kwargs) -> Iterator[str]:
        """
        Ingests raw documents from the given path or project input/raw directory.
        Applies optional deduplication by content hash (including image references).
        """
        yield "📥 Starting ingestion..."

        ingestion_manager = IngestionManager(
            log_file=self.project.get_log_path("ingestion")  # legacy ingestion log path (unchanged)
        )
        path = path or self.project.input_dir / "raw"

        if not path.exists():
            yield f"❌ Ingestion path does not exist: {path}"
            return

        raw_docs = ingestion_manager.ingest_path(path)
        if not raw_docs:
            yield "⚠️ No documents ingested."
            return

        # Deduplicate by content + image references
        new_docs = []
        for doc in raw_docs:
            hash_base = doc.content.strip()
            if "image_paths" in doc.metadata:
                hash_base += ",".join(doc.metadata["image_paths"])

            doc_hash = hashlib.sha256(hash_base.encode("utf-8")).hexdigest()
            doc.metadata["content_hash"] = doc_hash

            if doc_hash not in self.seen_hashes:
                new_docs.append(doc)
                self.seen_hashes.add(doc_hash)
            else:
                self.logger.info(
                    "ingest.duplicate", extra={"extra_data": {"source": doc.metadata.get("source_filepath")}}
                )

        self.raw_docs = new_docs
        yield f"✅ Ingested {len(new_docs)} unique documents from {path.name}"

    def step_chunk(self, **kwargs) -> Iterator[str]:
        """
        Applies chunking rules to all raw documents.
        Saves results to chunks_<doc_type>.tsv under the input directory.
        """
        yield "📚 Starting chunking..."

        if not self.raw_docs:
            yield "❌ No raw documents available. Run 'ingest' first."
            return

        all_chunks: list[Chunk] = []
        error_count = 0
        success_count = 0

        for i, doc in enumerate(self.raw_docs):
            doc_id = doc.metadata.get("source_filepath", f"doc_{i}")
            doc_type = doc.metadata.get("doc_type", "default")
            if not doc_type:
                yield f"⚠️ Skipping doc with missing doc_type: {doc_id}"
                continue

            meta = doc.metadata.copy()
            meta["doc_id"] = doc_id

            # Optional debug
            self.logger.debug(
                "chunk.debug", extra={"extra_data": {
                    "doc_id": doc_id,
                    "paragraph": meta.get("paragraph_number"),
                    "image_paths": meta.get("image_paths"),
                }}
            )

            try:
                chunks = chunk_text(doc.content, meta)
                all_chunks.extend(chunks)
                success_count += 1
                yield f"✂️ {len(chunks)} chunks from {doc_type.upper()} document: {doc_id}"
            except Exception as e:
                error_count += 1
                yield f"❌ Error chunking {doc_id}: {e}"
                self.logger.warning("chunk.fail", extra={"extra_data": {"doc_id": doc_id, "error": str(e)}})

        if not all_chunks:
            yield "⚠️ No chunks were produced."
            return

        self.chunks = all_chunks

        # Save chunks_*.tsv files grouped by doc_type
        by_type = defaultdict(list)
        for chunk in all_chunks:
            doc_type = chunk.meta.get("doc_type", "default")
            by_type[doc_type].append(chunk)

        for doc_type, chunks in by_type.items():
            chunk_path = self.project.input_dir / f"chunks_{doc_type}.tsv"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(chunk_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(["chunk_id", "doc_id", "text", "token_count", "meta_json"])
                    for chunk in chunks:
                        writer.writerow(
                            [
                                chunk.id,
                                chunk.doc_id,
                                chunk.text,
                                chunk.token_count,
                                json.dumps(chunk.meta),
                            ]
                        )
                yield f"💾 Saved {len(chunks)} chunks to: {chunk_path.name}"
            except Exception as e:
                error_count += 1
                yield f"❌ Failed to write chunks_{doc_type}.tsv: {e}"
                self.logger.error("chunk.write.fail", extra={"extra_data": {"doc_type": doc_type, "error": str(e)}})

        # Check if step should fail based on error policy
        if self._should_fail_step("chunk", error_count, success_count):
            error_msg = f"Chunking failed: {error_count} errors, {success_count} successes"
            self.logger.error("chunk.step.fail", extra={"extra_data": {"errors": error_count, "successes": success_count}})
            if self.run_log:
                self.run_log.error("chunk.step.fail", extra={"extra_data": {"errors": error_count, "successes": success_count}})
                self.run_log.info("run.end", extra={"extra_data": {"status": "failed"}})
            raise Exception(error_msg)

        yield f"✅ Chunking complete. Total chunks: {len(all_chunks)} (Errors: {error_count}, Successes: {success_count})"

    def step_embed(self, **kwargs) -> Iterator[str]:
        """
        Embeds and indexes chunked documents (optionally enriched).
        Uses self.chunks if available; otherwise loads from chunk files on disk.
        """
        yield "🧬 Starting embedding step..."

        embed_config = self.config.get("embedding", {})
        image_enrichment_enabled = embed_config.get("image_enrichment", False)
        use_async = embed_config.get("use_async_batch", False)

        embedder = UnifiedEmbedder(self.project, runtime_config=self.config)
        yield f"⚙️ Embedding mode: {'async-batch' if use_async else 'local/batch'}"

        # Case 1: Use in-memory chunks
        if self.chunks:
            yield f"📎 Using {len(self.chunks)} in-memory chunks..."
            try:
                embedder.run(self.chunks)
                yield "✅ Embedded and indexed all in-memory chunks."
            except Exception as e:
                yield f"❌ Embedding failed: {e}"
                self.logger.error("embed.fail", extra={"extra_data": {"mode": "memory", "error": str(e)}}, exc_info=True)
            return

        # Case 2: Load from file
        base_dir = self.project.input_dir
        enriched_dir = base_dir / "enriched"
        chunk_files = list(base_dir.glob("chunks_*.tsv"))

        if not chunk_files:
            yield "❌ No chunk files found in input/. Run 'chunk' first."
            return

        for chunk_path in chunk_files:
            doc_type = chunk_path.stem.split("_", 1)[-1]
            enriched_path = enriched_dir / f"chunks_{doc_type}.tsv"

            # Use enriched if available and enabled
            path_to_use = (
                enriched_path if image_enrichment_enabled and enriched_path.exists() else chunk_path
            )
            if image_enrichment_enabled and not enriched_path.exists():
                yield (
                    f"⚠️ Enrichment enabled, but enriched file not found for {doc_type}. "
                    "Using base chunks."
                )

            yield f"📄 Loading chunks: {path_to_use.name}"
            chunks = load_chunks(path_to_use)
            yield f"🔢 Loaded {len(chunks)} chunks for embedding..."

            try:
                embedder.run(chunks)
                yield f"✅ Embedded and indexed chunks for: {doc_type}"
            except Exception as e:
                yield f"❌ Embedding failed for {doc_type}: {e}"
                self.logger.error("embed.fail", extra={"extra_data": {"mode": "file", "doc_type": doc_type, "error": str(e)}}, exc_info=True)

        yield "📦 Embedding complete for all doc types."

    def step_enrich(self, overwrite: bool = False, **kwargs) -> Iterator[str]:
        """
        Enrich chunks that contain image references using an image insight agent.
        Loads chunks from memory if available, otherwise from disk.
        Outputs enriched chunks grouped by doc_type to input/enriched/.
        """
        yield "🧠 Starting image enrichment..."
        yield f"🐞 DEBUG: runner has {len(self.chunks)} chunks in memory before enrichment"

        # ─── Fallback: load chunk files from disk ───
        if not self.chunks:
            chunk_paths = list(self.project.input_dir.glob("chunks_*.tsv"))
            yield f"🐞 DEBUG: found {len(chunk_paths)} chunk file(s): {[p.name for p in chunk_paths]}"

            if not chunk_paths:
                yield "❌ No chunks available on disk. Please run 'chunk' first."
                return

            loaded = 0
            for path in chunk_paths:
                chunks = load_chunks(path)
                self.chunks.extend(chunks)
                loaded += len(chunks)

            yield f"🔄 Loaded {loaded} chunks from disk"

        agent = ImageInsightAgent(self.project)
        enriched_chunks: list[Chunk] = []

        count_total = 0
        count_enriched = 0

        for chunk in self.chunks:
            count_total += 1
            img_list = chunk.meta.get("image_paths") or []

            if not img_list:
                enriched_chunks.append(chunk)
                continue

            try:
                all_results: list[Chunk] = []

                for img_path in img_list:
                    temp_meta = dict(chunk.meta)
                    temp_meta["image_path"] = img_path

                    temp_chunk = Chunk(
                        id=chunk.id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        meta=temp_meta,
                    )

                    result = agent.run(temp_chunk, self.project)
                    all_results.extend(result if isinstance(result, list) else [result])

                enriched_chunks.extend(all_results if all_results else [chunk])
                if all_results:
                    count_enriched += 1
                yield f"🖼️ Enriched {len(img_list)} image(s) in chunk: {chunk.id}"

            except Exception as e:
                self.logger.warning("enrich.fail", extra={"extra_data": {"chunk_id": chunk.id, "error": str(e)}})
                enriched_chunks.append(chunk)
                yield f"⚠️ Failed to enrich chunk {chunk.id}: {e}"

        self.chunks = enriched_chunks

        # ─── Save enriched chunks by doc_type ───
        by_type = defaultdict(list)
        for chunk in self.chunks:
            doc_type = chunk.meta.get("doc_type", "default")
            by_type[doc_type].append(chunk)

        enriched_dir = self.project.input_dir / "enriched"
        enriched_dir.mkdir(parents=True, exist_ok=True)

        for doc_type, chunks in by_type.items():
            save_path = enriched_dir / f"chunks_{doc_type}.tsv"
            if save_path.exists() and not overwrite:
                yield (
                    f"⚠️ Enriched file already exists: {save_path.name}. "
                    "Use overwrite=True to replace."
                )
                continue

            try:
                with open(save_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(["chunk_id", "doc_id", "text", "token_count", "meta_json"])
                    for chunk in chunks:
                        writer.writerow(
                            [
                                chunk.id,
                                chunk.doc_id,
                                chunk.text,
                                chunk.token_count,
                                json.dumps(chunk.meta),
                            ]
                        )
                yield f"💾 Saved enriched chunks to: {save_path.name}"
            except Exception as e:
                yield f"❌ Failed to write enriched file: {e}"
                self.logger.error("enrich.write.fail", extra={"extra_data": {"doc_type": doc_type, "error": str(e)}})

        yield f"✅ Enrichment complete: {count_enriched}/{count_total} chunks enriched"

    def step_index_images(self, doc_types: list[str] = None, **kwargs) -> Iterator[str]:
        """
        Index enriched image descriptions into FAISS and metadata JSONL.
        Deduplicates using a SHA256 hash of the image description content.
        Can be run independently if enriched files exist in input/enriched/.
        """
        yield "🔎 Starting image indexing step..."

        doc_types = doc_types or ["pptx", "pdf", "docx"]
        enriched_dir = self.project.input_dir / "enriched"
        meta_path = self.project.output_dir / "image_metadata.jsonl"
        indexer = ImageIndexer(self.project)

        # Load existing hashes to prevent duplicates
        existing_hashes = set()
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            img_hash = record.get("image_hash")
                            if img_hash:
                                existing_hashes.add(img_hash)
                        except json.JSONDecodeError:
                            continue
                yield f"📄 Loaded {len(existing_hashes)} existing image hashes from metadata."
            except Exception as e:
                self.logger.warning("image_index.meta_read.fail", extra={"extra_data": {"error": str(e)}})

        count_total = 0
        count_skipped = 0

        for doc_type in doc_types:
            file_path = enriched_dir / f"chunks_{doc_type}.tsv"

            if not file_path.exists():
                yield f"⚠️ Skipping {doc_type} — no enriched file found: {file_path.name}"
                continue

            yield f"📂 Reading: {file_path.name}"
            image_chunks = []

            try:
                with open(file_path, encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    next(reader)  # Skip header

                    for row in reader:
                        if len(row) < 5:
                            continue
                        meta = json.loads(row[4])
                        summaries = meta.get("image_summaries", [])

                        for summary in summaries:
                            description = summary["description"]
                            if not description or not isinstance(description, str):
                                self.logger.warning("image_index.empty_desc")
                                continue
                            img_hash = hashlib.sha256(
                                description.strip().encode("utf-8")
                            ).hexdigest()

                            if img_hash in existing_hashes:
                                count_skipped += 1
                                continue

                            existing_hashes.add(img_hash)
                            image_chunks.append(
                                ImageChunk(
                                    id=str(uuid.uuid4()),
                                    description=description,
                                    meta={
                                        "image_path": summary["image_path"],
                                        "source_chunk_id": row[0],
                                        "doc_type": meta.get("doc_type"),
                                        "source_filepath": meta.get("source_filepath"),
                                        "page_number": meta.get("page_number"),
                                        "image_hash": img_hash,  # Persisted to metadata
                                    },
                                )
                            )

            except Exception as e:
                self.logger.error("image_index.read.fail", extra={"extra_data": {"file": file_path.name, "error": str(e)}}, exc_info=True)
                yield f"❌ Error reading {file_path.name}: {e}"
                continue

            if not image_chunks:
                yield f"⚠️ No new image summaries to index in {file_path.name}."
                continue

            try:
                indexer.run(image_chunks)
                count_total += len(image_chunks)
                yield f"✅ Indexed {len(image_chunks)} new image chunks for {doc_type}."
            except Exception as e:
                self.logger.error("image_index.index.fail", extra={"extra_data": {"doc_type": doc_type, "error": str(e)}}, exc_info=True)
                yield f"❌ Indexing failed for {doc_type}: {e}"

        if count_total:
            yield (
                f"🧠 Image indexing complete. "
                f"Total indexed: {count_total}, skipped: {count_skipped}"
            )
        else:
            yield f"⚠️ No new image chunks indexed. {count_skipped} duplicates skipped."

    def step_retrieve(
        self, query: str, top_k: int = 5, strategy: str = "late_fusion", **kwargs
    ) -> Iterator[str]:
        """
        Retrieves top-k results (text + image-aware) using late fusion.
        Stores results in self.retrieved_chunks for step_ask() or inspection.
        """
        yield "🔍 Starting retrieval..."
        self._ensure_run_logging()
        t0 = perf_counter()
        self.run_log.info(
            "retrieval.start",
            extra={"extra_data": {"query": query, "top_k": top_k, "strategy": strategy}},
        )

        if not query:
            yield "❌ No query provided."
            return

        try:
            retriever = RetrievalManager(self.project)
            yield f"🔢 Strategy: {strategy}, Top-K: {top_k}"
            chunks = retriever.retrieve(query=query, top_k=top_k, strategy=strategy)

            # Persist artifacts with the SAME run logger
            run_logger = self._run_logger  # type: ignore[assignment]
            try:
                run_logger.log_metadata(  # type: ignore[union-attr]
                    {
                        "query": query,
                        "top_k": top_k,
                        "strategy": strategy,
                        "timestamp": datetime.now().isoformat(),
                        "pipeline_steps": ["retrieve"],
                    }
                )
                run_logger.log_chunks(chunks)  # type: ignore[union-attr]

                # Optional: detect and log image matches
                image_chunks = [
                    c for c in chunks if getattr(c, "description", None) and "image_path" in c.meta
                ]
                if image_chunks:
                    run_logger.log_images(image_chunks)  # type: ignore[union-attr]
            except Exception as e:
                self.run_log.warning("runlogger.retrieve.fail", extra={"extra_data": {"error": str(e)}})

            if not chunks:
                self.run_log.info("retrieval.end", extra={"extra_data": {"hits": 0, "elapsed_ms": int((perf_counter()-t0)*1000)}})
                yield "⚠️ No results retrieved."
                return

            self.retrieved_chunks = chunks
            elapsed_ms = int((perf_counter() - t0) * 1000)
            self.run_log.info("retrieval.end", extra={"extra_data": {"hits": len(chunks), "elapsed_ms": elapsed_ms}})
            yield f"✅ Retrieved {len(chunks)} chunks for query: “{query[:40]}...”"

            for i, chunk in enumerate(chunks, 1):
                doc_id = getattr(chunk, "doc_id", "N/A")
                retriever_name = chunk.meta.get("_retriever", "unknown")
                score = chunk.meta.get("similarity", 0)

                if hasattr(chunk, "description") and not hasattr(chunk, "text"):
                    # ImageChunk
                    preview = chunk.description.strip()[:80].replace("\n", " ")
                    chunk_type = "🖼️ Image"
                else:
                    preview = chunk.text.strip()[:80].replace("\n", " ")
                    chunk_type = chunk.meta.get("doc_type", "text")

                yield (
                    f"[{i}] {chunk_type} | From: {retriever_name} | "
                    f"Score: {score:.3f} | doc_id: {doc_id}"
                )
                yield f"     → {preview}"

        except Exception as e:
            self.logger.error("retrieve.fail", extra={"extra_data": {"error": str(e)}}, exc_info=True)
            yield f"❌ Retrieval failed: {e}"
            # NEW: mark run failure (if run logging already started)
            if self.run_log:
                self.run_log.error("retrieval.fail", extra={"extra_data": {"error": str(e)}})
                self.run_log.info("run.end", extra={"extra_data": {"status": "failed"}})
            raise  # <-- IMPORTANT: bubble up so run_steps() logs step.fail and stops

    def step_ask(
        self,
        query: str = None,
        top_k: int = 5,
        model_name: str = None,
        temperature: float = 0.4,
        max_tokens: int = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generates an answer to the query using the previously retrieved chunks.
        """
        yield "🧠 Starting answer generation..."

        if not query:
            yield "❌ No query provided to step_ask."
            return

        if not self.retrieved_chunks:
            yield "⚠️ No chunks available. Run 'retrieve' first."
            return

        # If model_name is provided in the call, override the current setting
        model_to_use = model_name or self.get_model()
        if max_tokens is None:
            max_tokens = (
                self.config.get("llm", {}).get("max_tokens")  # from config.yml
                or 400  # reasonable fallback
            )

        self._ensure_run_logging()
        t0 = perf_counter()
        self.run_log.info(
            "ask.start",
            extra={"extra_data": {"model": model_to_use, "temperature": temperature, "max_tokens": max_tokens}},
        )

        try:
            prompt_builder = PromptBuilder()
            prompt = prompt_builder.build_prompt(query, context_chunks=self.retrieved_chunks)
            yield f"📜 Prompt built. Sending to model: {model_to_use}..."

            # Persist prompt via the SAME RunLogger
            run_logger = self._run_logger  # type: ignore[assignment]
            run_logger.log_prompt(prompt)  # type: ignore[union-attr]

            completer = OpenAICompleter(model_name=model_to_use)
            answer = completer.get_completion(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens
            )

            # Always log something (even if it's an error message string)
            if answer is not None:
                run_logger.log_response(str(answer))  # type: ignore[union-attr]
            else:
                run_logger.log_response("[ERROR] No answer returned from LLM")  # type: ignore[union-attr]

            self.last_answer = answer

            # Emit end log with duration + basic stats
            elapsed_ms = int((perf_counter() - t0) * 1000)
            self.run_log.info(
                "ask.end",
                extra={"extra_data": {
                    "elapsed_ms": elapsed_ms,
                    "answer_len": (len(answer) if isinstance(answer, str) else 0),
                }},
            )

            # Detect if the returned string is an error message
            if isinstance(answer, str) and answer.startswith("[ERROR]"):
                yield f"❌ LLM call failed: {answer}"
            else:
                yield "✅ Answer received from model."
                yield ""
                yield "💬 Final Answer:"
                yield answer.strip() if isinstance(answer, str) else str(answer)

            # Sources block
            sources = set()
            for chunk in self.retrieved_chunks:
                source_id = chunk.meta.get("source_filepath") or getattr(chunk, "doc_id", None)
                if source_id:
                    sources.add(str(source_id))

            if sources:
                yield ""
                yield "📄 Sources used:"
                for src in sorted(sources):
                    yield f"- {src}"

        except Exception as e:
            self.logger.error("ask.fail", extra={"extra_data": {"error": str(e)}}, exc_info=True)
            if self.run_log:
                self.run_log.error("ask.fail", extra={"extra_data": {"error": str(e)}})
                self.run_log.info("run.end", extra={"extra_data": {"status": "failed"}})
            yield f"❌ Failed to generate answer: {e}"
            raise

    # ----------------------------#
    #         Scenarios           #
    # ----------------------------#

    def run_full_pipeline(self, query: str) -> Iterator[str]:
        """
        Runs a complete RAG pipeline from raw files to answer.
        This includes: ingest → chunk → enrich → embed → retrieve → ask

        Args:
            query (str): The question to answer after processing the corpus.

        Yields:
            str: Progress messages for each step.
        """
        self.clear_steps()
        self.add_step("ingest")
        self.add_step("chunk")
        self.add_step("enrich")
        self.add_step("index_images")
        self.add_step("embed")
        self.add_step("retrieve", query=query)
        self.add_step("ask", query=query)

        yield from self.run_steps()

    def run_query_only(
        self, query: str, strategy: str = "late_fusion", top_k: int = 5, model_name: str = "gpt-4o"
    ) -> Iterator[str]:
        """
        Runs only the retrieval and answer generation steps using existing FAISS + metadata.

        Assumes data is already ingested, chunked, embedded, and indexed.

        Args:
            query (str): The user's natural language question.
            strategy (str): Retrieval strategy (default: 'late_fusion').
            top_k (int): Number of context chunks to retrieve.
            model_name (str): LLM model to use for answering.

        Yields:
            str: Progress messages for each step.
        """
        self.clear_steps()
        self.add_step("retrieve", query=query, strategy=strategy, top_k=top_k)
        self.add_step("ask", query=query, model_name=model_name)

        yield from self.run_steps()
