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
from scripts.agents.email_orchestrator import EmailOrchestratorAgent

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
# ✅ Pipeline Validation
# ─────────────────────────────────────────────
from scripts.pipeline.validator import PipelineValidator

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
    # Disk fallback helpers
    # ─────────────────────────────────────────────
    def _load_chunks_from_disk(self) -> list[Chunk]:
        """
        Load chunks from TSV files on disk.

        Returns:
            List of Chunk objects loaded from chunks_*.tsv files
        """
        chunks = []
        input_path = self.project.get_input_dir()
        chunk_files = list(input_path.glob("chunks_*.tsv"))

        if not chunk_files:
            return []

        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        # Parse metadata JSON
                        meta = json.loads(row.get('meta', '{}'))

                        # Create Chunk object
                        chunk = Chunk(
                            id=row.get('chunk_id', ''),
                            doc_id=row.get('doc_id', ''),
                            text=row.get('text', ''),
                            token_count=int(row.get('token_count', 0)),
                            meta=meta
                        )
                        chunks.append(chunk)

                self.logger.debug(
                    f"Loaded {len(chunks)} chunks from {chunk_file.name}",
                    extra={"chunk_file": str(chunk_file)}
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load chunks from {chunk_file.name}: {e}",
                    extra={"chunk_file": str(chunk_file), "error": str(e)},
                    exc_info=True
                )

        return chunks

    def _count_raw_files(self) -> int:
        """Count raw files in input/raw/ directory."""
        raw_dir = self.project.raw_docs_dir()
        if not raw_dir.exists():
            return 0

        raw_files = [f for f in raw_dir.rglob("*.*")
                    if f.is_file() and not f.name.startswith('.')]
        return len(raw_files)

    def _has_faiss_indices(self) -> bool:
        """Check if FAISS indices exist."""
        faiss_dir = self.project.root_dir / "output" / "faiss"
        if not faiss_dir.exists():
            return False

        indices = list(faiss_dir.glob("*.faiss"))
        return len(indices) > 0

    def _is_email_project(self) -> bool:
        """
        Detect if this is an email project by checking:
        1. Outlook configuration (sources.outlook.enabled = true)
        2. Top-level outlook config (outlook.enabled = true)
        3. Document types (mbox, msg, eml in config)

        Returns:
            True if this is an email project, False otherwise
        """
        # Check sources.outlook.enabled
        sources_config = self.config.get("sources", {})
        outlook_source = sources_config.get("outlook", {})
        if outlook_source.get("enabled", False):
            return True

        # Check top-level outlook config
        outlook_config = self.config.get("outlook", {})
        if outlook_config.get("enabled", False):
            return True

        # Check doc_types for email formats
        doc_types = self.config.get("doc_types", [])
        email_types = {"mbox", "msg", "eml", "email"}
        if any(dt in email_types for dt in doc_types):
            return True

        return False

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

    def run_steps(self, skip_validation: bool = False) -> Iterator[str]:
        """
        Executes all configured pipeline steps in order.
        Yields human-readable progress messages for UI or CLI.

        Args:
            skip_validation: If True, skip pre-flight validation (default: False)
        """
        # Run validation before executing steps (unless skipped)
        if not skip_validation and self.steps:
            yield "\n" + "="*80
            yield "VALIDATING PIPELINE CONFIGURATION"
            yield "="*80 + "\n"

            validator = PipelineValidator(self.project)
            report = validator.validate()

            # Print validation report (captured as string)
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            can_proceed = validator.print_report(report)

            sys.stdout = old_stdout
            validation_output = buffer.getvalue()

            # Yield validation output line by line
            for line in validation_output.split('\n'):
                if line.strip():
                    yield line

            # Block execution if validation failed
            if not can_proceed:
                yield "\n❌ Pipeline validation FAILED - fix errors before proceeding"
                return

            yield "\n✅ Validation passed - proceeding with pipeline...\n"

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

        # Save current state for future validation (unless skipped)
        if not skip_validation:
            validator = PipelineValidator(self.project)
            validator.save_current_state()
            yield "✓ Saved pipeline state for future validation"

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

        # Phase 1 Email Cleaning: Apply quote deduplication and signature detection
        # This must happen BEFORE chunking to preserve newlines for pattern detection
        email_docs = [doc for doc in self.raw_docs if doc.metadata.get('doc_type') == 'outlook_eml']

        if email_docs:
            yield f"🧹 Applying Phase 1 email cleaning to {len(email_docs)} emails..."

            from scripts.email.cleaning import QuoteDeduplicator, SignatureDetector

            quote_dedup = QuoteDeduplicator(
                min_quote_length=50,
                similarity_threshold=0.85
            )
            sig_detector = SignatureDetector(
                min_signature_length=10,
                max_signature_length=500,
                confidence_threshold=0.55  # Lowered to catch legal disclaimers
            )

            total_quote_removed = 0
            total_sig_removed = 0
            emails_cleaned = 0

            for doc in email_docs:
                original_length = len(doc.content)

                # Step 1: Remove quoted reply text
                cleaned_text, quote_stats = quote_dedup.deduplicate(doc.content)
                total_quote_removed += quote_stats.get('removed_chars', 0)

                # Step 2: Remove email signature
                cleaned_text, signature = sig_detector.detect_signature(cleaned_text)
                if signature:
                    total_sig_removed += len(signature)

                # Update document content with cleaned text
                if len(cleaned_text) < original_length:
                    doc.content = cleaned_text
                    emails_cleaned += 1

                    # Update content hash after cleaning
                    hash_base = doc.content.strip()
                    if "image_paths" in doc.metadata:
                        hash_base += ",".join(doc.metadata["image_paths"])
                    doc.metadata["content_hash"] = hashlib.sha256(hash_base.encode("utf-8")).hexdigest()

            if emails_cleaned > 0:
                total_removed = total_quote_removed + total_sig_removed
                reduction_pct = (total_removed / sum(len(d.content) for d in email_docs)) * 100
                yield f"✅ Cleaned {emails_cleaned} emails: removed {total_removed:,} chars ({reduction_pct:.1f}% reduction)"
                self.logger.info(
                    "email_cleaning.complete",
                    extra={
                        "extra_data": {
                            "emails_cleaned": emails_cleaned,
                            "quote_chars_removed": total_quote_removed,
                            "signature_chars_removed": total_sig_removed,
                            "total_removed": total_removed,
                            "reduction_percent": reduction_pct
                        }
                    }
                )

    def step_chunk(self, **kwargs) -> Iterator[str]:
        """
        Applies chunking rules to all raw documents.
        Saves results to chunks_<doc_type>.tsv under the input directory.
        """
        yield "📚 Starting chunking..."

        # Smart fallback: Check disk if no raw_docs in memory
        if not self.raw_docs:
            raw_file_count = self._count_raw_files()

            if raw_file_count > 0:
                yield f"⚠️  No raw documents in memory, but found {raw_file_count} raw file(s) on disk"
                yield "   Running 'ingest' step first..."

                # Run ingest to load raw files
                yield from self.step_ingest()

                # Check again after ingest
                if not self.raw_docs:
                    yield "❌ Ingest completed but no documents were loaded"
                    yield "   Check that raw files are in supported formats"
                    return
            else:
                yield "❌ No raw documents available."
                yield "   Options:"
                yield "   1. Run 'ingest' step first to load raw files"
                yield f"   2. Add files to {self.project.raw_docs_dir()}/ directory"
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

        # Case 2: Load from file (smart fallback)
        base_dir = self.project.input_dir
        enriched_dir = base_dir / "enriched"
        chunk_files = list(base_dir.glob("chunks_*.tsv"))

        if not chunk_files:
            # Check if we have raw files that need chunking
            raw_file_count = self._count_raw_files()

            if raw_file_count > 0:
                yield f"⚠️  No chunk files found, but found {raw_file_count} raw file(s) on disk"
                yield "   Running 'ingest' and 'chunk' steps first..."

                # Run ingest and chunk
                yield from self.step_ingest()
                yield from self.step_chunk()

                # Check again for chunk files
                chunk_files = list(base_dir.glob("chunks_*.tsv"))
                if not chunk_files:
                    yield "❌ Chunking completed but no chunk files were created"
                    return
                # Continue with embedding using newly created chunk files
            else:
                yield "❌ No chunks available for embedding."
                yield "   Options:"
                yield "   1. Run 'ingest' and 'chunk' steps first"
                yield f"   2. Add files to {self.project.raw_docs_dir()}/ directory"
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
        Retrieves top-k results using either EmailOrchestratorAgent (for email projects)
        or RetrievalManager (for non-email projects).
        Stores results in self.retrieved_chunks for step_ask() or inspection.
        """
        yield "🔍 Starting retrieval..."
        self._ensure_run_logging()
        t0 = perf_counter()

        if not query:
            yield "❌ No query provided."
            return

        # Smart fallback: Check if FAISS indices exist
        if not self._has_faiss_indices():
            # Check if we have chunks to embed
            chunk_files = list(self.project.get_input_dir().glob("chunks_*.tsv"))

            if chunk_files:
                yield f"⚠️  No FAISS indices found, but found {len(chunk_files)} chunk file(s) on disk"
                yield "   Running 'embed' step first..."

                # Run embed to create indices
                yield from self.step_embed()

                # Check again
                if not self._has_faiss_indices():
                    yield "❌ Embedding completed but no FAISS indices were created"
                    return
            else:
                # No chunks either - need full pipeline
                raw_file_count = self._count_raw_files()

                if raw_file_count > 0:
                    yield f"⚠️  No FAISS indices or chunks found, but found {raw_file_count} raw file(s)"
                    yield "   Running full pipeline (ingest → chunk → embed)..."

                    yield from self.step_ingest()
                    yield from self.step_chunk()
                    yield from self.step_embed()

                    if not self._has_faiss_indices():
                        yield "❌ Pipeline completed but no FAISS indices were created"
                        return
                else:
                    yield "❌ No FAISS indices available for retrieval."
                    yield "   Options:"
                    yield "   1. Run 'ingest', 'chunk', and 'embed' steps first"
                    yield f"   2. Add files to {self.project.raw_docs_dir()}/ directory"
                    return

        # Detect if this is an email project
        is_email = self._is_email_project()

        try:
            if is_email:
                # Use EmailOrchestratorAgent for email projects (Phases 1-4)
                yield "📧 Detected email project - using EmailOrchestratorAgent..."
                orchestrator = EmailOrchestratorAgent(self.project)

                # Enable LLM fallback for intent detection if configured
                llm_fallback = self.config.get("email", {}).get("llm_intent_fallback", True)
                if llm_fallback:
                    from scripts.agents.email_intent_detector import EmailIntentDetector
                    orchestrator.intent_detector = EmailIntentDetector(
                        use_llm_fallback=True,
                        llm_confidence_threshold=0.6
                    )
                    yield "🧠 LLM-enhanced intent detection enabled"

                # Retrieve using orchestrator (pass None to enable auto top_k adjustment)
                result = orchestrator.retrieve(query, top_k=None, max_tokens=kwargs.get("max_tokens", 2000))

                # Log orchestrator-specific metadata
                self.run_log.info(
                    "retrieval.start",
                    extra={"extra_data": {
                        "query": query,
                        "top_k": top_k,
                        "strategy": result["strategy"]["primary"],
                        "intent": result["intent"]["primary_intent"],
                        "confidence": result["intent"]["confidence"],
                        "detection_method": result["intent"].get("detection_method", "pattern")
                    }},
                )

                # Extract chunks from orchestrator result
                chunks = result["chunks"]
                actual_top_k = result["metadata"].get("chunk_count", len(chunks))

                # Show intent detection results
                yield f"🎯 Detected intent: {result['intent']['primary_intent']} (confidence: {result['intent']['confidence']:.2f})"
                yield f"🔢 Strategy: {result['strategy']['primary']}, Retrieved: {actual_top_k} chunks (auto-adjusted from intent)"

            else:
                # Use standard RetrievalManager for non-email projects
                yield f"📄 Using standard retrieval (strategy: {strategy})..."
                self.run_log.info(
                    "retrieval.start",
                    extra={"extra_data": {"query": query, "top_k": top_k, "strategy": strategy}},
                )

                retriever = RetrievalManager(self.project)
                yield f"🔢 Strategy: {strategy}, Top-K: {top_k}"
                chunks = retriever.retrieve(query=query, top_k=top_k, strategy=strategy)

            # Persist artifacts with the SAME run logger
            run_logger = self._run_logger  # type: ignore[assignment]

            # Determine strategy name for logging
            strategy_name = result["strategy"]["primary"] if is_email and "result" in locals() else strategy

            try:
                run_logger.log_metadata(  # type: ignore[union-attr]
                    {
                        "query": query,
                        "top_k": top_k,
                        "strategy": strategy_name,
                        "timestamp": datetime.now().isoformat(),
                        "pipeline_steps": ["retrieve"],
                        **({"intent": result["intent"], "orchestrator": "email"} if is_email and "result" in locals() else {})
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
                retriever_name = chunk.meta.get("_retriever", "standard")

                # Get score (similarity or relevance)
                score = chunk.meta.get("similarity", chunk.meta.get("relevance", 0))

                if hasattr(chunk, "description") and not hasattr(chunk, "text"):
                    # ImageChunk
                    preview = chunk.description.strip()[:80].replace("\n", " ")
                    chunk_type = "🖼️ Image"
                    yield (
                        f"[{i}] {chunk_type} | Retriever: {retriever_name} | "
                        f"Score: {score:.3f} | doc_id: {doc_id}"
                    )
                else:
                    preview = chunk.text.strip()[:80].replace("\n", " ")
                    chunk_type = chunk.meta.get("doc_type", "text")

                    # For email chunks, show sender name
                    if chunk_type in ["outlook_eml", "msg", "eml", "mbox"]:
                        sender = chunk.meta.get("sender_name", chunk.meta.get("sender", "Unknown"))
                        date = chunk.meta.get("date", "")
                        date_display = f" | Date: {date.split()[0]}" if date else ""

                        yield (
                            f"[{i}] 📧 Email | From: {sender}{date_display} | "
                            f"Retriever: {retriever_name} | Score: {score:.3f}"
                        )
                    else:
                        # Non-email chunks
                        yield (
                            f"[{i}] {chunk_type} | Retriever: {retriever_name} | "
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
            prompt_builder = PromptBuilder(project=self.project, run_id=self._run_id, config=self.config)
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
