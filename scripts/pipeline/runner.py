# ─────────────────────────────────────────────
# 🔧 Standard Library Imports
# ─────────────────────────────────────────────
from datetime import datetime
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

# ─────────────────────────────────────────────


class PipelineRunner:
    """
    Orchestrates sequential execution of modular pipeline steps
    (ingest, chunk, enrich, embed, index).
    """

    def __init__(self, project: ProjectManager, config: dict):
        self.project = project
        self.config = config
        self.steps: list[tuple[str, dict]] = []
        self.logger = LoggerManager.get_logger(
            "PipelineRunner",
            log_file=project.get_log_path("pipeline")
        )
        self.raw_docs: list[RawDoc] = []  # ← Store output of ingest
        self.seen_hashes: set[str] = set()  # ← Optional deduplication base
        self.chunks: list[Chunk] = []
        self.retrieved_chunks = []
        self.last_answer = None


    def add_step(self, name: str, **kwargs) -> None:
        """
        Adds a step to the pipeline by name, with optional keyword arguments.
        The step must have a corresponding method: `step_<name>()`.
        """
        method_name = f"step_{name}"
        if not hasattr(self, method_name):
            raise ValueError(f"Step '{name}' is not implemented (missing method: {method_name})")

        self.steps.append((name, kwargs))
        self.logger.info("Step added: %s %s", name, kwargs)

    def run_steps(self) -> Iterator[str]:
        """
        Executes all configured pipeline steps in order.
        Yields human-readable progress messages for UI or CLI.
        """
        self.logger.info("Starting pipeline execution...")
        yield "🚀 Starting pipeline execution..."

        for name, kwargs in self.steps:
            method_name = f"step_{name}"
            yield f"▶️ Running step: {name}"
            self.logger.info("Running step: %s with args: %s", name, kwargs)

            try:
                step_fn: Callable = getattr(self, method_name)
                if not callable(step_fn):
                    raise AttributeError(f"'{method_name}' is not callable.")

                result = step_fn(**kwargs)

                if isinstance(result, Iterator):
                    yield from result
                else:
                    yield f"✅ Step '{name}' completed."

                self.logger.info("Step '%s' completed.", name)

            except Exception as e:
                self.logger.error("Step '%s' failed: %s", name, e, exc_info=True)
                yield f"❌ Step '{name}' failed: {e}"
                raise

        yield "🏁 Pipeline finished."

    def clear_steps(self) -> None:
        """
        Clears all steps from the pipeline.
        Useful before re-running or resetting the workflow.
        """
        self.steps.clear()
        self.logger.info("All steps cleared from pipeline.")

    # ----------------------------#
    #           Steps             #
    # ----------------------------#

    def step_ingest(self, path: Path = None, **kwargs) -> Iterator[str]:
        """
        Ingests raw documents from the given path or project input/raw directory.
        Applies optional deduplication by content hash (including image references).
        """
        yield "📥 Starting ingestion..."

        ingestion_manager = IngestionManager(log_file=self.project.get_log_path("ingestion"))
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
                self.logger.info("Duplicate skipped: %s", doc.metadata.get("source_filepath"))

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

        for i, doc in enumerate(self.raw_docs):
            doc_id = doc.metadata.get("source_filepath", f"doc_{i}")
            doc_type = doc.metadata.get("doc_type", "default")
            if not doc_type:
                yield f"⚠️ Skipping doc with missing doc_type: {doc_id}"
                continue

            meta = doc.metadata.copy()
            meta["doc_id"] = doc_id

            # Optional debug
            self.logger.debug("Chunking doc_id: %s, paragraph: %s, image_paths: %s",
                            doc_id, meta.get("paragraph_number"), meta.get("image_paths"))

            try:
                chunks = chunk_text(doc.content, meta)
                all_chunks.extend(chunks)
                yield f"✂️ {len(chunks)} chunks from {doc_type.upper()} document: {doc_id}"
            except Exception as e:
                yield f"❌ Error chunking {doc_id}: {e}"
                self.logger.warning("Chunking failed for %s: %s", doc_id, e)

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
                        writer.writerow([
                            chunk.id,
                            chunk.doc_id,
                            chunk.text,
                            chunk.token_count,
                            json.dumps(chunk.meta)
                        ])
                yield f"💾 Saved {len(chunks)} chunks to: {chunk_path.name}"
            except Exception as e:
                yield f"❌ Failed to write chunks_{doc_type}.tsv: {e}"
                self.logger.error("Failed to write chunks for %s: %s", doc_type, e)

        yield f"✅ Chunking complete. Total chunks: {len(all_chunks)}"

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
                self.logger.error("Embedding failed for in-memory chunks: %s", e, exc_info=True)
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
                enriched_path
                if image_enrichment_enabled and enriched_path.exists()
                else chunk_path
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
                self.logger.error("Embedding failed for %s: %s", doc_type, e, exc_info=True)

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
                        meta=temp_meta
                    )

                    result = agent.run(temp_chunk, self.project)
                    all_results.extend(result if isinstance(result, list) else [result])

                enriched_chunks.extend(all_results if all_results else [chunk])
                if all_results:
                    count_enriched += 1
                yield f"🖼️ Enriched {len(img_list)} image(s) in chunk: {chunk.id}"

            except Exception as e:
                self.logger.warning("Image enrichment failed for chunk %s: %s", chunk.id, e)
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
                        writer.writerow([
                            chunk.id,
                            chunk.doc_id,
                            chunk.text,
                            chunk.token_count,
                            json.dumps(chunk.meta)
                        ])
                yield f"💾 Saved enriched chunks to: {save_path.name}"
            except Exception as e:
                yield f"❌ Failed to write enriched file: {e}"
                self.logger.error("Failed to save enriched chunks for %s: %s", doc_type, e)

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
                self.logger.warning("Failed to read existing image metadata: %s", e)

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
                    header = next(reader)

                    for row in reader:
                        if len(row) < 5:
                            continue
                        meta = json.loads(row[4])
                        summaries = meta.get("image_summaries", [])

                        for summary in summaries:
                            description = summary["description"]
                            if not description or not isinstance(description, str):
                                self.logger.warning(
                                    "Skipping image with empty or invalid description."
                                )
                                continue
                            img_hash = hashlib.sha256(description.strip().encode("utf-8")).hexdigest()

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
                self.logger.error("Failed to process %s: %s", file_path, e, exc_info=True)
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
                self.logger.error("Indexing failed for %s: %s", doc_type, e, exc_info=True)
                yield f"❌ Indexing failed for {doc_type}: {e}"

        if count_total:
            yield f"🧠 Image indexing complete. Total indexed: {count_total}, skipped: {count_skipped}"
        else:
            yield f"⚠️ No new image chunks indexed. {count_skipped} duplicates skipped."

    def step_retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "late_fusion",
        **kwargs
    ) -> Iterator[str]:
        """
        Retrieves top-k results (text + image-aware) using late fusion.
        Stores results in self.retrieved_chunks for step_ask() or inspection.
        """
        yield "🔍 Starting retrieval..."
        if not query:
            yield "❌ No query provided."
            return

        try:
            retriever = RetrievalManager(self.project)
            yield f"🔢 Strategy: {strategy}, Top-K: {top_k}"
            chunks = retriever.retrieve(query=query, top_k=top_k, strategy=strategy)
            # ---- LOGGING ----
            try:
                run_logger = RunLogger(self.project.root_dir)
                run_logger.log_metadata({
                    "query": query,
                    "top_k": top_k,
                    "strategy": strategy,
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_steps": ["retrieve"]
                })
                run_logger.log_chunks(chunks)

                # Optional: detect and log image matches
                image_chunks = [c for c in chunks if getattr(c, "description", None) and "image_path" in c.meta]
                if image_chunks:
                    from scripts.chunking.models import ImageChunk
                    run_logger.log_images(image_chunks)  # cast is safe due to structure
            except Exception as e:
                self.logger.warning(f"RunLogger failed in step_retrieve: {e}")
            
            # ---- LOGGING ends ----

            if not chunks:
                yield "⚠️ No results retrieved."
                return

            self.retrieved_chunks = chunks
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
            self.logger.error(f"Retrieval failed: {e}", exc_info=True)
            yield f"❌ Retrieval failed: {e}"


    def step_ask(
        self,
        query: str = None,
        top_k: int = 5,
        model_name: str = "gpt-4o",
        temperature: float = 0.4,
        max_tokens: int = 500,
        **kwargs
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

        try:
            prompt_builder = PromptBuilder()
            prompt = prompt_builder.build_prompt(query, context_chunks=self.retrieved_chunks)
            yield f"📜 Prompt built. Sending to model: {model_name}..."

            # Prepare RunLogger
            run_logger = RunLogger(self.project.root_dir)  # same timestamp folder
            run_logger.log_prompt(prompt)

            completer = OpenAICompleter(model_name=model_name)
            answer = completer.get_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            run_logger.log_response(answer)

            # Also update metadata
            run_logger.log_metadata({
                "query": query,
                "top_k": top_k,
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": datetime.now().isoformat(),
                "pipeline_steps": ["retrieve", "ask"] if self.retrieved_chunks else ["ask"]
            })


            self.last_answer = answer
            yield "✅ Answer received from model."
            yield ""
            yield "💬 Final Answer:"
            yield answer.strip()

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
            self.logger.error("Answer generation failed: %s", e, exc_info=True)
            yield f"❌ Failed to generate answer: {e}"


    # ----------------------------#
    #         secenarios          #
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
        self,
        query: str,
        strategy: str = "late_fusion",
        top_k: int = 5,
        model_name: str = "gpt-4o"
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
