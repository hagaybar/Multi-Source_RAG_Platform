from typing import Iterator
from pathlib import Path
import json
import csv
from collections import defaultdict
import hashlib
import uuid

# local imports
from scripts.agents.image_insight_agent import ImageInsightAgent
from scripts.core.project_manager import ProjectManager
from scripts.utils.logger import LoggerManager
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.embeddings.unified_embedder import UnifiedEmbedder
from scripts.utils.chunk_utils import load_chunks
from scripts.chunking.chunker_v3 import split as chunk_text
from scripts.chunking.models import Chunk
from scripts.ingestion.manager import IngestionManager
from scripts.ingestion.models import RawDoc
from scripts.prompting.prompt_builder import PromptBuilder
from scripts.api_clients.openai.completer import OpenAICompleter


class PipelineRunner:
    """
    Orchestrates sequential execution of modular pipeline steps:
    ingest, chunk, enrich, embed, index.
    """

    def __init__(self, project: ProjectManager, config: dict):
        self.project = project
        self.config = config
        self.steps: list[tuple[str, dict]] = []
        self.logger = LoggerManager.get_logger(
            "PipelineRunner", log_file=project.get_log_path("pipeline")
        )
        self.raw_docs: list[RawDoc] = []  # ← Store output of ingest
        self.seen_hashes: set[str] = set()  # ← Optional deduplication base
        self.chunks: list[Chunk] = []
        self.retrieved_chunks = []
        self.last_answer = None

    def add_step(self, name: str, **kwargs) -> None:
        """
        Adds a step by name, with optional keyword arguments.
        Steps must match a method named `step_<name>`.
        """
        if not hasattr(self, f"step_{name}"):
            raise ValueError(f"Step '{name}' not implemented.")
        self.steps.append((name, kwargs))
        self.logger.info(f"Step added: {name} {kwargs}")

    def clear_steps(self) -> None:
        """
        Clears all steps from the pipeline.
        """
        self.steps.clear()
        self.logger.info("All steps cleared from pipeline.")

    def run_steps(self) -> Iterator[str]:
        """
        Runs all configured steps in order. Yields status messages for UI or CLI.
        """
        self.logger.info("Running pipeline steps...")
        yield "🚀 Starting pipeline execution..."

        for name, kwargs in self.steps:
            # Get the step method with proper error handling
            step_method_name = f"step_{name}"
            if not hasattr(self, step_method_name):
                error_msg = f"Step method '{step_method_name}' not found"
                self.logger.error(error_msg)
                yield f"❌ {error_msg}"
                raise AttributeError(error_msg)

            step_fn = getattr(self, step_method_name)

            # Verify it's actually callable
            if not callable(step_fn):
                error_msg = f"Step '{step_method_name}' is not callable"
                self.logger.error(error_msg)
                yield f"❌ {error_msg}"
                raise TypeError(error_msg)

            yield f"▶️ Running step: {name}"
            self.logger.info(f"Running step: {name} with args: {kwargs}")

            try:
                result = step_fn(**kwargs)
                if isinstance(result, Iterator):
                    for msg in result:
                        yield msg
                else:
                    yield f"✅ Step '{name}' completed."
                self.logger.info(f"Step '{name}' completed successfully.")
            except Exception as e:
                self.logger.exception(f"Exception in step '{name}': {e}")
                yield f"❌ Step '{name}' failed: {e}"
                raise

        yield "🏁 Pipeline finished."

    # ----------------------------#
    #           Steps             #
    # ----------------------------#

    def step_ingest(self, path: Path = None, **kwargs) -> Iterator[str]:
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

        # Optional: hash-based deduplication (placeholder)
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
                self.logger.info(f"Duplicate skipped: {doc.metadata.get('source_filepath')}")

        self.raw_docs = new_docs
        yield f"✅ Ingested {len(new_docs)} unique documents from {path.name}"

    def step_chunk(self, **kwargs) -> Iterator[str]:
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
            # Debug info logged instead of printed
            self.logger.debug(
                f"Processing doc_id: {doc_id}, paragraph: {meta.get('paragraph_number')}, "
                f"image_paths: {meta.get('image_paths')}"
            )

            try:
                if "image_paths" in meta:
                    self.logger.debug(f"Passing image_paths for {doc_id}: {meta['image_paths']}")
                    self.logger.debug(
                        f"Paragraph {meta.get('paragraph_number')} - images: {meta['image_paths']}"
                    )
                chunks = chunk_text(doc.content, meta)
                all_chunks.extend(chunks)
                yield f"✂️ {len(chunks)} chunks from {doc_type.upper()} document: {doc_id}"
            except Exception as e:
                yield f"❌ Error chunking {doc_id}: {e}"
                self.logger.warning(f"Chunking failed for {doc_id}: {e}")

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
                yield f"❌ Failed to write chunks_{doc_type}.tsv: {e}"

        yield f"✅ Chunking complete. Total chunks: {len(all_chunks)}"

    def step_enrich(self, overwrite: bool = False, **kwargs) -> Iterator[str]:
        # ─── Initial notice & in‐memory debug ───
        yield "🧠 Starting image enrichment..."
        yield f"🐞 DEBUG: runner has {len(self.chunks)} chunks in memory before enrichment"

        # ─── Fallback to load from disk if none in memory ───

        if not self.chunks:
            # 1) List chunk files on disk
            chunk_glob = str(self.project.input_dir / "chunks_*.tsv")
            chunk_paths = list(self.project.input_dir.glob("chunks_*.tsv"))
            yield f"🐞 DEBUG: looking for chunk files with glob `{chunk_glob}`"
            yield (
                f"🐞 DEBUG: found {len(chunk_paths)} chunk file(s): {[p.name for p in chunk_paths]}"
            )

            if chunk_paths:
                loaded = 0
                for path in chunk_paths:
                    for chk in load_chunks(path):
                        self.chunks.append(chk)
                        loaded += 1
                yield f"🔄 Loaded {loaded} chunks from disk"
            else:
                yield "❌ No chunks available on disk. Please run 'chunk' first."
                return

        agent = ImageInsightAgent(self.project)
        enriched_chunks: list[Chunk] = []

        count_total = 0
        count_enriched = 0

        for chunk in self.chunks:
            count_total += 1

            # Skip chunks with no images
            img_list = chunk.meta.get("image_paths") or []
            if not img_list:
                enriched_chunks.append(chunk)
                continue

            try:
                # For each image in the list, call the agent and collect results
                all_results: list[Chunk] = []
                for img_path in img_list:
                    # create a temp chunk carrying one image at a time
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
                    result_list = result if isinstance(result, list) else [result]
                    all_results.extend(result_list)

                # if the agent returned nothing, fall back to the original chunk
                if not all_results:
                    enriched_chunks.append(chunk)
                else:
                    enriched_chunks.extend(all_results)
                    count_enriched += 1

                yield f"🖼️ Enriched {len(img_list)} image(s) in chunk: {chunk.id}"

            except Exception as e:
                self.logger.warning(f"Image enrichment failed for chunk {chunk.id}: {e}")
                enriched_chunks.append(chunk)
                yield f"⚠️ Failed to enrich chunk {chunk.id}: {e}"

        # Replace the in-memory chunk list with enriched versions
        self.chunks = enriched_chunks

        # Save by doc_type
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

        yield f"✅ Enrichment complete: {count_enriched}/{count_total} chunks enriched"

    def step_index_images(self, doc_types: list[str] = None, **kwargs) -> Iterator[str]:
        """
        Index enriched image descriptions into FAISS and metadata JSONL.
        """
        from scripts.chunking.models import ImageChunk
        from scripts.embeddings.image_indexer import ImageIndexer

        doc_types = doc_types or ["pptx", "pdf", "docx"]
        enriched_dir = self.project.input_dir / "enriched"
        indexer = ImageIndexer(self.project)

        count_total = 0

        for doc_type in doc_types:
            file_path = enriched_dir / f"chunks_{doc_type}.tsv"
            if not file_path.exists():
                yield f"⚠️ Skipping {doc_type} — no enriched chunks found."
                continue

            image_chunks = []
            with open(file_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                for row in reader:
                    if len(row) < 5:
                        continue
                    meta = json.loads(row[4])
                    summaries = meta.get("image_summaries", [])
                    for summary in summaries:
                        image_chunks.append(
                            ImageChunk(
                                id=str(uuid.uuid4()),
                                description=summary["description"],
                                meta={
                                    "image_path": summary["image_path"],
                                    "source_chunk_id": row[0],
                                    "doc_type": meta.get("doc_type"),
                                    "source_filepath": meta.get("source_filepath"),
                                    "page_number": meta.get("page_number"),
                                },
                            )
                        )

            if image_chunks:
                indexer.run(image_chunks)
                count_total += len(image_chunks)
                yield f"✅ Indexed {len(image_chunks)} image chunks for {doc_type}."
            else:
                yield f"⚠️ No image summaries found in {file_path.name}."

        yield f"🧠 Image indexing complete. Total: {count_total}"

    def step_embed(self, **kwargs) -> Iterator[str]:
        yield "🧬 Starting embedding step..."

        embed_config = self.config.get("embedding", {})
        image_enrichment_enabled = embed_config.get("image_enrichment", False)
        use_async = embed_config.get("use_async_batch", False)

        base_dir = self.project.input_dir
        enriched_dir = base_dir / "enriched"
        chunk_files = list(base_dir.glob("chunks_*.tsv"))

        if not chunk_files:
            yield "❌ No chunk files found in input/. Run 'chunk' first."
            return

        embedder = UnifiedEmbedder(self.project, runtime_config=self.config)
        yield f"⚙️ Embedding mode: {'async-batch' if use_async else 'local/batch'}"

        for chunk_path in chunk_files:
            doc_type = chunk_path.stem.split("_", 1)[-1]
            enriched_path = enriched_dir / f"chunks_{doc_type}.tsv"

            # Use enriched if allowed and available
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
                self.logger.error(f"Embedding failed for {doc_type}: {e}", exc_info=True)

        yield "📦 Embedding complete for all doc types."

    def step_retrieve(
        self, query: str, top_k: int = 5, strategy: str = "late_fusion", **kwargs
    ) -> Iterator[str]:
        yield "🔍 Starting retrieval..."
        if not query:
            yield "❌ No query provided."
            return

        try:
            retriever = RetrievalManager(self.project)
            yield f"🔢 Strategy: {strategy}, Top-K: {top_k}"
            chunks = retriever.retrieve(query=query, top_k=top_k, strategy=strategy)

            if not chunks:
                yield "⚠️ No results retrieved."
                return

            self.retrieved_chunks = chunks  # Store for step_ask()
            yield f"✅ Retrieved {len(chunks)} chunks for query: “{query[:40]}...”"

            for i, chunk in enumerate(chunks, 1):
                doc_id = chunk.doc_id
                source = chunk.meta.get("source_filepath", "N/A")
                sim = chunk.meta.get("similarity", 0)
                preview = chunk.text.strip()[:80].replace("\n", " ")
                yield f"[{i}] 📄 {doc_id} (score={sim:.3f}) → {preview}"

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
        **kwargs,
    ) -> Iterator[str]:
        yield "🧠 Starting answer generation..."

        if not query:
            yield "❌ No query provided to step_ask."
            return

        if not self.retrieved_chunks:
            yield "⚠️ No chunks available. Run 'retrieve' first."
            return

        try:
            prompt_builder = PromptBuilder()  # uses default template
            prompt = prompt_builder.build_prompt(query, context_chunks=self.retrieved_chunks)
            yield f"📜 Prompt built. Sending to model: {model_name}..."

            completer = OpenAICompleter(model_name=model_name)
            answer = completer.get_completion(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens
            )

            self.last_answer = answer
            yield "✅ Answer received from model."
            yield ""
            yield "💬 Final Answer:"
            yield answer.strip()

            # Optional: print sources
            sources = {
                chunk.meta.get("source_filepath", chunk.doc_id) for chunk in self.retrieved_chunks
            }
            if sources:
                yield ""
                yield "📄 Sources used:"
                for src in sorted(sources):
                    yield f"- {src}"

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}", exc_info=True)
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
