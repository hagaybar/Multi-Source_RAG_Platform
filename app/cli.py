# Workaround for OpenMP runtime conflict on Windows (libomp vs. libiomp)
# See: https://github.com/pytorch/pytorch/issues/37377 and https://openmp.llvm.org
import sys
import pathlib
import uuid

# Ensure the root directory (where pyproject.toml lives) is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from shutil import copy
import copy as copy_module
import logging  # Added for ask command

import typer  # type: ignore
import json, csv
from collections import defaultdict
from pathlib import Path

from scripts.ingestion.manager import IngestionManager
from scripts.chunking.chunker_v3 import split as chunker_split
from scripts.chunking.models import Chunk

# from scripts.embeddings.chunk_embedder import ChunkEmbedder
from scripts.embeddings.unified_embedder import UnifiedEmbedder
from scripts.utils.logger import LoggerManager
from scripts.utils.task_paths import TaskPaths
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.prompting.prompt_builder import PromptBuilder  # Added for ask command
from scripts.api_clients.openai.completer import (
    OpenAICompleter,  # Added for ask command
)
from scripts.agents.image_insight_agent import (
    ImageInsightAgent,  # Added for index_images command
)
from scripts.utils.run_logger import RunLogger

app = typer.Typer()

# Setup basic logging for the CLI
paths = TaskPaths()

cli_logger = LoggerManager.get_logger(
    name="cli",             # or "ui", "retrieval", "pipeline", etc.
    level="DEBUG",
    task_paths=paths,
    run_id=None,            # leave None for long-lived processes
    use_json=True           # JSON for file logs, color for console
)


@app.command()
def ingest(
    folder_path: pathlib.Path = typer.Argument(
        ..., help="Path to the folder to ingest."
    ),
    chunk: bool = typer.Option(
        False, "--chunk", help="Enable chunking of ingested documents."
    ),
):
    """
    Ingests documents from the specified folder and optionally chunks them.
    """
    # Generate run_id for this command
    run_logger = RunLogger(folder_path)
    run_id = run_logger.base_dir.name
    
    project = ProjectManager(folder_path)
    ingestion_manager = IngestionManager(
        log_file=str(project.get_log_path("ingestion")),
        run_id=run_id
    )
    chunker_logger = LoggerManager.get_logger(
        "chunker_project", log_file=str(project.get_log_path("chunker")), run_id=run_id
    )

    chunker_logger.info(f"Starting ingestion from folder: {folder_path}", extra={"run_id": run_id, "command": "ingest", "folder_path": str(folder_path)})
    if not folder_path.is_dir():
        chunker_logger.error(f"Error: {folder_path} is not a valid directory.")
        raise typer.Exit(code=1)

        # Debug logging for chunker setup
    chunker_logger.debug(f"Chunker log path: {project.get_log_path('chunker')}", extra={"run_id": run_id})
    chunker_logger.debug(
        f"Chunker log path as string: {str(project.get_log_path('chunker'))}", extra={"run_id": run_id}
    )
    chunker_logger.debug("Checking chunker logger handlers...", extra={"run_id": run_id})
    for handler in chunker_logger.handlers:
        if hasattr(handler, 'baseFilename'):
            chunker_logger.debug(
                f"Chunker FileHandler baseFilename: {handler.baseFilename}", extra={"run_id": run_id, "handler_type": "FileHandler"}
            )
    raw_docs = ingestion_manager.ingest_path(folder_path)

    # Changed "documents" to "text segments"
    chunker_logger.info(f"Ingested {len(raw_docs)} text segments from {folder_path}", extra={"run_id": run_id, "segment_count": len(raw_docs), "folder_path": str(folder_path)})

    if chunk:
        chunker_logger.info("Chunking is enabled. Proceeding with chunking...", extra={"run_id": run_id})
        if not raw_docs:
            chunker_logger.info("No documents were ingested, skipping chunking.", extra={"run_id": run_id})
            raise typer.Exit()

        chunker_logger.info("Chunking ingested documents...", extra={"run_id": run_id})
        all_chunks: list[Chunk] = []

        for raw_doc in raw_docs:
            # Ensure doc_id is properly assigned for chunking
            # RawDoc.metadata should contain 'source_filepath'
            doc_id = raw_doc.metadata.get('source_filepath', 'unknown_document')
            chunker_logger.debug(
                f"Processing document: {raw_doc.metadata.get('source_filepath')}",
                extra={"run_id": run_id, "source_filepath": raw_doc.metadata.get('source_filepath')}
            )
            if not raw_doc.metadata.get('doc_type'):
                chunker_logger.warning(
                    f"doc_type missing in metadata for {doc_id}",
                    extra={"run_id": run_id, "doc_id": doc_id, "content_preview": raw_doc.content[:100]}
                )
                # Potentially skip or assign default doc_type
                # BaseChunker will raise error if doc_type is missing.

            try:
                # Ensure raw_doc.metadata contains 'doc_id' as expected by
                # chunker_v3.py.
                # The 'doc_id' key should ideally be populated by the
                # IngestionManager or here if not.
                # For now, we rely on 'source_filepath' being in metadata and
                # chunker_v3 using meta.get('doc_id').
                # Let's ensure 'doc_id' is explicitly set in the metadata
                # passed to the chunker for clarity.
                current_meta = raw_doc.metadata.copy()
                current_meta['doc_id'] = (
                    doc_id  # doc_id is from
                    # raw_doc.metadata.get('source_filepath', ...)
                )

                document_chunks = chunker_split(
                    text=raw_doc.content,
                    meta=current_meta,
                    logger=chunker_logger,
                    # clean_options will use default from chunker_v3.split
                )
                chunker_logger.debug(
                    f"Chunked document: {raw_doc.metadata.get('source_filepath')} => "
                    f"{raw_doc.metadata.get('doc_type')}",
                    extra={"run_id": run_id, "source_filepath": raw_doc.metadata.get('source_filepath'), "doc_type": raw_doc.metadata.get('doc_type')}
                )
                all_chunks.extend(document_chunks)

            except ValueError as e:
                error_msg = (
                    f"Skipping chunking for a segment from {doc_id} due to error: {e}"
                )
                chunker_logger.error(error_msg, extra={"run_id": run_id, "doc_id": doc_id}, exc_info=True)
            except Exception as e:
                error_msg = (
                    f"An unexpected error occurred while chunking a segment "
                    f"from {doc_id}: {e}"
                )
                chunker_logger.error(error_msg, extra={"run_id": run_id, "doc_id": doc_id}, exc_info=True)

        chunker_logger.info(f"Generated {len(all_chunks)} chunks.", extra={"run_id": run_id, "total_chunks": len(all_chunks)})

        if all_chunks:
            # Group chunks by doc_type
            doc_type_map = defaultdict(list)
            for chk in all_chunks:
                doc_type = chk.meta.get("doc_type", "default")
                doc_type_map[doc_type].append(chk)

            for doc_type, chunks_list in doc_type_map.items():
                output_path = folder_path / "input" / f"chunks_{doc_type}.tsv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(
                        output_path, "w", newline="", encoding="utf-8"
                    ) as tsvfile:
                        writer = csv.writer(tsvfile, delimiter="\t")
                        header = [
                            'chunk_id', 'doc_id', 'text', 'token_count', 'meta_json'
                        ]
                        writer.writerow(header)
                        for chk_item in chunks_list:  # Renamed variable to
                            # avoid conflict
                            meta_json_str = json.dumps(chk_item.meta)
                            writer.writerow(
                                [
                                    chk_item.id,
                                    chk_item.doc_id,
                                    chk_item.text,
                                    chk_item.token_count,
                                    meta_json_str,
                                ]
                            )
                    chunker_logger.info(f"Wrote {len(chunks_list)} chunks to {output_path.name}", extra={"run_id": run_id, "doc_type": doc_type, "chunk_count": len(chunks_list), "output_file": output_path.name})
                except IOError as e:
                    error_msg = f"Error writing chunks for {doc_type}: {e}"
                    chunker_logger.error(error_msg)  # Use error level
                    raise typer.Exit(code=1)
        else:
            chunker_logger.info("No chunks were generated.", extra={"run_id": run_id})


@app.command()
def embed(
    project_dir: Path,
    use_async: bool = typer.Option(
        False, "--a-b", "--async-batch", help="Use OpenAI async batch embedding"
    ),
    with_image_index: bool = typer.Option(
        False,
        "--with-image-index",
        help="Run image enrichment and indexing after embedding"
    ),
) -> None:
    """
    Generate embeddings for chunks in the specified project directory.
    Optionally run image enrichment + indexing after embedding.
    """
    # Generate run_id for this command
    run_logger = RunLogger(project_dir)
    run_id = run_logger.base_dir.name
    
    cli_logger.debug("CLI embed() command starting", extra={"run_id": run_id, "command": "embed"})

    cli_logger.debug("CLI embed arguments received", extra={
        "run_id": run_id,
        "project_dir": str(project_dir),
        "use_async": use_async,
        "with_image_index": with_image_index
    })

    if not project_dir.exists():
        typer.echo(f"❌ Project directory does not exist: {project_dir}")
        raise typer.Exit(1)

    project = ProjectManager(project_dir)
    runtime_config = copy_module.deepcopy(project.config)

    if use_async:
        runtime_config.setdefault("embedding", {})["use_async_batch"] = True

    embedder = UnifiedEmbedder(project, runtime_config=runtime_config)
    embedder.run_from_folder()

    cli_logger.info("✅ Embedding complete.", extra={"run_id": run_id})

    # Optional post-processing: image enrichment and indexing
    if with_image_index:
        cli_logger.info("🧠 Starting image enrichment + indexing...", extra={"run_id": run_id})

        import subprocess

        doc_types = ["pptx", "pdf", "docx"]  # You can extend this as needed

        for doc_type in doc_types:
            enrich_cmd = (
                f"python cli.py enrich-images {project_dir} --doc-type {doc_type}"
            )
            index_cmd = (
                f"python cli.py index-images {project_dir} --doc-type {doc_type}"
            )

            cli_logger.debug(f"Running enrichment command", extra={"run_id": run_id, "doc_type": doc_type, "command": enrich_cmd})
            subprocess.call(enrich_cmd, shell=True)

            cli_logger.debug(f"Running index command", extra={"run_id": run_id, "doc_type": doc_type, "command": index_cmd})
            subprocess.call(index_cmd, shell=True)

        cli_logger.info("✅ Image indexing complete.", extra={"run_id": run_id})

    cli_logger.debug("CLI embed() command complete", extra={"run_id": run_id})


@app.command()
def retrieve(
    project_path: str = typer.Argument(..., help="Path to the RAG project directory"),
    query: str = typer.Argument(..., help="Search query string"),
    top_k: int = typer.Option(10, help="Number of top chunks to return"),
    strategy: str = typer.Option("late_fusion", help="Retrieval strategy to use"),
):
    """
    Retrieve top-k chunks from multiple document types using the configured strategy.
    """
    # Generate run_id for this command
    run_logger = RunLogger(Path(project_path))
    run_id = run_logger.base_dir.name
    
    cli_logger.info(f"Starting retrieval for project: {project_path}, query: '[QUERY_REDACTED]'", extra={
        "run_id": run_id,
        "command": "retrieve",
        "project_path": project_path,
        "top_k": top_k,
        "strategy": strategy,
        "query_length": len(query)
    })
    project = ProjectManager(project_path)
    rm = RetrievalManager(project, run_id=run_id)

    results = rm.retrieve(query=query, top_k=top_k, strategy=strategy)

    print(f"\n--- Top {len(results)} results for query: '{query}' ---")
    if not results:
        print("No results found.")
    for i, chunk_item in enumerate(results, 1):  # Renamed variable
        print(
            f"\n[{i}] From {chunk_item.meta.get('_retriever')} | "
            f"score: {chunk_item.meta.get('similarity', 0):.3f} | "
            f"doc_id: {chunk_item.doc_id}"
        )
        source_filepath = chunk_item.meta.get('source_filepath', 'N/A')
        print(f"    Source File: {source_filepath}")
        page_number = chunk_item.meta.get('page_number')
        if page_number:
            print(f"    Page: {page_number}")
        print(f"    Text: {chunk_item.text.strip()[:500]}...")
    cli_logger.info(f"Retrieved {len(results)} chunks.", extra={"run_id": run_id, "result_count": len(results)})


@app.command()
def ask(
    project_path: str = typer.Argument(..., help="Path to the RAG project directory."),
    query: str = typer.Argument(..., help="Your question to the RAG system."),
    top_k: int = typer.Option(5, help="Number of context chunks to retrieve."),
    temperature: float = typer.Option(
        0.7, help="LLM temperature for response generation."
    ),
    max_tokens: int = typer.Option(500, help="LLM maximum tokens for the response."),
    model_name: str = typer.Option(
        "gpt-3.5-turbo",
        help="OpenAI model to use for generating the answer (via LiteLLM)."
    ),
):
    """
    Asks a question to the RAG system.

    This command retrieves relevant context chunks from the indexed documents
    in the specified project, then uses an LLM (via LiteLLM, configured for OpenAI)
    to generate an answer based on your query and the retrieved context.

    Requires the OPENAI_API_KEY environment variable to be set for LLM access.
    """
    # Generate run_id for this command
    run_logger = RunLogger(Path(project_path))
    run_id = run_logger.base_dir.name
    
    cli_logger.info(f"Starting 'ask' command for project: {project_path}", extra={
        "run_id": run_id,
        "command": "ask",
        "project_path": project_path,
        "query_length": len(query),
        "top_k": top_k,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens
    })

    try:
        project = ProjectManager(project_path)
        cli_logger.debug(f"ProjectManager initialized for {project.root_dir}", extra={"run_id": run_id, "project_root": str(project.root_dir)})

        # 1. Retrieve context
        retrieval_manager = RetrievalManager(project, run_id=run_id)
        cli_logger.debug(
            f"RetrievalManager initialized. Retrieving top {top_k} chunks for query...",
            extra={"run_id": run_id, "top_k": top_k}
        )
        retrieved_chunks = retrieval_manager.retrieve(query=query, top_k=top_k)

        if not retrieved_chunks:
            cli_logger.warning(
                "No context chunks retrieved. Answering based on query alone might "
                "be difficult or impossible.",
                extra={"run_id": run_id}
            )
            print("\nWarning: No relevant context documents were found for your query.")
            # Decide if to proceed or exit. For now, proceed, LMM will be told
            # context is empty.
        else:
            cli_logger.info(f"Retrieved {len(retrieved_chunks)} chunks.", extra={"run_id": run_id, "chunk_count": len(retrieved_chunks)})
            print(f"\n--- Retrieved {len(retrieved_chunks)} context chunks ---")
            for i, chunk_item in enumerate(retrieved_chunks, 1):
                source_id = chunk_item.meta.get('source_filepath', chunk_item.doc_id)
                page_info = (
                    f", page {chunk_item.meta.get('page_number')}"
                    if chunk_item.meta.get('page_number')
                    else ""
                )
                print(
                    f"  [{i}] Source: {source_id}{page_info} "
                    f"(Score: {chunk_item.meta.get('similarity', 0):.3f})"
                )
                # print(f"      Text: {chunk_item.text[:100].strip()}...")
                # Optional: print chunk text preview

        # 2. Build prompt
        prompt_builder = PromptBuilder(project=project, run_id=run_id)  # Use project-specific logging
        cli_logger.debug("PromptBuilder initialized.", extra={"run_id": run_id})
        prompt_str = prompt_builder.build_prompt(
            query=query, context_chunks=retrieved_chunks
        )
        cli_logger.debug(f"Prompt built. Length: {len(prompt_str)} chars.", extra={"run_id": run_id, "prompt_length": len(prompt_str)})
        # cli_logger.debug(f"Generated Prompt:\n{prompt_str}") # Potentially very long

        # 3. Get LMM completion
        # API key for OpenAICompleter is handled internally
        # (expects OPENAI_API_KEY env var)
        try:
            completer = OpenAICompleter(model_name=model_name)
            cli_logger.debug(f"OpenAICompleter initialized for model {model_name}.", extra={"run_id": run_id, "model": model_name})
        except ValueError as e:
            cli_logger.error(f"Failed to initialize OpenAICompleter: {e}", extra={"run_id": run_id}, exc_info=True)
            print(
                f"\nError: Could not initialize the LLM completer. "
                f"Ensure OPENAI_API_KEY is set. Details: {e}"
            )
            raise typer.Exit(code=1)

        print(f"\n--- Asking LLM ({model_name}) ---")
        typer.echo("Waiting for response from LLM...")

        llm_answer = completer.get_completion(
            prompt=prompt_str,
            temperature=temperature,
            max_tokens=max_tokens,
            # model_name is passed to constructor, but can be overridden here if needed
        )
        cli_logger.debug("LLM completion attempt finished.", extra={"run_id": run_id})

        # 4. Print answer and sources
        if llm_answer:
            print("\n--- Answer ---")
            print(llm_answer)
            cli_logger.info(f"LLM Answer received. Length: {len(llm_answer)}", extra={"run_id": run_id, "answer_length": len(llm_answer)})
        else:
            print("\n--- Answer ---")
            print("The LLM did not provide an answer or an error occurred.")
            cli_logger.warning("LLM did not return an answer.", extra={"run_id": run_id})

        if retrieved_chunks:
            print("\n--- Sources Used for Context ---")
            unique_sources = set()
            for i, chunk_item in enumerate(retrieved_chunks, 1):
                source_id = chunk_item.meta.get('source_filepath', chunk_item.doc_id)
                page_info = (
                    f", page {chunk_item.meta.get('page_number')}"
                    if chunk_item.meta.get('page_number')
                    else ""
                )

                # Create a unique identifier for the source display if needed,
                # e.g. combining path and page
                display_source = f"{source_id}{page_info}"
                if display_source not in unique_sources:
                    print(f"  - {display_source} (Retrieved as context chunk {i})")
                    unique_sources.add(display_source)
        else:
            print(
                "\nNo specific sources were retrieved to form the context "
                "for this query."
            )

    except Exception as e:
        cli_logger.error(f"An error occurred in the 'ask' command: {e}", extra={"run_id": run_id}, exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def config(project_dir: Path) -> None:
    """Print config values from project directory."""
    # Generate run_id for this command
    run_logger = RunLogger(project_dir)
    run_id = run_logger.base_dir.name
    
    cli_logger.info(f"Reading config from: {project_dir}", extra={"run_id": run_id, "command": "config", "project_dir": str(project_dir)})

    # Create ProjectManager
    project = ProjectManager(project_dir)

    # Print basic info without exposing config values
    cli_logger.debug(f"Config type: {type(project.config)}", extra={"run_id": run_id, "config_type": str(type(project.config))})
    print(f"Config type: {type(project.config)}")
    print("Config values: [REDACTED FOR PRIVACY]")

    # If it's a dict, show only structure info, not values
    if isinstance(project.config, dict):
        config_keys = list(project.config.keys())
        cli_logger.debug(f"Config sections available: {config_keys}", extra={"run_id": run_id, "config_sections": config_keys})
        print(f"Config sections available: {config_keys}")
        
        embedding_config = project.config.get('embedding', {})
        if isinstance(embedding_config, dict):
            embedding_keys = list(embedding_config.keys())
            cli_logger.debug(f"Embedding config keys: {embedding_keys}", extra={"run_id": run_id, "embedding_keys": embedding_keys})
            print(f"Embedding config keys: {embedding_keys}")
            
            use_async_batch = embedding_config.get('use_async_batch', 'NOT_FOUND')
            cli_logger.debug(
                f"use_async_batch: {use_async_batch} (type: {type(use_async_batch)})",
                extra={"run_id": run_id, "use_async_batch": use_async_batch, "use_async_batch_type": str(type(use_async_batch))}
            )
            print(f"use_async_batch: {use_async_batch} (type: {type(use_async_batch)})")


@app.command()
def enrich_images(
    project_path: Path = typer.Argument(..., help="Path to the project folder."),
    doc_type: str = typer.Option(
        "pptx", help="Document type to enrich (e.g., pptx, pdf, docx)"
    ),
    overwrite: bool = typer.Option(
        False, help="Overwrite original TSV instead of saving to /enriched"
    ),
):
    """
    Enrich chunks with image summaries using the ImageInsightAgent.
    """
    # Generate run_id for this command
    run_logger = RunLogger(project_path)
    run_id = run_logger.base_dir.name
    
    cli_logger.info("Starting image enrichment", extra={"run_id": run_id, "command": "enrich_images", "doc_type": doc_type, "overwrite": overwrite})
    
    project = ProjectManager(project_path)
    agent = ImageInsightAgent(project)

    input_tsv = project_path / "input" / f"chunks_{doc_type}.tsv"
    output_dir = (
        (project_path / "input" / "enriched")
        if not overwrite
        else (project_path / "input")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tsv = output_dir / f"chunks_{doc_type}.tsv"

    enriched_chunks: list[Chunk] = []

    with open(input_tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            meta = json.loads(row[4])
            chunk = Chunk(
                id=row[0],
                doc_id=row[1],
                text=row[2],
                token_count=int(row[3]),
                meta=meta,
            )

            # Run only if image_path exists
            result = agent.run(chunk, project)
            enriched_chunks.extend(result if isinstance(result, list) else [result])

    # Write to output
    with open(output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['chunk_id', 'doc_id', 'text', 'token_count', 'meta_json'])
        for chunk in enriched_chunks:
            writer.writerow(
                [
                    chunk.id,
                    chunk.doc_id,
                    chunk.text,
                    chunk.token_count,
                    json.dumps(chunk.meta),
                ]
            )

    cli_logger.info(f"✅ Enriched {len(enriched_chunks)} chunks. Output written to: {output_tsv}", extra={"run_id": run_id, "enriched_count": len(enriched_chunks), "output_file": str(output_tsv)})
    print(f"✅ Enriched {len(enriched_chunks)} chunks. Output written to: {output_tsv}")


@app.command()
def index_images(
    project_path: Path = typer.Argument(..., help="Path to the RAG project directory."),
    doc_type: str = typer.Option(
        "pptx", help="Document type to read enriched chunks from"
    ),
):
    """
    Index enriched image summaries (ImageChunks) into image_index.faiss and
    image_metadata.jsonl.
    """
    # Generate run_id for this command
    run_logger = RunLogger(project_path)
    run_id = run_logger.base_dir.name
    
    cli_logger.info("Starting image indexing", extra={"run_id": run_id, "command": "index_images", "doc_type": doc_type})
    import csv
    import json
    from scripts.chunking.models import ImageChunk
    from scripts.core.project_manager import ProjectManager
    from scripts.embeddings.image_indexer import ImageIndexer

    project = ProjectManager(project_path)
    indexer = ImageIndexer(project)

    enriched_path = project_path / "input" / "enriched" / f"chunks_{doc_type}.tsv"
    if not enriched_path.exists():
        typer.echo(f"❌ Enriched TSV not found: {enriched_path}")
        raise typer.Exit(1)

    image_chunks: list[ImageChunk] = []

    with open(enriched_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            meta = json.loads(row[4])
            summaries = meta.get("image_summaries", [])
            for s in summaries:
                image_chunks.append(
                    ImageChunk(
                        id=str(uuid.uuid4()),
                        description=s["description"],
                        meta={
                            "image_path": s["image_path"],
                            "source_chunk_id": row[0],
                            "doc_type": meta.get("doc_type"),
                            "source_filepath": meta.get("source_filepath"),
                            "page_number": meta.get("page_number"),
                        },
                    )
                )

    indexer.run(image_chunks)
    cli_logger.info(f"✅ Indexed {len(image_chunks)} image chunks", extra={"run_id": run_id, "indexed_count": len(image_chunks)})
    typer.echo(
        f"✅ Indexed {len(image_chunks)} image chunks into FAISS and metadata JSONL."
    )


if __name__ == "__main__":
    # Configure root logger for CLI output if needed, or rely on LoggerManager
    # For example, to see INFO messages from modules if not configured by LoggerManager:
    # logging.basicConfig(level=logging.INFO)
    app()
