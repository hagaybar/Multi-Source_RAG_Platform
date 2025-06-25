# Workaround for OpenMP runtime conflict on Windows (libomp vs. libiomp)
# See: https://github.com/pytorch/pytorch/issues/37377 and https://openmp.llvm.org
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pathlib
from shutil import copy
import copy as copy_module

import typer  # type: ignore
import json  # Added import
import csv  # Added import
from collections import defaultdict
from pathlib import Path

from scripts.ingestion.manager import IngestionManager
from scripts.chunking.chunker_v3 import split as chunker_split
from scripts.chunking.models import Chunk
# from scripts.embeddings.chunk_embedder import ChunkEmbedder
from scripts.embeddings.unified_embedder import UnifiedEmbedder
from scripts.utils.logger import LoggerManager
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager

app = typer.Typer()


@app.command()
def ingest(
    folder_path: pathlib.Path = typer.Argument(
        ..., help="Path to the folder to ingest."
    ),
    chunk: bool = typer.Option(
        False, "--chunk", help="Enable chunking of ingested documents."
    )
):
    """
    Ingests documents from the specified folder and optionally chunks them.
    """
    
    project = ProjectManager(folder_path)
    ingestion_manager = IngestionManager(log_file=str(project.get_log_path("ingestion")))
    chunker_logger = LoggerManager.get_logger("chunker_project", log_file=str(project.get_log_path("chunker")))

    chunker_logger.info(f"Starting ingestion from folder: {folder_path}")
    if not folder_path.is_dir():
        chunker_logger.error(f"Error: {folder_path} is not a valid directory.")
        raise typer.Exit(code=1)
    
    
        # Add these debug lines:
    chunker_logger.info(f"Chunker log path: {project.get_log_path('chunker')}")
    chunker_logger.info(f"Chunker log path as string: {str(project.get_log_path('chunker'))}")
    chunker_logger.info("Checking chunker logger handlers...")
    for handler in chunker_logger.handlers:
        if hasattr(handler, 'baseFilename'):
            chunker_logger.info(f"Chunker FileHandler baseFilename: {handler.baseFilename}")
    raw_docs = ingestion_manager.ingest_path(folder_path)

    # Changed "documents" to "text segments"
    chunker_logger.info(f"Ingested {len(raw_docs)} text segments from {folder_path}")

    if chunk:
        chunker_logger.info("Chunking is enabled. Proceeding with chunking...")
        if not raw_docs:
            chunker_logger.info("No documents were ingested, skipping chunking.")
            raise typer.Exit()

        chunker_logger.info("Chunking ingested documents...")
        all_chunks: list[Chunk] = []

        for raw_doc in raw_docs:
            chunker_logger.info(f"Processing document: {raw_doc.metadata.get('source_filepath')}")  # Add this line
            # Ensure doc_id is properly assigned for chunking
            # RawDoc.metadata should contain 'source_filepath'
            doc_id = raw_doc.metadata.get('source_filepath', 'unknown_document')
            chunker_logger.info(f"Processing document: {raw_doc.metadata.get('source_filepath')}")  # Add this line
            if not raw_doc.metadata.get('doc_type'):
                warning_msg = (
                    f"Warning: doc_type missing in metadata for {doc_id}, "
                    f"content: {raw_doc.content[:100]}..."
                )
                chunker_logger.info(warning_msg)
                # Potentially skip or assign default doc_type
                # BaseChunker will raise error if doc_type is missing.

            try:
                # Ensure raw_doc.metadata contains 'doc_id' as expected by chunker_v3.py.
                # The 'doc_id' key should ideally be populated by the IngestionManager or here if not.
                # For now, we rely on 'source_filepath' being in metadata and chunker_v3 using meta.get('doc_id').
                # Let's ensure 'doc_id' is explicitly set in the metadata passed to the chunker for clarity.
                current_meta = raw_doc.metadata.copy()
                current_meta['doc_id'] = doc_id # doc_id is from raw_doc.metadata.get('source_filepath', ...)

                document_chunks = chunker_split(
                    text=raw_doc.content,
                    meta=current_meta,
                    logger=chunker_logger 
                    # clean_options will use default from chunker_v3.split
                )
                print(f"[CHUNK] {raw_doc.metadata.get('source_filepath')} => {raw_doc.metadata.get('doc_type')}")
                all_chunks.extend(document_chunks)
                
            except ValueError as e:
                error_msg = (
                    f"Skipping chunking for a segment from {doc_id} "
                    f"due to error: {e}"
                )
                print(error_msg)
            except Exception as e:
                error_msg = (
                    f"An unexpected error occurred while chunking a segment "
                    f"from {doc_id}: {e}"
                )
                print(error_msg)

        print(f"Generated {len(all_chunks)} chunks.")

        if all_chunks:
            # Group chunks by doc_type
            doc_type_map = defaultdict(list)
            for chk in all_chunks:
                doc_type = chk.meta.get("doc_type", "default")
                doc_type_map[doc_type].append(chk)

            for doc_type, chunks in doc_type_map.items():
                output_path = folder_path / "input" / f"chunks_{doc_type}.tsv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(output_path, "w", newline="", encoding="utf-8") as tsvfile:
                        writer = csv.writer(tsvfile, delimiter="\t")
                        header = ['chunk_id', 'doc_id', 'text', 'token_count', 'meta_json']
                        writer.writerow(header)
                        for chk in chunks:
                            meta_json_str = json.dumps(chk.meta)
                            writer.writerow([chk.id, chk.doc_id, chk.text, chk.token_count, meta_json_str])
                    print(f"Wrote {len(chunks)} chunks to {output_path.name}")
                except IOError as e:
                    error_msg = f"Error writing chunks for {doc_type}: {e}"
                    chunker_logger(error_msg)
                    raise typer.Exit(code=1)
        else:
            print("No chunks were generated.")

@app.command()
def embed(
    project_dir: Path,
    use_async: bool = typer.Option(False, "--a-b", "--async-batch", help="Use OpenAI async batch embedding")
) -> None:
    """
    Generate embeddings for chunks in the specified project directory.
    """
    print("\n" + "=" * 120)
    print("DEBUG: CLI embed() command STARTING")
    print("=" * 120)
    
    print(f"DEBUG: CLI Arguments received:")
    print(f"DEBUG:   - project_dir: {project_dir}")
    print(f"DEBUG:   - use_async: {use_async}")
    print(f"DEBUG:   - use_async type: {type(use_async)}")
    
    if not project_dir.exists():
        error_msg = f"Project directory does not exist: {project_dir}"
        print(f"ERROR: {error_msg}")
        typer.echo(f"Error: {error_msg}")
        raise typer.Exit(1)
    
    logger = LoggerManager.get_logger("cli")
    
    # Initialize project manager
    print("DEBUG: Creating ProjectManager...")
    project = ProjectManager(project_dir)
    print(f"DEBUG: ProjectManager created for: {project_dir}")
    runtime_config = copy_module.deepcopy(project.config)
    print(f"DEBUG: Project config loaded: {runtime_config}")

    # Override config if async flag is provided
    if use_async:
        print("DEBUG: CLI use_async is TRUE - Overriding config")
        logger.info("Embedding mode override: use_async_batch=True")
        
        # Set the runtime config directly (config is a plain dict)
        if 'embedding' not in runtime_config:
            runtime_config['embedding'] = {}
        
        runtime_config['embedding']['use_async_batch'] = True
        print("DEBUG: Set runtime_config['embedding']['use_async_batch'] = True")
        print(f"DEBUG: Updated runtime config: {runtime_config}")
        print(f"DEBUG: Original project.config unchanged: {project.config}")
        
        # Verify the setting in runtime_config (since config is a plain dict)
        if 'embedding' in runtime_config and 'use_async_batch' in runtime_config['embedding']:
            async_batch_value = runtime_config['embedding']['use_async_batch']
            print(f"DEBUG: Verification - runtime_config['embedding']['use_async_batch'] = {async_batch_value}")
        else:
            print("DEBUG: use_async_batch not found in runtime_config")
        
    else:
        print("DEBUG: CLI use_async is FALSE - Using default config")
        logger.info("Embedding mode: using default configuration")
    
    print("DEBUG: About to create UnifiedEmbedder...")
    embedder = UnifiedEmbedder(project, runtime_config=runtime_config)
    
    print(f"DEBUG: UnifiedEmbedder created:")
    print(f"DEBUG:   - embedder.use_async_batch: {embedder.use_async_batch}")
    print(f"DEBUG:   - Expected: {use_async}")
    
    if use_async and not embedder.use_async_batch:
        print("ERROR: CLI flag --async was True but embedder.use_async_batch is False!")
        print("ERROR: Configuration override failed!")
    elif use_async and embedder.use_async_batch:
        print("SUCCESS: CLI flag --async correctly set embedder.use_async_batch = True")



    logger.info(f"CLI: Created embedder with use_async_batch={embedder.use_async_batch}")
    
    print("DEBUG: About to call embedder.run_from_folder()...")
    embedder.run_from_folder()
    
    print("=" * 120)
    print("DEBUG: CLI embed() command COMPLETE")
    print("=" * 120)

@app.command()
def retrieve(
    project_path: str = typer.Argument(..., help="Path to the RAG project directory"),
    query: str = typer.Argument(..., help="Search query string"),
    top_k: int = typer.Option(10, help="Number of top chunks to return"),
    strategy: str = typer.Option("late_fusion", help="Retrieval strategy to use")
):
    """
    Retrieve top-k chunks from multiple document types using the configured strategy.
    """
    project = ProjectManager(project_path)
    rm = RetrievalManager(project)

    results = rm.retrieve(query=query, top_k=top_k, strategy=strategy)

    print(f"\n--- Top {len(results)} results ---")
    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] From {chunk.meta.get('_retriever')} | score: {chunk.meta.get('similarity', 0):.3f}")
        print(chunk.text.strip()[:500])  # Preview max 500 chars

@app.command()
def config(project_dir: Path) -> None:
    """Print config values from project directory."""
    
    print(f"Reading config from: {project_dir}")
    
    # Create ProjectManager
    project = ProjectManager(project_dir)
    
    # Print basic info
    print(f"Config type: {type(project.config)}")
    print(f"Config value: {project.config}")
    
    # If it's a dict, show the embedding section
    if isinstance(project.config, dict):
        embedding = project.config.get('embedding', {})
        print(f"Embedding section: {embedding}")
        
        if isinstance(embedding, dict):
            use_async_batch = embedding.get('use_async_batch', 'NOT_FOUND')
            print(f"use_async_batch: {use_async_batch} (type: {type(use_async_batch)})")


if __name__ == "__main__":
    app()