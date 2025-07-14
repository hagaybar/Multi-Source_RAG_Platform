# Workaround for OpenMP runtime conflict on Windows (libomp vs. libiomp)
# See: https://github.com/pytorch/pytorch/issues/37377 and https://openmp.llvm.org
import sys
import pathlib

# Ensure the root directory (where pyproject.toml lives) is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from shutil import copy
import copy as copy_module
import logging # Added for ask command

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
from scripts.prompting.prompt_builder import PromptBuilder # Added for ask command
from scripts.api_clients.openai.completer import OpenAICompleter # Added for ask command

app = typer.Typer()

# Setup basic logging for the CLI
cli_logger = LoggerManager.get_logger("cli_app")


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

            for doc_type, chunks_list in doc_type_map.items():
                output_path = folder_path / "input" / f"chunks_{doc_type}.tsv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(output_path, "w", newline="", encoding="utf-8") as tsvfile:
                        writer = csv.writer(tsvfile, delimiter="\t")
                        header = ['chunk_id', 'doc_id', 'text', 'token_count', 'meta_json']
                        writer.writerow(header)
                        for chk_item in chunks_list: # Renamed variable to avoid conflict
                            meta_json_str = json.dumps(chk_item.meta)
                            writer.writerow([chk_item.id, chk_item.doc_id, chk_item.text, chk_item.token_count, meta_json_str])
                    print(f"Wrote {len(chunks_list)} chunks to {output_path.name}")
                except IOError as e:
                    error_msg = f"Error writing chunks for {doc_type}: {e}"
                    chunker_logger.error(error_msg) # Use error level
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
    cli_logger.info("\n" + "=" * 120)
    cli_logger.info("DEBUG: CLI embed() command STARTING")
    cli_logger.info("=" * 120)
    
    cli_logger.info(f"DEBUG: CLI Arguments received:")
    cli_logger.info(f"DEBUG:   - project_dir: {project_dir}")
    cli_logger.info(f"DEBUG:   - use_async: {use_async}")
    cli_logger.info(f"DEBUG:   - use_async type: {type(use_async)}")
    
    if not project_dir.exists():
        error_msg = f"Project directory does not exist: {project_dir}"
        cli_logger.error(f"ERROR: {error_msg}")
        typer.echo(f"Error: {error_msg}")
        raise typer.Exit(1)
    
    # logger = LoggerManager.get_logger("cli") # Already have cli_logger
    
    # Initialize project manager
    cli_logger.info("DEBUG: Creating ProjectManager...")
    project = ProjectManager(project_dir)
    cli_logger.info(f"DEBUG: ProjectManager created for: {project_dir}")
    runtime_config = copy_module.deepcopy(project.config)
    cli_logger.info(f"DEBUG: Project config loaded: {runtime_config}")

    # Override config if async flag is provided
    if use_async:
        cli_logger.info("DEBUG: CLI use_async is TRUE - Overriding config")
        cli_logger.info("Embedding mode override: use_async_batch=True")
        
        # Set the runtime config directly (config is a plain dict)
        if 'embedding' not in runtime_config:
            runtime_config['embedding'] = {}
        
        runtime_config['embedding']['use_async_batch'] = True
        cli_logger.info("DEBUG: Set runtime_config['embedding']['use_async_batch'] = True")
        cli_logger.info(f"DEBUG: Updated runtime config: {runtime_config}")
        cli_logger.info(f"DEBUG: Original project.config unchanged: {project.config}")
        
        # Verify the setting in runtime_config (since config is a plain dict)
        if 'embedding' in runtime_config and 'use_async_batch' in runtime_config['embedding']:
            async_batch_value = runtime_config['embedding']['use_async_batch']
            cli_logger.info(f"DEBUG: Verification - runtime_config['embedding']['use_async_batch'] = {async_batch_value}")
        else:
            cli_logger.info("DEBUG: use_async_batch not found in runtime_config")
        
    else:
        cli_logger.info("DEBUG: CLI use_async is FALSE - Using default config")
        cli_logger.info("Embedding mode: using default configuration")
    
    cli_logger.info("DEBUG: About to create UnifiedEmbedder...")
    embedder = UnifiedEmbedder(project, runtime_config=runtime_config)
    
    cli_logger.info(f"DEBUG: UnifiedEmbedder created:")
    cli_logger.info(f"DEBUG:   - embedder.use_async_batch: {embedder.use_async_batch}")
    cli_logger.info(f"DEBUG:   - Expected: {use_async}")
    
    if use_async and not embedder.use_async_batch:
        cli_logger.error("ERROR: CLI flag --async was True but embedder.use_async_batch is False!")
        cli_logger.error("ERROR: Configuration override failed!")
    elif use_async and embedder.use_async_batch:
        cli_logger.info("SUCCESS: CLI flag --async correctly set embedder.use_async_batch = True")

    cli_logger.info(f"CLI: Created embedder with use_async_batch={embedder.use_async_batch}")
    
    cli_logger.info("DEBUG: About to call embedder.run_from_folder()...")
    embedder.run_from_folder()
    
    cli_logger.info("=" * 120)
    cli_logger.info("DEBUG: CLI embed() command COMPLETE")
    cli_logger.info("=" * 120)

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
    cli_logger.info(f"Starting retrieval for project: {project_path}, query: '{query}'")
    project = ProjectManager(project_path)
    rm = RetrievalManager(project)

    results = rm.retrieve(query=query, top_k=top_k, strategy=strategy)

    print(f"\n--- Top {len(results)} results for query: '{query}' ---")
    if not results:
        print("No results found.")
    for i, chunk_item in enumerate(results, 1): # Renamed variable
        print(f"\n[{i}] From {chunk_item.meta.get('_retriever')} | score: {chunk_item.meta.get('similarity', 0):.3f} | doc_id: {chunk_item.doc_id}")
        source_filepath = chunk_item.meta.get('source_filepath', 'N/A')
        print(f"    Source File: {source_filepath}")
        page_number = chunk_item.meta.get('page_number')
        if page_number:
            print(f"    Page: {page_number}")
        print(f"    Text: {chunk_item.text.strip()[:500]}...")
    cli_logger.info(f"Retrieved {len(results)} chunks.")


@app.command()
def ask(
    project_path: str = typer.Argument(..., help="Path to the RAG project directory."),
    query: str = typer.Argument(..., help="Your question to the RAG system."),
    top_k: int = typer.Option(5, help="Number of context chunks to retrieve."),
    temperature: float = typer.Option(0.7, help="LLM temperature for response generation."),
    max_tokens: int = typer.Option(500, help="LLM maximum tokens for the response."),
    model_name: str = typer.Option("gpt-3.5-turbo", help="OpenAI model to use for generating the answer (via LiteLLM).")
):
    """
    Asks a question to the RAG system.

    This command retrieves relevant context chunks from the indexed documents
    in the specified project, then uses an LLM (via LiteLLM, configured for OpenAI)
    to generate an answer based on your query and the retrieved context.

    Requires the OPENAI_API_KEY environment variable to be set for LLM access.
    """
    cli_logger.info(f"Starting 'ask' command for project: {project_path}")
    cli_logger.info(f"Query: '{query}', top_k: {top_k}, model: {model_name}")

    try:
        project = ProjectManager(project_path)
        cli_logger.info(f"ProjectManager initialized for {project.root_dir}")

        # 1. Retrieve context
        retrieval_manager = RetrievalManager(project)
        cli_logger.info(f"RetrievalManager initialized. Retrieving top {top_k} chunks for query...")
        retrieved_chunks = retrieval_manager.retrieve(query=query, top_k=top_k)

        if not retrieved_chunks:
            cli_logger.warning("No context chunks retrieved. Answering based on query alone might be difficult or impossible.")
            print("\nWarning: No relevant context documents were found for your query.")
            # Decide if to proceed or exit. For now, proceed, LMM will be told context is empty.
        else:
            cli_logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
            print(f"\n--- Retrieved {len(retrieved_chunks)} context chunks ---")
            for i, chunk_item in enumerate(retrieved_chunks, 1):
                source_id = chunk_item.meta.get('source_filepath', chunk_item.doc_id)
                page_info = f", page {chunk_item.meta.get('page_number')}" if chunk_item.meta.get('page_number') else ""
                print(f"  [{i}] Source: {source_id}{page_info} (Score: {chunk_item.meta.get('similarity', 0):.3f})")
                # print(f"      Text: {chunk_item.text[:100].strip()}...") # Optional: print chunk text preview

        # 2. Build prompt
        prompt_builder = PromptBuilder() # Uses default template
        cli_logger.info("PromptBuilder initialized.")
        prompt_str = prompt_builder.build_prompt(query=query, context_chunks=retrieved_chunks)
        cli_logger.info(f"Prompt built. Length: {len(prompt_str)} chars.")
        # cli_logger.debug(f"Generated Prompt:\n{prompt_str}") # Potentially very long

        # 3. Get LMM completion
        # API key for OpenAICompleter is handled internally (expects OPENAI_API_KEY env var)
        try:
            completer = OpenAICompleter(model_name=model_name)
            cli_logger.info(f"OpenAICompleter initialized for model {model_name}.")
        except ValueError as e:
            cli_logger.error(f"Failed to initialize OpenAICompleter: {e}")
            print(f"\nError: Could not initialize the LLM completer. Ensure OPENAI_API_KEY is set. Details: {e}")
            raise typer.Exit(code=1)

        print(f"\n--- Asking LLM ({model_name}) ---")
        typer.echo("Waiting for response from LLM...")

        llm_answer = completer.get_completion(
            prompt=prompt_str,
            temperature=temperature,
            max_tokens=max_tokens
            # model_name is passed to constructor, but can be overridden here if needed
        )
        cli_logger.info("LLM completion attempt finished.")

        # 4. Print answer and sources
        if llm_answer:
            print("\n--- Answer ---")
            print(llm_answer)
            cli_logger.info(f"LLM Answer received. Length: {len(llm_answer)}")
        else:
            print("\n--- Answer ---")
            print("The LLM did not provide an answer or an error occurred.")
            cli_logger.warning("LLM did not return an answer.")

        if retrieved_chunks:
            print("\n--- Sources Used for Context ---")
            unique_sources = set()
            for i, chunk_item in enumerate(retrieved_chunks, 1):
                source_id = chunk_item.meta.get('source_filepath', chunk_item.doc_id)
                page_info = f", page {chunk_item.meta.get('page_number')}" if chunk_item.meta.get('page_number') else ""

                # Create a unique identifier for the source display if needed, e.g. combining path and page
                display_source = f"{source_id}{page_info}"
                if display_source not in unique_sources:
                    print(f"  - {display_source} (Retrieved as context chunk {i})")
                    unique_sources.add(display_source)
        else:
            print("\nNo specific sources were retrieved to form the context for this query.")

    except Exception as e:
        cli_logger.error(f"An error occurred in the 'ask' command: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def config(project_dir: Path) -> None:
    """Print config values from project directory."""
    
    cli_logger.info(f"Reading config from: {project_dir}")
    
    # Create ProjectManager
    project = ProjectManager(project_dir)
    
    # Print basic info
    cli_logger.info(f"Config type: {type(project.config)}")
    cli_logger.info(f"Config value: {project.config}")
    print(f"Config type: {type(project.config)}")
    print(f"Config value: {project.config}")
    
    # If it's a dict, show the embedding section
    if isinstance(project.config, dict):
        embedding_config = project.config.get('embedding', {}) # Renamed variable
        cli_logger.info(f"Embedding section: {embedding_config}")
        print(f"Embedding section: {embedding_config}")
        
        if isinstance(embedding_config, dict):
            use_async_batch = embedding_config.get('use_async_batch', 'NOT_FOUND')
            cli_logger.info(f"use_async_batch: {use_async_batch} (type: {type(use_async_batch)})")
            print(f"use_async_batch: {use_async_batch} (type: {type(use_async_batch)})")


@app.command()
def enrich_images(
    project_path: Path = typer.Argument(..., help="Path to the project folder."),
    doc_type: str = typer.Option("pptx", help="Document type to enrich (e.g., pptx, pdf, docx)"),
    overwrite: bool = typer.Option(False, help="Overwrite original TSV instead of saving to /enriched")
):
    """
    Enrich chunks with image summaries using the ImageInsightAgent.
    """
    from scripts.agents.image_insight_agent import ImageInsightAgent
    from scripts.chunking.models import Chunk
    import csv
    import json

    project = ProjectManager(project_path)
    agent = ImageInsightAgent()

    input_tsv = project_path / "input" / f"chunks_{doc_type}.tsv"
    output_dir = (project_path / "input" / "enriched") if not overwrite else (project_path / "input")
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
                meta=meta
            )

            # Run only if image_path exists
            result = agent.run(chunk, project)
            enriched_chunks.extend(result if isinstance(result, list) else [result])

    # Write to output
    with open(output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['chunk_id', 'doc_id', 'text', 'token_count', 'meta_json'])
        for chunk in enriched_chunks:
            writer.writerow([
                chunk.id,
                chunk.doc_id,
                chunk.text,
                chunk.token_count,
                json.dumps(chunk.meta)
            ])

    print(f"✅ Enriched {len(enriched_chunks)} chunks. Output written to: {output_tsv}")



if __name__ == "__main__":
    # Configure root logger for CLI output if needed, or rely on LoggerManager
    # For example, to see INFO messages from modules if not configured by LoggerManager:
    # logging.basicConfig(level=logging.INFO)
    app()