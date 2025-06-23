import pathlib
import typer  # type: ignore
import json  # Added import
import csv  # Added import
from collections import defaultdict

from scripts.ingestion.manager import IngestionManager
from scripts.chunking.chunker_v3 import split as chunker_split
from scripts.chunking.models import Chunk
# from scripts.embeddings.chunk_embedder import ChunkEmbedder
from scripts.embeddings.unified_embedder import UnifiedEmbedder
from scripts.utils.logger import LoggerManager
from scripts.core.project_manager import ProjectManager


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
    project_dir: pathlib.Path = typer.Argument(..., help="Path to the project directory")
):
    """
    Embeds chunks.tsv in the given project and saves FAISS index + metadata.
    """
    logger = LoggerManager.get_logger("embedder")

    if not project_dir.is_dir():
        logger.error(f"Provided project_dir does not exist: {project_dir}")
        raise typer.Exit(code=1)
    
    project = ProjectManager(project_dir)
    embedder = UnifiedEmbedder(project)
    # embedder.run_from_file()
    embedder.run_from_folder()


if __name__ == "__main__":
    app()
