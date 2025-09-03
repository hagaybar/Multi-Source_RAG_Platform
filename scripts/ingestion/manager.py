import pathlib
import inspect  # Added import
from typing import List
import logging
from . import LOADER_REGISTRY
from .models import RawDoc, UnsupportedFileError
from scripts.utils.logger import LoggerManager
from pathlib import Path


class IngestionManager:
    def __init__(self, log_file: Path | None = None, run_id: str | None = None):
        """
        Initializes the IngestionManager.
        This manager is responsible for ingesting documents from a specified path.
        It uses a registry of loaders to handle different file types.
        """
        self.run_id = run_id
        self.logger = LoggerManager.get_logger("ingestion", log_file=str(log_file), run_id=run_id)
        
        # Debug logging for handler information
        self.logger.debug(f"IngestionManager received log_file: {log_file}", extra={"run_id": run_id} if run_id else {})
        self.logger.debug("Logger created, checking handlers...", extra={"run_id": run_id} if run_id else {})
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.debug(f"FileHandler baseFilename: {handler.baseFilename}", extra={"run_id": run_id, "handler_type": "FileHandler", "log_filename": handler.baseFilename} if run_id else {"handler_type": "FileHandler", "log_filename": handler.baseFilename})

    def ingest_path(self, path: str | pathlib.Path) -> List[RawDoc]:
        self.logger.info(f"Starting ingestion from: {path.resolve()}", extra={"run_id": self.run_id, "ingestion_path": str(path.resolve())} if self.run_id else {"ingestion_path": str(path.resolve())})
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        raw_docs = []
        for item in path.rglob("*"):  # rglob for recursive search
            if item.is_file() and item.suffix in LOADER_REGISTRY:
                loader_or_class = LOADER_REGISTRY[item.suffix]
                base_metadata = {
                    'source_filepath': str(item),
                    'doc_type': item.suffix.lstrip('.')
                }
                try:
                    if inspect.isclass(loader_or_class):
                        # Handle class-based ingestors (e.g., PptxIngestor)
                        ingestor_instance = loader_or_class()
                        # PptxIngestor.ingest() returns:
                        # list[tuple[str, dict]]
                        ingested_segments = ingestor_instance.ingest(str(item))
                        for text_segment, seg_meta in ingested_segments:
                            final_meta = base_metadata.copy()
                            # segment_meta includes doc_type from PptxIngestor
                            final_meta.update(seg_meta)
                            raw_docs.append(
                                RawDoc(content=text_segment, metadata=final_meta)
                            )
                            self.logger.debug(
                                f"Ingested segment: {len(raw_docs)} total",
                                extra={"run_id": self.run_id, "total_segments": len(raw_docs), "file_path": str(item)} if self.run_id else {"total_segments": len(raw_docs), "file_path": str(item)}
                            )

                    else:
                        # Handle function-based loaders
                        # Assuming: (content: str, metadata: dict)
                        if not callable(loader_or_class):
                            # This case should ideally not be reached if 
                        # LOADER_REGISTRY is set up correctly
                            self.logger.error(f"Loader for {item.suffix} is not callable", extra={"run_id": self.run_id, "file_suffix": item.suffix, "file_path": str(item)} if self.run_id else {"file_suffix": item.suffix, "file_path": str(item)})
                            continue
                        result = loader_or_class(str(item))
                        if isinstance(result, list):
                            for text_segment, seg_meta in result:
                                final_meta = base_metadata.copy()
                                final_meta.update(seg_meta)
                                raw_docs.append(
                                    RawDoc(content=text_segment, metadata=final_meta)
                                )
                                self.logger.debug(
                                    f"Ingested segment from {item} (function loader "
                                    f"list): {len(raw_docs)} total",
                                    extra={"run_id": self.run_id, "total_segments": len(raw_docs), "file_path": str(item), "loader_type": "function_list"} if self.run_id else {"total_segments": len(raw_docs), "file_path": str(item), "loader_type": "function_list"}
                                )
                        else:
                            content, metadata = result
                            final_meta = base_metadata.copy()
                            final_meta.update(metadata)
                            raw_docs.append(
                                RawDoc(content=content, metadata=final_meta)
                            )
                            self.logger.debug(
                                f"Ingested segment from {item} (function loader): "
                                f"{len(raw_docs)} total",
                                extra={"run_id": self.run_id, "total_segments": len(raw_docs), "file_path": str(item), "loader_type": "function"} if self.run_id else {"total_segments": len(raw_docs), "file_path": str(item), "loader_type": "function"}
                            )

                except UnsupportedFileError as e:
                    self.logger.warning(
                        f"Loader for {item.suffix} is not callable. Found error: "
                        f"{e} Skipping.",
                        extra={"run_id": self.run_id, "file_suffix": item.suffix, "file_path": str(item)} if self.run_id else {"file_suffix": item.suffix, "file_path": str(item)},
                        exc_info=True
                    )
                except Exception as e:
                    # Or handle more gracefully
                    self.logger.error(f"Error loading {item}: {e}", extra={"run_id": self.run_id, "file_path": str(item)} if self.run_id else {"file_path": str(item)}, exc_info=True)
        return raw_docs
