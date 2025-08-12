import pathlib
import inspect  # Added import
from typing import List
import logging
from . import LOADER_REGISTRY
from .models import RawDoc, UnsupportedFileError
from scripts.utils.logger import LoggerManager
from pathlib import Path


class IngestionManager:
    def __init__(self, log_file: Path | None = None):
        """
        Initializes the IngestionManager.
        This manager is responsible for ingesting documents from a specified path.
        It uses a registry of loaders to handle different file types.
        """
        print(f"IngestionManager received log_file: {log_file}")  # Add this
        self.logger = LoggerManager.get_logger("ingestion", log_file=str(log_file))
        print(f"Logger created, checking handlers...")  # Add this
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                print(f"FileHandler baseFilename: {handler.baseFilename}")  # Add this

    def ingest_path(self, path: str | pathlib.Path) -> List[RawDoc]:
        self.logger.info(f"Starting ingestion from: {path.resolve()}")
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
                                f"Ingested segment: {len(raw_docs)} total"
                            )

                    else:
                        # Handle function-based loaders
                        # Assuming: (content: str, metadata: dict)
                        if not callable(loader_or_class):
                            # This case should ideally not be reached if 
                        # LOADER_REGISTRY is set up correctly
                            print(f"Error: Loader for {item.suffix} is not callable.")
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
                                    f"list): {len(raw_docs)} total"
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
                                f"{len(raw_docs)} total"
                            )

                except UnsupportedFileError as e:
                    self.logger.warning(
                        f"Loader for {item.suffix} is not callable. Found error: "
                        f"{e} Skipping."
                    )
                except Exception as e:
                    # Or handle more gracefully
                    # print(f"Error loading {item}: {e}")
                    self.logger.warning(f"Error loading {item}: {e}")
        return raw_docs
