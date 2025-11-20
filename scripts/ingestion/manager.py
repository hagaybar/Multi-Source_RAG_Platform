import pathlib
import inspect  # Added import
from typing import List, TYPE_CHECKING
import logging
from . import LOADER_REGISTRY
from .models import RawDoc, UnsupportedFileError
from scripts.utils.logger import LoggerManager
from pathlib import Path

if TYPE_CHECKING:
    from scripts.connectors.outlook_connector import OutlookConfig


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

    def ingest_outlook(self, outlook_config: "OutlookConfig") -> List[RawDoc]:
        """
        Ingest emails from Microsoft Outlook using environment-aware connector.

        This method connects to a local Outlook client (Windows) or uses the
        WSL helper (WSL) to extract emails from a specified folder, converting
        them to RawDoc format for the ingestion pipeline.

        Args:
            outlook_config: OutlookConfig object with connection parameters
                           (account_name, folder_path, days_back, max_emails)

        Returns:
            List of RawDoc objects, one per email extracted

        Example:
            >>> from scripts.connectors.outlook_connector import OutlookConfig
            >>> config = OutlookConfig(
            ...     account_name="user@company.com",
            ...     folder_path="Inbox > Work",
            ...     days_back=30
            ... )
            >>> manager = IngestionManager()
            >>> raw_docs = manager.ingest_outlook(config)
            >>> print(f"Ingested {len(raw_docs)} emails")
        """
        from scripts.connectors.outlook_wsl_client import get_outlook_connector

        self.logger.info(
            f"Starting Outlook ingestion from {outlook_config.account_name}/{outlook_config.folder_path}",
            extra={
                "run_id": self.run_id,
                "account_name": outlook_config.account_name,
                "folder_path": outlook_config.folder_path,
                "days_back": outlook_config.days_back,
                "source_type": "outlook"
            } if self.run_id else {
                "account_name": outlook_config.account_name,
                "folder_path": outlook_config.folder_path,
                "days_back": outlook_config.days_back,
                "source_type": "outlook"
            }
        )

        try:
            # Create connector using factory (auto-detects WSL vs Windows)
            connector = get_outlook_connector(outlook_config)
            email_tuples = connector.extract_emails()

            # Convert to RawDoc objects
            raw_docs = []
            for content, metadata in email_tuples:
                raw_docs.append(RawDoc(content=content, metadata=metadata))

            self.logger.info(
                f"Successfully ingested {len(raw_docs)} emails from Outlook",
                extra={
                    "run_id": self.run_id,
                    "email_count": len(raw_docs),
                    "source_type": "outlook",
                    "account_name": outlook_config.account_name,
                    "folder_path": outlook_config.folder_path
                } if self.run_id else {
                    "email_count": len(raw_docs),
                    "source_type": "outlook",
                    "account_name": outlook_config.account_name,
                    "folder_path": outlook_config.folder_path
                }
            )

            return raw_docs

        except ImportError as e:
            self.logger.error(
                f"Failed to import Outlook connector: {e}. Check pywin32 (Windows) or helper setup (WSL).",
                extra={"run_id": self.run_id} if self.run_id else {},
                exc_info=True
            )
            return []

        except Exception as e:
            self.logger.error(
                f"Error during Outlook ingestion: {e}",
                extra={
                    "run_id": self.run_id,
                    "source_type": "outlook",
                    "error": str(e)
                } if self.run_id else {
                    "source_type": "outlook",
                    "error": str(e)
                },
                exc_info=True
            )
            return []
