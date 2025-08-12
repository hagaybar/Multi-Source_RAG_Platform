from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RawDoc:
    """
    Represents a raw document extracted by an ingestor,
    before chunking.
    """

    content: str
    metadata: dict


class AbstractIngestor(ABC):
    """
    Abstract base class for ingestors.
    """

    @abstractmethod
    def ingest(self, filepath: str) -> list[tuple[str, dict]]:
        """
        Ingests data from the given filepath.

        Args:
            filepath: Path to the file to ingest.

        Returns:
            A list of tuples, where each tuple contains the extracted text
            and associated metadata. For ingestors that produce multiple
            text segments from a single file (like PptxIngestor), each
            segment is a tuple in the list. For ingestors that produce
            a single block of text for the file, the list would contain
            one tuple.
        """
        pass


class UnsupportedFileError(Exception):
    """Custom exception for unsupported file types or corrupted files."""

    pass
