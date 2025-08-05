from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Chunk:
    doc_id: str
    text: str
    meta: Dict[str, Any]
    token_count: int  # Add this line
    title: Optional[str] = None
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Doc:
    doc_id: str
    chunks: List[Chunk]
    meta: Dict[str, Any]

    def __repr__(self):
        return (
            f"Doc(doc_id='{self.doc_id}', num_chunks={len(self.chunks)}, "
            f"meta={self.meta})"
        )


@dataclass
class ImageChunk:
    id: str
    description: str
    meta: dict
