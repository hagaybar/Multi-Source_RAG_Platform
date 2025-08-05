from typing import List
import hashlib
import csv
import json
from scripts.chunking.models import Chunk


def deduplicate_chunks(
    chunks: List[Chunk],
    existing_hashes: set[str],
    skip_duplicates: bool,
    logger=None,
) -> List[Chunk]:
    print(
        f"[DEBUG] deduplicate_chunks called with skip_duplicates={skip_duplicates}"
    )
    new_chunks = []
    seen_hashes = set()
    print(f"[DEBUG] deduplicate_chunks received {len(chunks)} chunks")
    print(f"[DEBUG] First chunk type: {type(chunks[0])}")

    for chunk in chunks:
        raw = getattr(chunk, "text", None) or getattr(
            chunk, "description", None
        )
        if not raw:
            logger.warning(
                f"Skipping chunk with no text or description (id={chunk.id})"
            )
            continue

        content_hash = hashlib.sha256(
            raw.strip().encode("utf-8")
        ).hexdigest()

        if skip_duplicates and content_hash in existing_hashes:
            if logger:
                logger.debug(
                    f"Skipping duplicate chunk: {content_hash[:16]}..."
                )
            continue

        # Optional: prevent duplicates within the same batch
        if content_hash in seen_hashes:
            continue

        seen_hashes.add(content_hash)
        chunk.meta[
            "content_hash"
        ] = content_hash  # Optional but useful for debugging
        new_chunks.append(chunk)

    return new_chunks


def load_chunks(chunks_path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            chunk = Chunk(
                id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                token_count=int(row["token_count"]),
                meta=json.loads(row["meta_json"]),
            )
            chunks.append(chunk)
    return chunks
