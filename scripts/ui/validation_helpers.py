# scripts/ui/validation_helpers.py

from pathlib import Path


def validate_steps(project, steps: list[str], query: str) -> tuple[bool, list[str]]:
    errors = []
    warnings = []

    input_dir = project.input_dir
    output_dir = project.output_dir

    has_chunks = any(input_dir.glob("chunks_*.tsv"))
    has_enriched = (input_dir / "enriched").exists()
    has_faiss = any((output_dir / "faiss").glob("*.faiss"))
    has_jsonl = any(output_dir.glob("*.jsonl"))

    if "chunk" in steps and not any(project.raw_docs_dir().glob("**/*")):
        warnings.append(
            "No raw files found. Chunking may fail unless you ingest first."
        )

    if "enrich" in steps and not has_chunks:
        warnings.append(
            "No chunks found. Enrichment will fail unless chunking is done first."
        )

    if "index_images" in steps and not has_enriched:
        warnings.append(
            "No enriched directory found. Image indexing will likely fail."
        )

    if "embed" in steps and not has_chunks and not has_enriched:
        errors.append("No chunks available for embedding. Please run chunking first.")

    if "retrieve" in steps or "ask" in steps:
        if not has_faiss or not has_jsonl:
            errors.append("Retrieval requires FAISS index and metadata files. Run embed first.")
        if not query.strip():
            errors.append("A query is required for retrieve/ask steps.")

    return (len(errors) == 0), errors + warnings
