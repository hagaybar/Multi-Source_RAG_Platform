#!/usr/bin/env python3
"""
Process downloaded OpenAI batch embeddings and build FAISS index.
Fixed version with proper TSV parsing using csv module.

Usage:
    python process_batch_embeddings_fixed.py \
        --batch-output data/projects/Primo_List_2/output/batch_692379f6bc8c8190888247a136599cc7_output.jsonl \
        --chunks data/projects/Primo_List_2/input/chunks_outlook_eml.tsv \
        --output-dir data/projects/Primo_List_2/output
"""
import sys
import json
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List
import faiss

def load_chunks_from_tsv(tsv_path: Path) -> Dict[str, dict]:
    """Load chunks from TSV file using proper CSV parsing."""
    print(f"\n📖 Loading chunks from: {tsv_path}")
    chunks = {}

    with open(tsv_path, 'r', encoding='utf-8') as f:
        # Use csv.reader with tab delimiter and quote handling
        reader = csv.reader(f, delimiter='\t', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL,
                          doublequote=True,
                          skipinitialspace=False)

        # Skip header
        next(reader)

        for line_num, row in enumerate(reader, start=2):
            if len(row) != 5:
                if line_num % 1000 == 0:  # Print every 1000th skip to avoid spam
                    print(f"⚠️  Skipping malformed line {line_num}: {len(row)} parts")
                continue

            chunk_id = row[0]
            doc_id = row[1]
            text = row[2]

            try:
                token_count = int(row[3])
            except ValueError:
                continue

            try:
                meta = json.loads(row[4])
            except json.JSONDecodeError:
                continue

            chunks[chunk_id] = {
                'id': chunk_id,
                'doc_id': doc_id,
                'text': text,
                'token_count': token_count,
                'meta': meta
            }

            if line_num % 10000 == 0:
                print(f"  Loaded {len(chunks):,} chunks so far...")

    print(f"✅ Loaded {len(chunks):,} chunks")
    return chunks

def load_batch_output(batch_output_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings from batch output file."""
    print(f"\n📥 Loading batch embeddings from: {batch_output_path}")
    embeddings = {}
    errors = 0

    with open(batch_output_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
                custom_id = result['custom_id']

                # Check if request succeeded
                if result['response']['status_code'] != 200:
                    if errors < 10:  # Only print first 10 errors
                        print(f"⚠️  Request failed for {custom_id}: Status {result['response']['status_code']}")
                    errors += 1
                    continue

                # Extract embedding
                embedding_data = result['response']['body']['data'][0]['embedding']
                embeddings[custom_id] = np.array(embedding_data, dtype=np.float32)

                if line_num % 1000 == 0:
                    print(f"  Loaded {len(embeddings):,} embeddings so far...")

            except (KeyError, json.JSONDecodeError) as e:
                if errors < 10:
                    print(f"⚠️  Error parsing line {line_num}: {e}")
                errors += 1
                continue

    print(f"✅ Loaded {len(embeddings):,} embeddings")
    if errors > 0:
        print(f"⚠️  {errors} errors encountered")

    return embeddings

def build_faiss_index(embeddings_dict: Dict[str, np.ndarray],
                      chunks_dict: Dict[str, dict],
                      output_dir: Path,
                      doc_type: str = "outlook_eml"):
    """Build FAISS index and metadata from embeddings."""
    print(f"\n🏗️  Building FAISS index...")

    # Sort by chunk ID for consistency
    sorted_ids = sorted(embeddings_dict.keys())

    # Check for missing chunks
    missing_chunks = set(sorted_ids) - set(chunks_dict.keys())
    if missing_chunks:
        print(f"⚠️  Warning: {len(missing_chunks)} embeddings have no matching chunks")
        if len(missing_chunks) < 10:
            for cid in list(missing_chunks)[:10]:
                print(f"    Missing: {cid}")

    # Filter to only chunks that have embeddings
    valid_ids = [cid for cid in sorted_ids if cid in chunks_dict]
    print(f"✅ Processing {len(valid_ids):,} chunks with embeddings")

    # Get embeddings and metadata in same order
    print("   Building embedding matrix...")
    embeddings_matrix = np.vstack([embeddings_dict[cid] for cid in valid_ids])

    print("   Collecting metadata...")
    metadata_list = [chunks_dict[cid] for cid in valid_ids]

    # Create FAISS index
    dimension = embeddings_matrix.shape[1]
    print(f"   Embedding dimension: {dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)

    print(f"✅ FAISS index built with {index.ntotal:,} vectors")

    # Save FAISS index
    faiss_dir = output_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = faiss_dir / f"{doc_type}.faiss"

    print(f"   Saving FAISS index to {faiss_path}...")
    faiss.write_index(index, str(faiss_path))
    print(f"✅ Saved FAISS index: {faiss_path}")

    # Save metadata
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{doc_type}_metadata.jsonl"

    print(f"   Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for chunk_data in metadata_list:
            # Combine chunk data with its metadata
            full_metadata = {
                'id': chunk_data['id'],
                'doc_type': chunk_data['meta'].get('doc_type', doc_type),
                'source_filepath': chunk_data['doc_id'],
                'text': chunk_data['text'],
                'token_count': chunk_data['token_count'],
                **chunk_data['meta']  # Include all original metadata
            }
            f.write(json.dumps(full_metadata, ensure_ascii=False) + '\n')

    print(f"✅ Saved metadata: {metadata_path}")

    return {
        'faiss_path': faiss_path,
        'metadata_path': metadata_path,
        'total_vectors': index.ntotal,
        'dimension': dimension
    }

def main():
    parser = argparse.ArgumentParser(
        description="Process OpenAI batch embeddings and build FAISS index"
    )
    parser.add_argument("--batch-output", type=str, required=True,
                       help="Path to batch output JSONL file")
    parser.add_argument("--chunks", type=str, required=True,
                       help="Path to chunks TSV file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for FAISS and metadata")
    parser.add_argument("--doc-type", type=str, default="outlook_eml",
                       help="Document type (default: outlook_eml)")

    args = parser.parse_args()

    print("="*80)
    print("  PROCESS BATCH EMBEDDINGS → FAISS INDEX (FIXED VERSION)")
    print("="*80)

    # Convert to Path objects
    batch_output_path = Path(args.batch_output)
    chunks_path = Path(args.chunks)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not batch_output_path.exists():
        print(f"❌ Batch output file not found: {batch_output_path}")
        return 1

    if not chunks_path.exists():
        print(f"❌ Chunks file not found: {chunks_path}")
        return 1

    print(f"\nInput files:")
    print(f"  Batch output: {batch_output_path} ({batch_output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Chunks TSV: {chunks_path} ({chunks_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Load data
    chunks_dict = load_chunks_from_tsv(chunks_path)
    embeddings_dict = load_batch_output(batch_output_path)

    # Verify we have data
    if not chunks_dict:
        print("❌ No chunks loaded")
        return 1

    if not embeddings_dict:
        print("❌ No embeddings loaded")
        return 1

    # Build FAISS index
    result = build_faiss_index(embeddings_dict, chunks_dict, output_dir, args.doc_type)

    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    print(f"✅ FAISS Index: {result['faiss_path']}")
    print(f"✅ Metadata: {result['metadata_path']}")
    print(f"✅ Total vectors: {result['total_vectors']:,}")
    print(f"✅ Dimension: {result['dimension']}")
    print(f"\n💡 Note: Only {result['total_vectors']:,} unique chunks were embedded")
    print(f"   (Deduplication reduced 83,131 chunks to {result['total_vectors']:,} unique)")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
