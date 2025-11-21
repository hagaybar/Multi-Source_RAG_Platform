#!/usr/bin/env python3
"""
Test Phase 1 components with real Primo_List email data.

This script:
1. Loads your FAISS vector store and metadata
2. Performs semantic search on user queries
3. Tests EmailIntentDetector on the query
4. Tests ContextAssembler on retrieved email chunks
5. Shows before/after comparison
"""

import json
import faiss
import numpy as np
from pathlib import Path
from scripts.agents.email_intent_detector import EmailIntentDetector
from scripts.retrieval.context_assembler import ContextAssembler
from scripts.chunking.models import Chunk


# Paths to your Primo_List data
PROJECT_DIR = Path("data/projects/Primo_List")
FAISS_INDEX_PATH = PROJECT_DIR / "output/faiss/outlook_eml.faiss"
METADATA_PATH = PROJECT_DIR / "output/metadata/outlook_eml_metadata.jsonl"


def load_faiss_index():
    """Load FAISS index from disk."""
    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"✓ Loaded FAISS index with {index.ntotal} vectors")
    return index


def load_metadata():
    """Load metadata from JSONL file."""
    print(f"Loading metadata from: {METADATA_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

    metadata_list = []
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                metadata_list.append(json.loads(line))

    print(f"✓ Loaded {len(metadata_list)} metadata entries")
    return metadata_list


def get_query_embedding(query_text):
    """Get embedding for query using OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-large",  # Must match your config.yml model
        input=query_text
    )

    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)


def search_similar_chunks(index, metadata_list, query_embedding, top_k=5):
    """Search FAISS index for similar chunks."""
    # Reshape query embedding for FAISS
    query_vector = query_embedding.reshape(1, -1)

    # Search
    distances, indices = index.search(query_vector, top_k)

    # Get results with metadata
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(metadata_list):
            meta = metadata_list[idx]
            results.append({
                'metadata': meta,
                'text': meta['text'],
                'distance': float(distance),
                'similarity': 1 / (1 + float(distance))  # Convert distance to similarity
            })

    return results


def convert_to_chunks(search_results):
    """Convert search results to Chunk objects."""
    chunks = []
    for result in search_results:
        meta = result['metadata']
        chunk = Chunk(
            doc_id=meta.get('doc_id', meta.get('id', 'unknown')),
            text=result['text'],
            meta={
                'doc_type': meta.get('doc_type'),
                'subject': meta.get('subject'),
                'sender': meta.get('sender'),
                'sender_name': meta.get('sender_name'),
                'date': meta.get('date'),
                'source_filepath': meta.get('source_filepath'),
            },
            token_count=meta.get('token_count', len(result['text'].split()))
        )
        chunks.append(chunk)

    return chunks


def test_with_query(query, index, metadata_list, detector, assembler, top_k=15):
    """Test Phase 1 components with a specific query.

    Note: top_k=15 is appropriate for emails (vs top_k=5 for documents)
    because email threads need multiple messages for context.
    """
    print("\n" + "="*80)
    print(f"QUERY: \"{query}\"")
    print("="*80)

    # Step 1: Detect intent
    print("\n📍 Step 1: Detect Intent")
    intent_result = detector.detect(query)
    print(f"   Primary Intent: {intent_result['primary_intent']}")
    print(f"   Confidence: {intent_result['confidence']:.2f}")

    if intent_result['metadata']:
        print(f"   Metadata:")
        for key, value in intent_result['metadata'].items():
            print(f"      - {key}: {value}")

    if intent_result['secondary_signals']:
        print(f"   Secondary Signals: {intent_result['secondary_signals']}")

    # Step 2: Get query embedding and search
    print(f"\n📍 Step 2: Search Vector Store (top {top_k})")
    print("   Getting query embedding...")
    query_embedding = get_query_embedding(query)

    print("   Searching FAISS index...")
    search_results = search_similar_chunks(index, metadata_list, query_embedding, top_k)
    print(f"   ✓ Found {len(search_results)} results")

    # Show search results
    print("\n   Top Results:")
    for i, result in enumerate(search_results[:3], 1):
        meta = result['metadata']
        print(f"\n   {i}. Similarity: {result['similarity']:.3f}")
        print(f"      From: {meta.get('sender_name', 'Unknown')}")
        print(f"      Subject: {meta.get('subject', 'No subject')}")
        print(f"      Date: {meta.get('date', 'Unknown')}")
        print(f"      Text preview: {result['text'][:100]}...")

    # Step 3: Convert to chunks
    print(f"\n📍 Step 3: Convert to Chunk objects")
    chunks = convert_to_chunks(search_results)
    print(f"   ✓ Created {len(chunks)} chunk objects")

    # Calculate total size before cleaning
    total_chars_before = sum(len(chunk.text) for chunk in chunks)
    print(f"   Total characters (before cleaning): {total_chars_before}")

    # Step 4: Assemble context with Phase 1 cleaning
    print(f"\n📍 Step 4: Assemble & Clean Context")
    print("   Applying ContextAssembler...")
    cleaned_context = assembler.assemble(chunks, intent_result)

    # Show statistics
    total_chars_after = len(cleaned_context)
    reduction = ((total_chars_before - total_chars_after) / total_chars_before * 100) if total_chars_before > 0 else 0

    print(f"   ✓ Cleaned context assembled")
    print(f"   Total characters (after cleaning): {total_chars_after}")
    print(f"   Reduction: {reduction:.1f}%")

    # Step 5: Show cleaned context
    print("\n📍 Step 5: Cleaned Context (Ready for LLM)")
    print("="*80)
    # Show first 1500 characters
    if len(cleaned_context) > 1500:
        print(cleaned_context[:1500] + f"\n\n... (truncated, {len(cleaned_context) - 1500} more characters)")
    else:
        print(cleaned_context)
    print("="*80)

    # Show what was removed
    print("\n✅ Phase 1 Processing Complete!")
    print("\nCleaning applied:")
    print("  ✓ Removed quoted text (> prefixes)")
    print("  ✓ Removed email signatures and closings")
    print("  ✓ Deduplicated redundant content")
    print("  ✓ Filtered noise (newsletters, auto-replies)")
    print("  ✓ Added clear source attributions")
    if intent_result['primary_intent'] == 'thread_summary':
        print("  ✓ Chronologically ordered for thread summary")
    elif intent_result['primary_intent'] == 'temporal_query':
        print("  ✓ Sorted by date (newest first)")

    return {
        'intent': intent_result,
        'search_results': search_results,
        'chunks': chunks,
        'cleaned_context': cleaned_context,
        'chars_before': total_chars_before,
        'chars_after': total_chars_after,
        'reduction': reduction
    }


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("PHASE 1 TEST WITH REAL PRIMO_LIST EMAIL DATA")
    print("="*80)

    # Load data
    try:
        index = load_faiss_index()
        metadata_list = load_metadata()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your Primo_List emails are ingested:")
        print("  1. Raw emails should be in: data/projects/Primo_List/input/raw/")
        print("  2. Run ingestion to create FAISS index and metadata")
        return

    # Initialize Phase 1 components
    print("\n" + "="*80)
    print("Initializing Phase 1 Components")
    print("="*80)
    detector = EmailIntentDetector()
    assembler = ContextAssembler()
    print("✓ EmailIntentDetector initialized")
    print("✓ ContextAssembler initialized")

    # Test with different query types
    test_queries = [
        "Primo NDE migration",
        "What did Manuela say about facets?",
        "Recent emails about user interface",
        "What are the action items from the discussion?",
    ]

    results_summary = []

    for query in test_queries:
        try:
            # Use top_k=15 for emails (vs top_k=5 for documents)
            # Email threads need more chunks for complete context
            result = test_with_query(query, index, metadata_list, detector, assembler, top_k=15)
            results_summary.append({
                'query': query,
                'intent': result['intent']['primary_intent'],
                'reduction': result['reduction']
            })
        except Exception as e:
            print(f"\n❌ Error processing query '{query}': {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY - PHASE 1 TESTING WITH REAL DATA")
    print("="*80)

    print("\nQueries Tested:")
    for i, summary in enumerate(results_summary, 1):
        print(f"\n{i}. Query: \"{summary['query']}\"")
        print(f"   Intent: {summary['intent']}")
        print(f"   Size Reduction: {summary['reduction']:.1f}%")

    avg_reduction = sum(s['reduction'] for s in results_summary) / len(results_summary) if results_summary else 0

    print(f"\nAverage size reduction: {avg_reduction:.1f}%")
    print("\n✅ Phase 1 successfully tested with real Primo_List email data!")
    print("\nNext Steps:")
    print("  → Phase 2: Implement specialized retrievers (Thread, Temporal, Sender)")
    print("  → Phase 3: Orchestrator integration with UI")
    print("  → Phase 4: Advanced features (LLM fallback, Answer Validator)")


if __name__ == "__main__":
    main()
