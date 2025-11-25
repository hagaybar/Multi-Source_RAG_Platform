#!/usr/bin/env python3
"""
Quick test to verify Chicago citation emails are now searchable.
"""
import sys
from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager

def main():
    project_path = Path("data/projects/Primo_List_2")
    query = "What is the latest information about Citation Styles?"

    print("=" * 80)
    print("  TESTING CHICAGO CITATION QUERY")
    print("=" * 80)
    print(f"\nProject: {project_path}")
    print(f"Query: {query}\n")

    # Load project
    project = ProjectManager(project_path)

    # Create retrieval manager
    retriever = RetrievalManager(project)

    # Retrieve chunks
    print("🔍 Retrieving chunks...")
    chunks = retriever.retrieve(query, top_k=15, strategy="late_fusion")

    print(f"\n✅ Retrieved {len(chunks)} chunks\n")

    # Check for Chicago citation emails
    chicago_count = 0
    for i, chunk in enumerate(chunks, 1):
        subject = chunk.meta.get("subject", "")
        if "Chicago" in subject or "citation" in subject.lower():
            chicago_count += 1
            print(f"{i}. [Chicago Found!] {subject[:80]}")
            print(f"   Date: {chunk.meta.get('date', 'N/A')}")
            print(f"   Similarity: {chunk.meta.get('similarity', 'N/A'):.4f}")
            print(f"   Text preview: {chunk.text[:150]}...")
            print()

    print("=" * 80)
    print(f"📊 RESULTS:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Chicago-related: {chicago_count}")
    print("=" * 80)

    if chicago_count > 0:
        print("\n✅ SUCCESS: Chicago citation emails are now searchable!")
        return 0
    else:
        print("\n❌ ISSUE: No Chicago citation emails found in results")
        return 1

if __name__ == "__main__":
    sys.exit(main())
