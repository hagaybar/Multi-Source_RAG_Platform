#!/usr/bin/env python3
"""
Debug script to test multi_aspect retrieval strategy.
"""
import sys
from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager

def main():
    project_path = Path("data/projects/Primo_List_2")
    query = "What is the latest information about Citation Styles?"

    print("=" * 80)
    print("  TESTING MULTI_ASPECT RETRIEVAL")
    print("=" * 80)
    print(f"\nProject: {project_path}")
    print(f"Query: {query}\n")

    # Load project
    project = ProjectManager(project_path)

    # Create retrieval manager
    retriever = RetrievalManager(project)

    # Test multi_aspect strategy
    print("🔍 Testing multi_aspect strategy...")
    try:
        chunks = retriever.retrieve(query, top_k=15, strategy="multi_aspect")
        print(f"✅ Retrieved {len(chunks)} chunks with multi_aspect\n")

        if len(chunks) > 0:
            for i, chunk in enumerate(chunks[:5], 1):
                subject = chunk.meta.get("subject", "N/A")
                print(f"{i}. {subject[:80]}")
        else:
            print("⚠️ No chunks returned!")

    except Exception as e:
        print(f"❌ Error with multi_aspect: {e}")
        import traceback
        traceback.print_exc()

    # Test late_fusion strategy
    print("\n" + "-" * 80)
    print("🔍 Testing late_fusion strategy...")
    try:
        chunks = retriever.retrieve(query, top_k=15, strategy="late_fusion")
        print(f"✅ Retrieved {len(chunks)} chunks with late_fusion\n")

        if len(chunks) > 0:
            for i, chunk in enumerate(chunks[:5], 1):
                subject = chunk.meta.get("subject", "N/A")
                print(f"{i}. {subject[:80]}")
        else:
            print("⚠️ No chunks returned!")

    except Exception as e:
        print(f"❌ Error with late_fusion: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    sys.exit(main())
