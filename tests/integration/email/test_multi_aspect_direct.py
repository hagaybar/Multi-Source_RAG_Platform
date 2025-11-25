#!/usr/bin/env python3
"""
Test MultiAspectRetriever directly.
"""
import sys
from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.email_multi_aspect_retriever import MultiAspectRetriever

def main():
    project_path = Path("data/projects/Primo_List_2")
    query = "What is the latest information about Citation Styles?"

    print("=" * 80)
    print("  TESTING MultiAspectRetriever DIRECTLY")
    print("=" * 80)
    print(f"\nProject: {project_path}")
    print(f"Query: {query}\n")

    # Load project
    project = ProjectManager(project_path)

    # Create MultiAspectRetriever
    print("🔧 Creating MultiAspectRetriever...")
    retriever = MultiAspectRetriever(project)

    # Create intent (temporal_query like the pipeline detected)
    intent = {
        "primary_intent": "temporal_query",
        "confidence": 0.8,
        "metadata": {}
    }

    print(f"🎯 Intent: {intent}\n")

    # Retrieve
    print("🔍 Retrieving chunks...")
    try:
        chunks = retriever.retrieve(query, intent=intent, top_k=15)
        print(f"✅ Retrieved {len(chunks)} chunks\n")

        if len(chunks) > 0:
            for i, chunk in enumerate(chunks[:5], 1):
                subject = chunk.meta.get("subject", "N/A")
                print(f"{i}. {subject[:80]}")
        else:
            print("⚠️ No chunks returned!")
            print("\n🔍 Checking FAISS index existence...")

            # Check if FAISS index exists
            faiss_path = project.metadata_dir.parent / "faiss" / "outlook_eml.faiss"
            metadata_path = project.metadata_dir / "outlook_eml_metadata.jsonl"

            print(f"   FAISS path: {faiss_path}")
            print(f"   Exists: {faiss_path.exists()}")
            print(f"   Metadata path: {metadata_path}")
            print(f"   Exists: {metadata_path.exists()}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    sys.exit(main())
