#!/usr/bin/env python3
"""
Test EmailOrchestratorAgent directly.
"""
import sys
from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.agents.email_orchestrator import EmailOrchestratorAgent

def main():
    project_path = Path("data/projects/Primo_List_2")
    query = "What is the latest information about Citation Styles?"

    print("=" * 80)
    print("  TESTING EmailOrchestratorAgent DIRECTLY")
    print("=" * 80)
    print(f"\nProject: {project_path}")
    print(f"Query: {query}\n")

    # Load project
    project = ProjectManager(project_path)

    # Create EmailOrchestratorAgent
    print("🔧 Creating EmailOrchestratorAgent...")
    orchestrator = EmailOrchestratorAgent(project)

    # Retrieve (with auto top_k adjustment like the pipeline does)
    print("🔍 Retrieving with orchestrator...")
    try:
        result = orchestrator.retrieve(query, top_k=None, max_tokens=2000)

        print(f"\n✅ Orchestrator result:")
        print(f"   Intent: {result['intent']['primary_intent']} (confidence: {result['intent']['confidence']:.2f})")
        print(f"   Strategy: {result['strategy']['primary']}")
        print(f"   Chunks: {len(result['chunks'])}")
        print(f"   Metadata chunk_count: {result['metadata'].get('chunk_count', 'N/A')}")

        if len(result['chunks']) > 0:
            print(f"\n📧 First 5 chunks:")
            for i, chunk in enumerate(result['chunks'][:5], 1):
                subject = chunk.meta.get("subject", "N/A")
                print(f"{i}. {subject[:80]}")
        else:
            print("\n⚠️ No chunks in result!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    sys.exit(main())
