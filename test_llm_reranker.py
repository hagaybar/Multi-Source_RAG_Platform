"""
Quick test of LLM-based reranking with gpt-4o-mini.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.core.project_manager import ProjectManager
from scripts.agents.email_orchestrator import EmailOrchestratorAgent


def test_llm_reranker():
    """Test LLM reranker with a sample query."""

    # Initialize project
    project_path = project_root / "data" / "projects" / "Primo_List_2"
    project = ProjectManager(project_path)

    print("=" * 80)
    print("LLM RERANKER TEST (gpt-4o-mini)")
    print("=" * 80)

    # Test query
    test_query = "What are the pressing issues with Chicago bibliography?"

    print(f"\n📝 Query: {test_query}")
    print("-" * 80)

    # Initialize orchestrator
    print("\n1️⃣ Initializing EmailOrchestratorAgent with LLM reranker...")
    orchestrator = EmailOrchestratorAgent(project)

    # Check configuration
    reranker_type = orchestrator.reranking_config.get("type", "unknown")
    print(f"   - Reranker type: {reranker_type}")
    print(f"   - Reranker initialized: {orchestrator.reranker is not None}")

    if orchestrator.reranker:
        if reranker_type == "llm":
            print(f"   - LLM model: {orchestrator.reranker.model}")
            print(f"   - Batch size: {orchestrator.reranker.batch_size}")

    # Run retrieval with top_k=10 (small for faster testing)
    print(f"\n2️⃣ Running retrieval with top_k=10...")
    result = orchestrator.retrieve(query=test_query, top_k=10)

    chunks = result["chunks"]
    print(f"   - Retrieved {len(chunks)} chunks")

    # Check for rerank scores
    print(f"\n3️⃣ Checking rerank scores...")
    chunks_with_scores = [c for c in chunks if "rerank_score" in c.meta]
    print(f"   - Chunks with rerank_score: {len(chunks_with_scores)}/{len(chunks)}")

    if chunks_with_scores:
        scores = [c.meta["rerank_score"] for c in chunks_with_scores]
        raw_scores = [c.meta.get("rerank_score_raw", 0) for c in chunks_with_scores]

        print(f"   - Normalized scores (0-1): [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"   - Raw scores (0-10): [{min(raw_scores):.1f}, {max(raw_scores):.1f}]")

        print(f"\n📊 Top 5 chunks by relevance:")
        for i, chunk in enumerate(chunks_with_scores[:5], 1):
            score = chunk.meta["rerank_score"]
            raw_score = chunk.meta.get("rerank_score_raw", 0)
            subject = chunk.meta.get("subject", "N/A")[:60]
            preview = chunk.text[:100].replace("\n", " ")

            print(f"\n   {i}. Score: {raw_score:.1f}/10 ({score:.3f})")
            print(f"      Subject: {subject}")
            print(f"      Preview: {preview}...")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED")
    print("=" * 80)

    return orchestrator, result


if __name__ == "__main__":
    try:
        orchestrator, result = test_llm_reranker()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
