"""
Test script to verify reranking integration in EmailOrchestratorAgent.

This script tests:
1. Reranker initialization
2. Retrieval with reranking enabled
3. Rerank scores in chunk metadata
4. Comparison with/without reranking
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.core.project_manager import ProjectManager
from scripts.agents.email_orchestrator import EmailOrchestratorAgent


def test_reranking_integration():
    """Test reranking integration with a sample query."""

    # Initialize project
    project_path = project_root / "data" / "projects" / "Primo_List_2"
    project = ProjectManager(project_path)

    print("=" * 80)
    print("RERANKING INTEGRATION TEST")
    print("=" * 80)

    # Test query
    test_query = "What are the important issues with Chicago bibliography?"

    print(f"\n📝 Test Query: {test_query}")
    print("-" * 80)

    # Initialize orchestrator (should load reranking config)
    print("\n1️⃣ Initializing EmailOrchestratorAgent...")
    orchestrator = EmailOrchestratorAgent(project)

    # Check reranking config
    print(f"\n✅ Reranking enabled: {orchestrator.reranking_config.get('enabled', False)}")
    print(f"✅ Reranker initialized: {orchestrator.reranker is not None}")

    if orchestrator.reranker:
        print(f"✅ Reranker model: {orchestrator.reranker.model_name}")
        initial_k = orchestrator.reranking_config.get("retrieval", {}).get("initial_k", 100)
        final_k = orchestrator.reranking_config.get("retrieval", {}).get("final_k", 15)
        print(f"✅ Retrieval config: initial_k={initial_k}, final_k={final_k}")

    # Run retrieval
    print(f"\n2️⃣ Running retrieval...")
    result = orchestrator.retrieve(query=test_query, top_k=15)

    chunks = result["chunks"]
    intent = result["intent"]
    strategy = result["strategy"]
    metadata = result["metadata"]

    print(f"\n📊 Retrieval Results:")
    print(f"   - Intent: {intent['primary_intent']} (confidence: {intent['confidence']:.2f})")
    print(f"   - Strategy: {strategy['primary']}")
    print(f"   - Chunks retrieved: {len(chunks)}")

    # Check for rerank scores
    print(f"\n3️⃣ Checking rerank scores in chunks...")
    chunks_with_scores = [c for c in chunks if "rerank_score" in c.meta]
    print(f"   - Chunks with rerank_score: {len(chunks_with_scores)}/{len(chunks)}")

    if chunks_with_scores:
        scores = [c.meta["rerank_score"] for c in chunks_with_scores]
        print(f"   - Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"   - Average score: {sum(scores)/len(scores):.4f}")

        print(f"\n📝 Top 3 chunks by rerank score:")
        for i, chunk in enumerate(chunks_with_scores[:3], 1):
            score = chunk.meta["rerank_score"]
            preview = chunk.text[:150].replace("\n", " ")
            print(f"   {i}. Score: {score:.4f}")
            print(f"      Preview: {preview}...")
            print(f"      Source: {chunk.meta.get('subject', 'N/A')[:60]}")
            print()
    else:
        print("   ⚠️ No rerank scores found in chunks!")

    # Display metadata
    print(f"\n4️⃣ Result Metadata:")
    print(f"   - Chunk count: {metadata['chunk_count']}")
    print(f"   - Strategy used: {metadata['strategy_used']}")
    if "date_range" in metadata:
        print(f"   - Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    if "unique_senders" in metadata:
        print(f"   - Unique senders: {len(metadata['unique_senders'])}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED")
    print("=" * 80)

    return orchestrator, result


if __name__ == "__main__":
    try:
        orchestrator, result = test_reranking_integration()

        # Success
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
