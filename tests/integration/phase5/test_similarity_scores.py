"""
Test that FAISS similarity scores are now correctly captured.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.core.project_manager import ProjectManager
from scripts.agents.email_orchestrator import EmailOrchestratorAgent


def test_similarity_scores():
    """Test that similarity scores are captured and reasonable."""

    # Initialize project
    project_path = project_root / "data" / "projects" / "Primo_List_2"
    project = ProjectManager(project_path)

    print("=" * 80)
    print("SIMILARITY SCORE CAPTURE TEST")
    print("=" * 80)

    # Initialize orchestrator
    orchestrator = EmailOrchestratorAgent(project)

    # Test query
    test_query = "Chicago bibliography issues"

    print(f"\n📝 Test Query: {test_query}")
    print("-" * 80)

    # Retrieve chunks
    print("\n1️⃣ Retrieving chunks...")
    result = orchestrator.retrieve(query=test_query, top_k=10)
    chunks = result["chunks"]

    print(f"   Retrieved {len(chunks)} chunks")

    # Check similarity scores
    print(f"\n2️⃣ Checking similarity scores...")

    chunks_with_similarity = [c for c in chunks if "similarity" in c.meta]
    chunks_with_rerank = [c for c in chunks if "rerank_score" in c.meta]

    print(f"   - Chunks with FAISS similarity: {len(chunks_with_similarity)}/{len(chunks)}")
    print(f"   - Chunks with rerank_score: {len(chunks_with_rerank)}/{len(chunks)}")

    if chunks_with_similarity:
        similarities = [c.meta["similarity"] for c in chunks_with_similarity]
        print(f"\n📊 FAISS Similarity Scores:")
        print(f"   - Range: [{min(similarities):.4f}, {max(similarities):.4f}]")
        print(f"   - Average: {sum(similarities)/len(similarities):.4f}")

        print(f"\n   Top 5 chunks by FAISS similarity:")
        for i, chunk in enumerate(chunks_with_similarity[:5], 1):
            sim = chunk.meta["similarity"]
            subject = chunk.meta.get("subject", "N/A")[:60]
            preview = chunk.text[:80].replace("\n", " ")

            print(f"   {i}. Similarity: {sim:.4f}")
            print(f"      Subject: {subject}")
            print(f"      Preview: {preview}...")
            print()
    else:
        print("   ⚠️ No FAISS similarity scores found!")

    if chunks_with_rerank:
        rerank_scores = [c.meta["rerank_score"] for c in chunks_with_rerank]
        print(f"\n📊 Rerank Scores:")
        print(f"   - Range: [{min(rerank_scores):.4f}, {max(rerank_scores):.4f}]")
        print(f"   - Average: {sum(rerank_scores)/len(rerank_scores):.4f}")

    # Check quality assessment
    qa = result["quality_assessment"]
    print(f"\n3️⃣ Quality Assessment:")
    print(f"   - Quality: {qa['quality']}")
    print(f"   - Confidence: {qa['confidence']:.2%}")
    print(f"   - Heuristics used FAISS scores: {qa['heuristics'].get('has_rerank_scores', False)}")

    print("\n" + "=" * 80)

    if chunks_with_similarity and min(similarities) > 0:
        print("✅ TEST PASSED: Similarity scores are captured and non-zero!")
    else:
        print("❌ TEST FAILED: Similarity scores missing or zero!")

    print("=" * 80)

    return chunks_with_similarity


if __name__ == "__main__":
    try:
        chunks = test_similarity_scores()
        sys.exit(0 if chunks else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
