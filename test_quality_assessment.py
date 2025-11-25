"""
Test Retrieval Quality Assessment Agent

Tests the agent with different retrieval scenarios:
1. Good retrieval (high scores, relevant chunks)
2. Poor retrieval (low scores, few chunks)
3. Edge cases (no chunks, no rerank scores)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.core.project_manager import ProjectManager
from scripts.agents.email_orchestrator import EmailOrchestratorAgent
from scripts.agents.retrieval_quality_agent import RetrievalQualityAgent


def test_quality_assessment():
    """Test quality assessment with real retrieval results."""

    # Initialize project
    project_path = project_root / "data" / "projects" / "Primo_List_2"
    project = ProjectManager(project_path)

    print("=" * 80)
    print("RETRIEVAL QUALITY ASSESSMENT TEST")
    print("=" * 80)

    # Initialize components
    orchestrator = EmailOrchestratorAgent(project)
    quality_agent = RetrievalQualityAgent(model="gpt-4o-mini", use_llm_assessment=True)

    # Test queries with expected quality levels
    test_cases = [
        {
            "query": "What are the important issues with Chicago bibliography?",
            "expected_quality": "HIGH",
            "description": "Specific query with likely good retrieval"
        },
        {
            "query": "What happened with Primo VE last month?",
            "expected_quality": "MEDIUM",
            "description": "Temporal query that previously had issues"
        },
        {
            "query": "How do I configure facets in Primo?",
            "expected_quality": "HIGH",
            "description": "Factual query with clear answer"
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected_quality"]
        description = test_case["description"]

        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {description}")
        print(f"Query: {query}")
        print(f"Expected Quality: {expected}")
        print("-" * 80)

        # Retrieve chunks
        print("\n1️⃣ Retrieving chunks...")
        result = orchestrator.retrieve(query=query, top_k=10)
        chunks = result["chunks"]
        intent = result["intent"]

        print(f"   - Retrieved {len(chunks)} chunks")
        print(f"   - Intent: {intent['primary_intent']}")

        # Assess quality
        print("\n2️⃣ Assessing quality...")
        assessment = quality_agent.assess(query, chunks, intent)

        quality = assessment["quality"]
        confidence = assessment["confidence"]
        reasoning = assessment["reasoning"]
        recommendation = assessment["recommendation"]

        print(f"\n📊 Assessment Results:")
        print(f"   - Quality: {quality} (expected: {expected})")
        print(f"   - Confidence: {confidence:.2%}")
        print(f"   - Reasoning: {reasoning}")
        print(f"   - Recommendation: {recommendation}")

        # Heuristics
        heuristics = assessment["heuristics"]
        print(f"\n📈 Heuristics:")
        print(f"   - Chunk count: {heuristics['chunk_count']}")
        print(f"   - Avg rerank score: {heuristics['avg_rerank_score']:.3f}")
        print(f"   - Max rerank score: {heuristics['max_rerank_score']:.3f}")
        print(f"   - Score variance: {heuristics['score_variance']:.3f}")
        print(f"   - Heuristic quality: {heuristics['heuristic_quality']}")
        if heuristics['issues']:
            print(f"   - Issues: {', '.join(heuristics['issues'])}")

        # LLM assessment
        if "llm_assessment" in assessment:
            llm = assessment["llm_assessment"]
            print(f"\n🤖 LLM Assessment:")
            print(f"   - Can answer: {llm['can_answer']}")
            print(f"   - Confidence: {llm['confidence']:.2%}")
            print(f"   - Reasoning: {llm['reasoning']}")
            if llm.get('missing_info'):
                print(f"   - Missing: {llm['missing_info']}")

        # Check if matches expected
        match = quality == expected
        results.append({
            "query": query,
            "expected": expected,
            "actual": quality,
            "match": match,
            "confidence": confidence,
            "recommendation": recommendation
        })

        print(f"\n✅ Match: {match}")

    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print("=" * 80)

    matches = sum(1 for r in results if r["match"])
    total = len(results)

    print(f"\nResults: {matches}/{total} tests matched expected quality")
    print(f"\nDetailed Results:")

    for i, r in enumerate(results, 1):
        status = "✅" if r["match"] else "⚠️"
        print(f"{status} {i}. {r['query'][:60]}...")
        print(f"   Expected: {r['expected']}, Actual: {r['actual']}, "
              f"Confidence: {r['confidence']:.2%}, Rec: {r['recommendation']}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED")
    print("=" * 80)

    return results


if __name__ == "__main__":
    try:
        results = test_quality_assessment()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
