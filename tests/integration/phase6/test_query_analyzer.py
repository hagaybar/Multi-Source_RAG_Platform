"""
Test Query Analyzer (Phase 6, Task 6.1)

Tests the QueryAnalyzer's ability to:
- Identify specific vs vague queries
- Extract domain entities
- Assess intent quality
- Suggest clarifications and reformulations
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.agents.query_analyzer import QueryAnalyzer


def test_query_analyzer():
    """Test QueryAnalyzer on various query types."""

    print("=" * 80)
    print("QUERY ANALYZER TEST (Phase 6, Task 6.1)")
    print("=" * 80)

    analyzer = QueryAnalyzer(model="gpt-4o-mini")

    # Test queries: mix of specific and vague
    test_queries = [
        # Specific queries (should score high)
        {
            "query": "How do I configure facets in Primo VE?",
            "expected": "specific",
            "description": "Specific factual question"
        },
        {
            "query": "What is the process for updating CSL files in GitHub?",
            "expected": "specific",
            "description": "Specific process question"
        },
        {
            "query": "When will the Chicago 18th edition be available in Primo?",
            "expected": "specific",
            "description": "Specific temporal question"
        },

        # Vague queries (should be flagged)
        {
            "query": "What happened with Primo VE last month?",
            "expected": "vague",
            "description": "Vague temporal query"
        },
        {
            "query": "Tell me about NDE",
            "expected": "vague",
            "description": "Vague exploratory query"
        },
        {
            "query": "Problems with Primo",
            "expected": "vague",
            "description": "Vague problem statement"
        }
    ]

    results = []

    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected = test["expected"]
        description = test["description"]

        print(f"\n{'='*80}")
        print(f"TEST {i}/6: {description}")
        print(f"{'='*80}")
        print(f"Query: \"{query}\"")
        print(f"Expected: {expected}")
        print("-" * 80)

        # Analyze query
        analysis = analyzer.analyze(query)

        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"  - Is Specific: {analysis.is_specific}")
        print(f"  - Intent Quality: {analysis.intent_quality:.2f}")
        print(f"  - Query Type: {analysis.query_type}")
        print(f"  - Entities: {analysis.entities}")
        print(f"  - Issues: {analysis.issues}")

        if analysis.clarification_questions:
            print(f"\n❓ Clarification Questions:")
            for q in analysis.clarification_questions:
                print(f"  - {q}")

        if analysis.suggested_reformulations:
            print(f"\n💡 Suggested Reformulations:")
            for j, ref in enumerate(analysis.suggested_reformulations, 1):
                print(f"  {j}. {ref}")

        print(f"\n📝 Reasoning: {analysis.reasoning}")

        # Check if analysis matches expectation
        if expected == "specific":
            success = analysis.is_specific and analysis.intent_quality > 0.6
        else:  # vague
            success = not analysis.is_specific or analysis.intent_quality < 0.7

        if success:
            print(f"\n✅ PASS: Analysis correctly identified query as {expected}")
        else:
            print(f"\n❌ FAIL: Expected {expected}, but got different classification")

        results.append({
            "query": query,
            "expected": expected,
            "analysis": analysis,
            "success": success
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["success"])
    total = len(results)
    pass_rate = (passed / total) * 100

    print(f"\nTests Passed: {passed}/{total} ({pass_rate:.0f}%)")

    print(f"\n📋 Detailed Results:")
    for i, r in enumerate(results, 1):
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"  {i}. {status} - \"{r['query'][:50]}...\"")
        print(f"      Expected: {r['expected']}, "
              f"Specific: {r['analysis'].is_specific}, "
              f"Quality: {r['analysis'].intent_quality:.2f}")

    # Final verdict
    print(f"\n{'='*80}")
    if pass_rate >= 80:
        print("✅ OVERALL: Query Analyzer is working well!")
        print("   - Correctly identifies specific vs vague queries")
        print("   - Extracts domain entities")
        print("   - Suggests useful reformulations")
        return 0
    else:
        print("⚠️ OVERALL: Query Analyzer needs improvement")
        print(f"   - Pass rate: {pass_rate:.0f}% (target: 80%+)")
        return 1


if __name__ == "__main__":
    try:
        exit_code = test_query_analyzer()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
