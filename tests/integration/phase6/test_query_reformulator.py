"""
Test Query Reformulator (Phase 6, Task 6.2)

Tests the QueryReformulator's ability to:
- Detect vague queries and reformulate them
- Leave specific queries unchanged
- Generate 2-4 specific sub-queries from vague queries
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.agents.query_reformulator import QueryReformulator


def test_query_reformulator():
    """Test QueryReformulator on various query types."""

    print("=" * 80)
    print("QUERY REFORMULATOR TEST (Phase 6, Task 6.2)")
    print("=" * 80)

    reformulator = QueryReformulator(model="gpt-4o-mini")

    # Test queries
    test_cases = [
        # Specific queries (should NOT be reformulated)
        {
            "query": "How do I configure facets in Primo VE?",
            "expected_type": "specific",
            "description": "Specific factual question"
        },
        {
            "query": "What is the process for updating CSL files in GitHub?",
            "expected_type": "specific",
            "description": "Specific process question"
        },

        # Vague queries (SHOULD be reformulated)
        {
            "query": "What happened with Primo VE last month?",
            "expected_type": "vague",
            "description": "Vague temporal query",
            "context": {"current_date": "2025-11-25", "current_month": "November 2025"}
        },
        {
            "query": "Tell me about NDE",
            "expected_type": "vague",
            "description": "Vague exploratory query"
        },
        {
            "query": "Problems with Primo",
            "expected_type": "vague",
            "description": "Vague problem statement"
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected_type = test["expected_type"]
        description = test["description"]
        context = test.get("context")

        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}: {description}")
        print(f"{'='*80}")
        print(f"Query: \"{query}\"")
        print(f"Expected: {expected_type}")
        if context:
            print(f"Context: {context}")
        print("-" * 80)

        # Reformulate query
        sub_queries = reformulator.reformulate(query, context=context)

        # Display results
        print(f"\n📊 Reformulation Results:")
        print(f"  - Number of sub-queries: {len(sub_queries)}")

        if len(sub_queries) == 1 and sub_queries[0] == query:
            print(f"  - Status: Query unchanged (specific)")
        else:
            print(f"  - Status: Query reformulated (vague)")

        print(f"\n💡 Sub-queries:")
        for j, sq in enumerate(sub_queries, 1):
            print(f"  {j}. {sq}")

        # Validate result
        if expected_type == "specific":
            # Should return original query unchanged
            success = (len(sub_queries) == 1 and sub_queries[0] == query)
            if success:
                print(f"\n✅ PASS: Query correctly identified as specific (not reformulated)")
            else:
                print(f"\n❌ FAIL: Specific query was reformulated (should be unchanged)")

        else:  # vague
            # Should return 2-4 reformulated queries
            success = (
                len(sub_queries) >= 2 and
                len(sub_queries) <= 4 and
                sub_queries[0] != query  # Reformulated, not original
            )
            if success:
                print(f"\n✅ PASS: Vague query correctly reformulated into {len(sub_queries)} sub-queries")
            else:
                print(f"\n❌ FAIL: Expected 2-4 reformulated queries, got {len(sub_queries)}")

        results.append({
            "query": query,
            "expected_type": expected_type,
            "sub_queries": sub_queries,
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
        print(f"      Expected: {r['expected_type']}, Got: {len(r['sub_queries'])} sub-queries")

    # Final verdict
    print(f"\n{'='*80}")
    if pass_rate >= 80:
        print("✅ OVERALL: Query Reformulator is working well!")
        print("   - Correctly identifies specific queries (no reformulation)")
        print("   - Reformulates vague queries into 2-4 specific sub-queries")
        return 0
    else:
        print("⚠️ OVERALL: Query Reformulator needs improvement")
        print(f"   - Pass rate: {pass_rate:.0f}% (target: 80%+)")
        return 1


if __name__ == "__main__":
    try:
        exit_code = test_query_reformulator()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
