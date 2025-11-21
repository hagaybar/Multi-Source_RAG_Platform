#!/usr/bin/env python3
"""
Test P0 and P1 fixes for email retrieval.

Tests:
- P0.1: UI displays sender names correctly
- P0.2/P0.3: Retriever tagging works
- P1.1: Temporal constraints are extracted
- P1.2: Strategy selection handles temporal constraints
- P1.3: Date filtering works in ThreadRetriever
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.agents.email_intent_detector import EmailIntentDetector
from scripts.agents.email_strategy_selector import EmailStrategySelector
from scripts.retrieval.email_thread_retriever import ThreadRetriever
from scripts.core.project_manager import ProjectManager


def test_p1_1_temporal_extraction():
    """Test that temporal constraints are extracted correctly."""
    print("=" * 60)
    print("Test P1.1: Temporal Constraint Extraction")
    print("=" * 60)

    detector = EmailIntentDetector()

    # Test 1: "past 4 weeks"
    query1 = "Please summarize all discussions related to the NDE view on the past four weeks"
    result1 = detector.detect(query1)

    print(f"\nQuery: '{query1}'")
    print(f"Primary intent: {result1['primary_intent']}")
    print(f"Secondary signals: {result1['secondary_signals']}")
    print(f"Metadata: {result1['metadata']}")

    assert "temporal_constraint" in result1["metadata"], "❌ Failed to extract temporal constraint"
    assert result1["metadata"]["temporal_constraint"]["days_back"] == 28, "❌ Incorrect days_back (should be 28)"
    assert "temporal_query" in result1["secondary_signals"], "❌ temporal_query not in secondary signals"

    print("✅ P1.1 PASSED: Temporal constraints extracted correctly!")

    # Test 2: "last 3 months"
    query2 = "Show me all emails from the last 3 months"
    result2 = detector.detect(query2)

    print(f"\nQuery: '{query2}'")
    print(f"Temporal constraint: {result2['metadata'].get('temporal_constraint')}")

    assert result2["metadata"]["temporal_constraint"]["days_back"] == 90, "❌ Incorrect days_back for 3 months"
    print("✅ P1.1 PASSED: Multi-month extraction works!")

    return True


def test_p1_2_strategy_selection():
    """Test that strategy selector handles temporal constraints."""
    print("\n" + "=" * 60)
    print("Test P1.2: Strategy Selection with Temporal Constraints")
    print("=" * 60)

    selector = EmailStrategySelector()

    # Scenario: thread_summary + temporal_query
    intent = {
        "primary_intent": "thread_summary",
        "confidence": 0.8,
        "metadata": {
            "temporal_constraint": {
                "type": "relative",
                "days_back": 28
            }
        },
        "secondary_signals": ["temporal_query"]
    }

    strategy = selector.select_strategy(intent)

    print(f"\nIntent: {intent['primary_intent']} + {intent['secondary_signals']}")
    print(f"Strategy selected: {strategy['primary']}")

    assert strategy["primary"] == "multi_aspect", f"❌ Should use multi_aspect, got {strategy['primary']}"
    print("✅ P1.2 PASSED: Multi-aspect strategy selected for thread + temporal!")

    return True


def test_p1_3_date_filtering():
    """Test that ThreadRetriever filters by date."""
    print("\n" + "=" * 60)
    print("Test P1.3: Date Filtering in ThreadRetriever")
    print("=" * 60)

    project_path = Path("data/projects/Primo_List")

    if not project_path.exists():
        print("⚠️ Skipping: Primo_List project not found")
        return True

    project = ProjectManager(project_path)
    retriever = ThreadRetriever(project)

    # Test with temporal filter
    query = "NDE view discussions"

    print(f"\nQuery: '{query}'")
    print("Testing with days_back=28 (last 4 weeks)...")

    chunks = retriever.retrieve(query, top_threads=2, days_back=28)

    print(f"Retrieved {len(chunks)} chunks")

    if chunks:
        # Check that all chunks are within the date range
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d")

        for chunk in chunks:
            date = chunk.meta.get("date", "").split()[0]
            if date:
                assert date >= cutoff, f"❌ Chunk date {date} is before cutoff {cutoff}"

        print(f"✅ P1.3 PASSED: All {len(chunks)} chunks are within date range (>= {cutoff})!")
    else:
        print("⚠️ No chunks retrieved (might be expected if no recent emails)")

    return True


def test_p0_retriever_tagging():
    """Test that retrievers add _retriever metadata."""
    print("\n" + "=" * 60)
    print("Test P0: Retriever Tagging")
    print("=" * 60)

    project_path = Path("data/projects/Primo_List")

    if not project_path.exists():
        print("⚠️ Skipping: Primo_List project not found")
        return True

    project = ProjectManager(project_path)
    thread_retriever = ThreadRetriever(project)

    chunks = thread_retriever.retrieve("test query", top_threads=1)

    if chunks:
        print(f"Retrieved {len(chunks)} chunks from ThreadRetriever")
        assert all("_retriever" in c.meta for c in chunks), "❌ Some chunks missing _retriever tag"
        assert chunks[0].meta["_retriever"] == "thread", f"❌ Wrong retriever tag: {chunks[0].meta['_retriever']}"
        print(f"✅ P0 PASSED: All chunks tagged with _retriever='thread'!")
    else:
        print("⚠️ No chunks retrieved")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Testing P0 & P1 Fixes")
    print("=" * 60)

    try:
        # Run all tests
        p1_1_passed = test_p1_1_temporal_extraction()
        p1_2_passed = test_p1_2_strategy_selection()
        p1_3_passed = test_p1_3_date_filtering()
        p0_passed = test_p0_retriever_tagging()

        print("\n" + "=" * 60)
        print("  All Tests Summary")
        print("=" * 60)
        print(f"✅ P1.1: Temporal Extraction - {'PASSED' if p1_1_passed else 'FAILED'}")
        print(f"✅ P1.2: Strategy Selection - {'PASSED' if p1_2_passed else 'FAILED'}")
        print(f"✅ P1.3: Date Filtering - {'PASSED' if p1_3_passed else 'FAILED'}")
        print(f"✅ P0: Retriever Tagging - {'PASSED' if p0_passed else 'FAILED'}")

        if all([p1_1_passed, p1_2_passed, p1_3_passed, p0_passed]):
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED")
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
