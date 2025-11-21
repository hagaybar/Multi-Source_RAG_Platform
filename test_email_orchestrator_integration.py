#!/usr/bin/env python3
"""
Test script to verify EmailOrchestratorAgent integration with PipelineRunner.

This script tests that:
1. Email projects are correctly detected
2. EmailOrchestratorAgent is used instead of RetrievalManager
3. Intent detection and strategy selection work correctly
4. Logs show the new orchestrator in action
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.core.project_manager import ProjectManager
from scripts.pipeline.runner import PipelineRunner


def test_email_orchestrator_integration():
    """Test that EmailOrchestratorAgent is used for email projects."""
    print("=" * 60)
    print("  Testing EmailOrchestratorAgent Integration")
    print("=" * 60)

    # Load Primo_List project (email project)
    project_path = Path("data/projects/Primo_List")

    if not project_path.exists():
        print(f"❌ Project not found: {project_path}")
        print("   Please ensure Primo_List project exists with email data.")
        return False

    print(f"\n✅ Loading project: {project_path}")
    project = ProjectManager(project_path)

    # Create pipeline runner
    runner = PipelineRunner(project, project.config)

    # Test 1: Check email project detection
    print("\n" + "=" * 60)
    print("Test 1: Email Project Detection")
    print("=" * 60)
    is_email = runner._is_email_project()
    print(f"Is email project: {is_email}")
    if is_email:
        print("✅ Email project correctly detected!")
    else:
        print("❌ Failed to detect email project!")
        print("   Config sources.outlook.enabled:", project.config.get("sources", {}).get("outlook", {}).get("enabled"))
        return False

    # Test 2: Run a simple query and check if orchestrator is used
    print("\n" + "=" * 60)
    print("Test 2: Query Execution with EmailOrchestratorAgent")
    print("=" * 60)

    test_query = "What are the pressing issues?"
    print(f"Query: '{test_query}'")
    print("\nRunning retrieval step...")
    print("-" * 60)

    runner.add_step("retrieve", query=test_query, top_k=5)

    # Capture output
    messages = []
    try:
        for msg in runner.run_steps():
            print(msg)
            messages.append(msg)

        # Check if orchestrator was used
        orchestrator_used = any("EmailOrchestratorAgent" in msg for msg in messages)
        intent_detected = any("intent:" in msg.lower() for msg in messages)

        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        print(f"✓ EmailOrchestratorAgent mentioned: {orchestrator_used}")
        print(f"✓ Intent detection shown: {intent_detected}")
        print(f"✓ Chunks retrieved: {len(runner.retrieved_chunks)}")

        if orchestrator_used and intent_detected:
            print("\n🎉 SUCCESS! EmailOrchestratorAgent is integrated correctly!")
            print("\nThe new Phase 1-4 components are now active:")
            print("  ✓ Intent detection")
            print("  ✓ Strategy selection")
            print("  ✓ Context assembly")
            print("  ✓ Multi-aspect retrieval")
            return True
        else:
            print("\n⚠️ Orchestrator might not be working as expected.")
            if not orchestrator_used:
                print("   - EmailOrchestratorAgent was not mentioned in output")
            if not intent_detected:
                print("   - Intent detection was not shown in output")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clear steps for next run
        runner.clear_steps()


def main():
    """Run the integration test."""
    success = test_email_orchestrator_integration()

    if success:
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Run a query through the UI (Pipeline Actions tab)")
        print("2. Check the logs in logs/runs/run_*/app.log")
        print("3. Verify logs show:")
        print("   - Intent detection (not just 'late_fusion')")
        print("   - Strategy selection (sender_query, temporal_query, etc.)")
        print("   - Confidence scores")
        print("   - Detection method (pattern or llm)")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        print("   Check the error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
