#!/usr/bin/env python3
"""
Demo: Email Agentic Strategy in Action

This script demonstrates how to use the complete email RAG pipeline
with all Phase 1-4 features enabled.

Prerequisites:
1. Emails must be ingested into the system (see README for ingestion steps)
2. OPEN_AI environment variable must be set (for LLM features)
3. Project must be initialized with email data

Usage:
    python demo_email_rag.py
"""

import os
import sys
from scripts.project_manager.project_manager import ProjectManager
from scripts.agents.email_orchestrator import EmailOrchestratorAgent
from scripts.agents.action_item_extractor import ActionItemExtractor
from scripts.agents.decision_extractor import DecisionExtractor
from scripts.agents.answer_validator import AnswerValidator


def print_separator(title=""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"{'='*60}\n")


def demonstrate_basic_query(orchestrator, query):
    """Demonstrate a basic email query."""
    print_separator(f"Query: {query}")

    result = orchestrator.retrieve(query, top_k=10, max_tokens=2000)

    # Show intent detection
    print("📊 INTENT DETECTION:")
    print(f"  Primary Intent: {result['intent']['primary_intent']}")
    print(f"  Confidence: {result['intent']['confidence']:.2f}")
    print(f"  Detection Method: {result['intent'].get('detection_method', 'pattern')}")
    if result['intent']['metadata']:
        print(f"  Metadata: {result['intent']['metadata']}")
    if result['intent']['secondary_signals']:
        print(f"  Secondary Signals: {result['intent']['secondary_signals']}")

    # Show strategy selection
    print("\n🎯 STRATEGY SELECTION:")
    print(f"  Primary Strategy: {result['strategy']['primary']}")
    if result['strategy']['params']:
        print(f"  Parameters: {result['strategy']['params']}")

    # Show retrieval results
    print("\n📧 RETRIEVAL RESULTS:")
    print(f"  Chunks Retrieved: {result['metadata']['chunk_count']}")
    if result['metadata'].get('date_range'):
        print(f"  Date Range: {result['metadata']['date_range']['start']} to {result['metadata']['date_range']['end']}")
    if result['metadata'].get('unique_senders'):
        print(f"  Senders: {', '.join(result['metadata']['unique_senders'][:3])}")
        if len(result['metadata']['unique_senders']) > 3:
            print(f"           ... and {len(result['metadata']['unique_senders']) - 3} more")

    # Show context preview
    print("\n📝 CONTEXT (Preview):")
    context_preview = result['context'][:500]
    print(f"  {context_preview}")
    if len(result['context']) > 500:
        print(f"  ... ({len(result['context']) - 500} more characters)")

    return result


def demonstrate_action_items(orchestrator, query):
    """Demonstrate action item extraction."""
    print_separator(f"Action Items: {query}")

    # First retrieve relevant emails
    result = orchestrator.retrieve(query, top_k=15, max_tokens=3000)

    print(f"📧 Retrieved {len(result['chunks'])} emails\n")

    # Extract action items with LLM
    extractor = ActionItemExtractor(use_llm=True)

    print("🔍 Extracting action items with GPT-4o...")
    try:
        action_items = extractor.extract(result['chunks'])

        if action_items:
            print(f"\n✅ FOUND {len(action_items)} ACTION ITEMS:\n")
            for i, item in enumerate(action_items, 1):
                print(f"{i}. {item['task']}")
                if item.get('deadline'):
                    print(f"   ⏰ Deadline: {item['deadline']}")
                if item.get('assigned_to'):
                    print(f"   👤 Assigned to: {item['assigned_to']}")
                print(f"   📎 Source: {item.get('source', item.get('source_email', 'Unknown'))}")
                print()
        else:
            print("ℹ️  No action items found in the retrieved emails.")
    except Exception as e:
        print(f"⚠️  LLM extraction failed: {e}")
        print("💡 Falling back to pattern-based extraction...")
        extractor_fallback = ActionItemExtractor(use_llm=False)
        action_items = extractor_fallback.extract(result['chunks'])
        if action_items:
            print(f"\n✅ FOUND {len(action_items)} ACTION ITEMS (pattern-based):\n")
            for i, item in enumerate(action_items, 1):
                print(f"{i}. {item['task']}")


def demonstrate_decisions(orchestrator, query):
    """Demonstrate decision extraction."""
    print_separator(f"Decisions: {query}")

    # First retrieve relevant emails
    result = orchestrator.retrieve(query, top_k=15, max_tokens=3000)

    print(f"📧 Retrieved {len(result['chunks'])} emails\n")

    # Extract decisions with LLM
    extractor = DecisionExtractor(use_llm=True)

    print("🔍 Extracting decisions with GPT-4o...")
    try:
        decisions = extractor.extract(result['chunks'])

        if decisions:
            print(f"\n✅ FOUND {len(decisions)} DECISIONS:\n")
            for i, dec in enumerate(decisions, 1):
                print(f"{i}. {dec['decision']}")
                if dec.get('made_by'):
                    print(f"   👤 Made by: {dec['made_by']}")
                if dec.get('date'):
                    print(f"   📅 Date: {dec['date']}")
                print(f"   📎 Source: {dec.get('source', dec.get('source_email', 'Unknown'))}")
                print()
        else:
            print("ℹ️  No decisions found in the retrieved emails.")
    except Exception as e:
        print(f"⚠️  LLM extraction failed: {e}")


def demonstrate_answer_validation(orchestrator, query, expected_answer):
    """Demonstrate answer validation."""
    print_separator(f"Answer Validation: {query}")

    # Retrieve context
    result = orchestrator.retrieve(query, top_k=10, max_tokens=2000)

    print(f"📧 Retrieved {len(result['chunks'])} emails")
    print(f"\n📝 Answer to validate:\n  \"{expected_answer}\"\n")

    # Validate the answer
    validator = AnswerValidator(use_llm=True)

    print("🔍 Validating answer...")
    try:
        validation = validator.validate(
            answer=expected_answer,
            context=result['context'],
            intent=result['intent']
        )

        print(f"\n{'✅ VALID' if validation['is_valid'] else '❌ INVALID'}")
        print(f"Confidence: {validation['confidence']:.2f}\n")

        if validation['issues']:
            print("⚠️  ISSUES FOUND:")
            for issue in validation['issues']:
                print(f"  • {issue}")
            print()

        if validation['suggestions']:
            print("💡 SUGGESTIONS:")
            for suggestion in validation['suggestions']:
                print(f"  • {suggestion}")
            print()

        if validation['is_valid']:
            print("✓ Answer passes validation checks")

    except Exception as e:
        print(f"⚠️  Validation failed: {e}")


def main():
    """Main demonstration function."""
    print_separator("EMAIL AGENTIC STRATEGY DEMO")

    # Check for OpenAI API key
    if not os.getenv("OPEN_AI"):
        print("⚠️  WARNING: OPEN_AI environment variable not set!")
        print("   LLM-powered features (Phase 4) will be limited.\n")

    # Initialize project
    print("🔧 Initializing project...")
    try:
        # Try to load an existing project
        # You'll need to replace 'your_project_name' with actual project name
        project = ProjectManager.load_project("your_project_name")
        print(f"✅ Loaded project: {project.name}\n")
    except Exception as e:
        print(f"❌ Failed to load project: {e}")
        print("\n💡 SETUP REQUIRED:")
        print("   1. Ingest email data using the ingestion pipeline")
        print("   2. Create a project with email documents")
        print("   3. Update 'your_project_name' in this script")
        print("\n   See README for detailed setup instructions.\n")
        return

    # Initialize orchestrator with Phase 4 features enabled
    print("🚀 Initializing Email Orchestrator with Phase 4 features...")
    orchestrator = EmailOrchestratorAgent(project)

    # Enable LLM fallback for intent detection
    from scripts.agents.email_intent_detector import EmailIntentDetector
    orchestrator.intent_detector = EmailIntentDetector(
        use_llm_fallback=True,
        llm_confidence_threshold=0.6
    )
    print("✅ Orchestrator ready!\n")

    # Demo 1: Basic sender query
    demonstrate_basic_query(
        orchestrator,
        "What did Alice say about the budget?"
    )

    input("\nPress Enter to continue to next demo...")

    # Demo 2: Temporal query
    demonstrate_basic_query(
        orchestrator,
        "Show me emails from last week about migration"
    )

    input("\nPress Enter to continue to next demo...")

    # Demo 3: Thread summary
    demonstrate_basic_query(
        orchestrator,
        "Summarize the discussion about the vendor selection"
    )

    input("\nPress Enter to continue to next demo...")

    # Demo 4: Action items extraction
    demonstrate_action_items(
        orchestrator,
        "What are the action items from the project emails?"
    )

    input("\nPress Enter to continue to next demo...")

    # Demo 5: Decision extraction
    demonstrate_decisions(
        orchestrator,
        "What was decided about the budget approval?"
    )

    input("\nPress Enter to continue to next demo...")

    # Demo 6: Answer validation
    demonstrate_answer_validation(
        orchestrator,
        "What did Alice say about the budget?",
        "Alice mentioned the budget needs to increase by 20% for Q4."
    )

    print_separator("DEMO COMPLETE")
    print("✅ You've seen all the Phase 1-4 features in action!")
    print("\n📚 For more information:")
    print("   • Phase 4 Docs: docs/archive/EMAIL_PHASE4_COMPLETION.md")
    print("   • Full Strategy: docs/automation/EMAIL_AGENTIC_STRATEGY_MERGED.md")
    print("   • UI Redesign Plan: docs/future/UI_REDESIGN_PLAN.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
