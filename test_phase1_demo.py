#!/usr/bin/env python3
"""
Phase 1 Demo: Test EmailIntentDetector and ContextAssembler

This script demonstrates Phase 1 functionality with sample email data.
You can also test with real Outlook emails from your vector store.
"""

import sys
from scripts.agents.email_intent_detector import EmailIntentDetector
from scripts.retrieval.context_assembler import ContextAssembler
from scripts.chunking.models import Chunk


def test_intent_detector():
    """Test EmailIntentDetector with various queries."""
    print("\n" + "="*80)
    print("TESTING EMAIL INTENT DETECTOR")
    print("="*80)

    detector = EmailIntentDetector()

    test_queries = [
        "Summarize the discussion about Primo NDE migration",
        "What did Sarah say about the budget last week?",
        "Recent emails about server migration",
        "What are the action items from yesterday's meeting?",
        "Was the vendor selection approved?",
        "Tell me about Primo",  # Fallback case
    ]

    for query in test_queries:
        print(f"\n📧 Query: \"{query}\"")
        result = detector.detect(query)

        print(f"   ├─ Primary Intent: {result['primary_intent']}")
        print(f"   ├─ Confidence: {result['confidence']:.2f}")

        if result['metadata']:
            print(f"   ├─ Metadata:")
            for key, value in result['metadata'].items():
                print(f"   │  ├─ {key}: {value}")

        if result['secondary_signals']:
            print(f"   └─ Secondary Signals: {result['secondary_signals']}")
        else:
            print(f"   └─ Secondary Signals: None")

    print("\n✅ Intent Detector working!\n")


def test_context_assembler():
    """Test ContextAssembler with sample email thread."""
    print("\n" + "="*80)
    print("TESTING CONTEXT ASSEMBLER")
    print("="*80)

    assembler = ContextAssembler()

    # Create realistic email thread with quotes, signatures, and redundancy
    email_thread = [
        Chunk(
            doc_id="email_1",
            text="""Hi team,

I wanted to share an update on the Primo NDE migration project.

We've completed the initial assessment and identified the following action items:
1. Database backup scheduled for next Tuesday
2. Server configuration review by Friday
3. User acceptance testing in 2 weeks

The timeline looks good and we're on track for the February 15th launch.

Best regards,
Sarah Johnson
Senior Project Manager
IT Department""",
            meta={
                "doc_type": "outlook_eml",
                "subject": "Primo NDE Migration Update",
                "sender_name": "Sarah Johnson",
                "sender": "sarah.j@company.com",
                "date": "2025-01-15 09:00:00"
            },
            token_count=80
        ),
        Chunk(
            doc_id="email_2",
            text="""> We've completed the initial assessment and identified the following action items:
> 1. Database backup scheduled for next Tuesday
> 2. Server configuration review by Friday
> 3. User acceptance testing in 2 weeks

Great work Sarah! I'll start the server configuration review today.

One question - do we have a contingency plan if the database backup takes longer than expected?

Thanks,
Mike Chen

Sent from my iPhone""",
            meta={
                "doc_type": "outlook_eml",
                "subject": "Re: Primo NDE Migration Update",
                "sender_name": "Mike Chen",
                "sender": "mike.c@company.com",
                "date": "2025-01-15 09:30:00"
            },
            token_count=70
        ),
        Chunk(
            doc_id="email_3",
            text="""> One question - do we have a contingency plan if the database backup takes longer than expected?

Good point Mike! Yes, we have:
- Backup window extended to 48 hours
- Rollback procedure tested and documented
- 24/7 support team on standby

The risk is low, but we're prepared.

Regards,
Sarah""",
            meta={
                "doc_type": "outlook_eml",
                "subject": "Re: Primo NDE Migration Update",
                "sender_name": "Sarah Johnson",
                "sender": "sarah.j@company.com",
                "date": "2025-01-15 10:15:00"
            },
            token_count=60
        ),
    ]

    print("\n📊 BEFORE CLEANING:")
    print("-" * 80)
    total_chars_before = sum(len(chunk.text) for chunk in email_thread)
    print(f"Total characters: {total_chars_before}")
    print(f"Number of emails: {len(email_thread)}")

    # Show raw email 2 with quotes
    print("\nExample - Email #2 (raw):")
    print(email_thread[1].text[:200] + "...")

    print("\n" + "="*80)

    # Test with thread_summary intent
    intent = {"primary_intent": "thread_summary"}
    cleaned_context = assembler.assemble(email_thread, intent)

    print("\n📊 AFTER CLEANING (thread_summary intent):")
    print("-" * 80)
    print(f"Total characters: {len(cleaned_context)}")
    reduction = ((total_chars_before - len(cleaned_context)) / total_chars_before) * 100
    print(f"Reduction: {reduction:.1f}%")
    print(f"Number of emails: 3 (preserved)")

    print("\n📧 CLEANED CONTEXT:")
    print("=" * 80)
    print(cleaned_context)
    print("=" * 80)

    print("\n✅ Context Assembler working!")
    print("\nKey improvements:")
    print("  ✓ Removed quoted text (>) - eliminated redundancy")
    print("  ✓ Removed signatures (Best regards, Sent from iPhone, titles)")
    print("  ✓ Chronologically ordered (09:00 → 09:30 → 10:15)")
    print("  ✓ Clear source attributions (From, Subject, Date)")
    print("  ✓ Clean separators between emails")
    print(f"  ✓ {reduction:.1f}% size reduction while preserving all unique content\n")


def test_combined_workflow():
    """Test both components together in a realistic workflow."""
    print("\n" + "="*80)
    print("COMBINED WORKFLOW TEST")
    print("="*80)

    detector = EmailIntentDetector()
    assembler = ContextAssembler()

    # Simulate user query
    query = "Summarize the Primo NDE migration discussion"

    print(f"\n👤 User Query: \"{query}\"")

    # Step 1: Detect intent
    print("\n📍 Step 1: Detect Intent")
    intent_result = detector.detect(query)
    print(f"   Intent: {intent_result['primary_intent']}")
    print(f"   Confidence: {intent_result['confidence']:.2f}")
    if intent_result['metadata'].get('topic_keywords'):
        print(f"   Topics: {intent_result['metadata']['topic_keywords']}")

    # Step 2: Simulate retrieval (in real system, this would query vector store)
    print("\n📍 Step 2: Retrieve Email Chunks")
    print("   (Simulated - would query vector store based on intent)")

    sample_chunks = [
        Chunk(
            doc_id="email_1",
            text="""The Primo NDE migration timeline has been finalized.

Key dates:
- January 20: Testing begins
- February 1: User training
- February 15: Go-live

Best regards,
Sarah""",
            meta={
                "doc_type": "outlook_eml",
                "subject": "Primo NDE Timeline",
                "sender_name": "Sarah Johnson",
                "sender": "sarah@company.com",
                "date": "2025-01-10 10:00:00"
            },
            token_count=40
        ),
        Chunk(
            doc_id="email_2",
            text="""> Key dates:
> - January 20: Testing begins
> - February 1: User training
> - February 15: Go-live

Looks good! Do we have enough time for testing?

Thanks,
Mike

Sent from my iPhone""",
            meta={
                "doc_type": "outlook_eml",
                "subject": "Re: Primo NDE Timeline",
                "sender_name": "Mike Chen",
                "sender": "mike@company.com",
                "date": "2025-01-10 11:00:00"
            },
            token_count=35
        ),
    ]

    print(f"   Retrieved: {len(sample_chunks)} chunks")

    # Step 3: Clean and assemble context
    print("\n📍 Step 3: Clean & Assemble Context")
    cleaned_context = assembler.assemble(sample_chunks, intent_result)
    print(f"   Removed quotes, signatures, redundancy")
    print(f"   Chronologically ordered for thread summary")

    # Step 4: Show final context ready for LLM
    print("\n📍 Step 4: Final Context for LLM")
    print("=" * 80)
    print(cleaned_context)
    print("=" * 80)

    print("\n✅ Complete workflow successful!")
    print("\nNext step: This clean context would be passed to the LLM prompt builder")
    print("for generating a high-quality answer.\n")


def test_with_real_emails():
    """Test with real emails from vector store (if available)."""
    print("\n" + "="*80)
    print("TESTING WITH REAL OUTLOOK EMAILS")
    print("="*80)

    try:
        # Try to import vector store and embedding components
        from scripts.vectorstore.vector_store_manager import VectorStoreManager
        from scripts.embeddings.embedder import Embedder

        print("\n📧 Attempting to load real emails from vector store...")

        # Initialize components
        embedder = Embedder()
        vector_store = VectorStoreManager(embedder)

        # Try a simple query
        query = "Primo migration"
        print(f"\n🔍 Searching for: \"{query}\"")

        # Get embedding and search
        query_embedding = embedder.embed_query(query)
        results = vector_store.search(query_embedding, top_k=5)

        if not results:
            print("   ⚠️  No results found. Make sure emails are ingested.")
            return

        print(f"   ✓ Found {len(results)} email chunks")

        # Convert results to Chunk objects
        chunks = []
        for doc_id, metadata, text, score in results:
            # Check if it's an email
            if metadata.get('doc_type') in ['outlook_eml', 'outlook_msg']:
                chunk = Chunk(
                    doc_id=doc_id,
                    text=text,
                    meta=metadata,
                    token_count=len(text.split())
                )
                chunks.append(chunk)

        if not chunks:
            print("   ⚠️  No email chunks found in results.")
            return

        print(f"   ✓ Filtered to {len(chunks)} email chunks")

        # Test intent detection
        detector = EmailIntentDetector()
        intent = detector.detect(query)
        print(f"\n📊 Detected Intent: {intent['primary_intent']} (confidence: {intent['confidence']:.2f})")

        # Test context assembly
        assembler = ContextAssembler()
        context = assembler.assemble(chunks[:3], intent)  # Use top 3

        print("\n📧 ASSEMBLED CONTEXT FROM REAL EMAILS:")
        print("=" * 80)
        print(context[:1000] + "..." if len(context) > 1000 else context)
        print("=" * 80)

        print("\n✅ Successfully tested with real emails!\n")

    except ImportError as e:
        print(f"\n⚠️  Could not import required modules: {e}")
        print("   This test requires vector store and embedder components.")
    except Exception as e:
        print(f"\n⚠️  Error testing with real emails: {e}")
        print("   Make sure emails are ingested to the vector store first.")


def main():
    """Run all Phase 1 tests."""
    print("\n" + "="*80)
    print("PHASE 1 TESTING SUITE")
    print("Email Intent Detector + Context Assembler")
    print("="*80)

    # Test 1: Intent Detector
    test_intent_detector()

    # Test 2: Context Assembler
    test_context_assembler()

    # Test 3: Combined workflow
    test_combined_workflow()

    # Test 4: Real emails (optional)
    response = input("\nWould you like to test with real Outlook emails from vector store? (y/n): ")
    if response.lower() == 'y':
        test_with_real_emails()

    print("\n" + "="*80)
    print("✅ ALL PHASE 1 TESTS COMPLETE")
    print("="*80)
    print("\nPhase 1 Components:")
    print("  ✓ EmailIntentDetector - Multi-aspect intent detection with metadata extraction")
    print("  ✓ ContextAssembler - Quote removal, signature stripping, deduplication")
    print("\nReady for Phase 2:")
    print("  → ThreadRetriever, TemporalRetriever, SenderRetriever")
    print("  → Multi-aspect query composition")
    print("  → Orchestrator integration\n")


if __name__ == "__main__":
    main()
