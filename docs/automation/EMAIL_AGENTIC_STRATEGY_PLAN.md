# Email Agentic Strategy - Implementation Plan

**Date:** 2025-11-20
**Status:** 📋 Planning
**Branch:** `feature/email-agentic-strategy`
**Previous Milestone:** Email integration merged to main ✅

---

## Overview

Implement an **agentic orchestration system** for intelligent email handling that:
1. **Detects user intent** from queries
2. **Selects optimal retrieval strategy** based on intent
3. **Orchestrates email ingestion** with specialized agents
4. **Provides adaptive context** for better LLM responses

This transforms the RAG system from static retrieval to intelligent, context-aware email processing.

---

## Motivation

### Current Limitations

**Retrieval Strategy:**
- ❌ Fixed K=5 chunks (not enough for email threads)
- ❌ No thread awareness (fragmented conversations)
- ❌ No temporal ordering (emails out of chronological order)
- ❌ No sender-based filtering
- ❌ Treats all queries the same way

**Example Problem:**
```
Query: "Summarize the discussion about Primo NDE"
Current: Returns 5 random chunks from different threads
Needed: Complete thread(s) about Primo NDE, chronologically ordered
```

### Proposed Solution: Agentic Orchestration

**Intent-Driven Retrieval:**
- ✅ Query → Intent Detection → Strategy Selection
- ✅ Thread reconstruction for "discussion" queries
- ✅ Temporal filtering for "recent" queries
- ✅ Sender filtering for "what did X say" queries
- ✅ Adaptive K based on intent

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                  Email Orchestrator Agent               │
│                                                         │
│  ┌──────────────────┐      ┌──────────────────────┐   │
│  │  Intent Detector │──→───│  Strategy Selector   │   │
│  └──────────────────┘      └──────────────────────┘   │
│                                      │                  │
│                                      ↓                  │
│         ┌────────────────────────────────────────┐     │
│         │      Retrieval Strategy Router         │     │
│         └────────────────────────────────────────┘     │
│                          │                              │
│          ┌───────────────┼───────────────┐             │
│          ↓               ↓               ↓             │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│    │ Thread  │    │Temporal │    │ Sender  │          │
│    │Retriever│    │Retriever│    │Retriever│          │
│    └─────────┘    └─────────┘    └─────────┘          │
│          │               │               │             │
│          └───────────────┴───────────────┘             │
│                          ↓                              │
│              ┌──────────────────────┐                  │
│              │  Context Assembler   │                  │
│              └──────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
                  ┌─────────────┐
                  │     LLM     │
                  └─────────────┘
```

---

## Component Design

### 1. Intent Detector

**Purpose:** Classify user query into intent categories

**Intent Types:**

| Intent | Description | Query Examples | Optimal Strategy |
|--------|-------------|----------------|------------------|
| `thread_summary` | Summarize email thread/discussion | "Summarize discussion about X", "What was the conversation about Y?" | Thread Retrieval |
| `sender_query` | Find what specific person said | "What did Sarah say?", "John's opinion on X" | Sender Filtering |
| `temporal_query` | Find emails in time range | "Recent emails about X", "What happened last week?" | Temporal Filtering |
| `action_items` | Extract tasks/deadlines | "What are the action items?", "Any deadlines mentioned?" | Action Extraction |
| `decision_tracking` | Find decisions made | "What was decided?", "Final decision on X?" | Decision Pattern Matching |
| `factual_lookup` | Find specific information | "What is the budget for X?", "Who is responsible for Y?" | Standard Retrieval |

**Implementation:**

```python
class EmailIntentDetector:
    """Detects user intent from email queries."""

    def __init__(self):
        self.patterns = {
            "thread_summary": [
                r"summarize.*discussion",
                r"what.*conversation",
                r"thread about",
                r"exchange about"
            ],
            "sender_query": [
                r"what did (\w+) say",
                r"(\w+)'s (opinion|view|response)",
                r"emails from (\w+)",
                r"did (\w+) mention"
            ],
            "temporal_query": [
                r"recent",
                r"latest",
                r"last (week|month|day)",
                r"in the past",
                r"yesterday",
                r"this (week|month)"
            ],
            "action_items": [
                r"action items",
                r"tasks",
                r"deadlines",
                r"todo",
                r"need to (do|complete)"
            ],
            "decision_tracking": [
                r"what was decided",
                r"final decision",
                r"agreed (on|to)",
                r"conclusion"
            ]
        }

    def detect(self, query: str) -> dict:
        """
        Detect intent from query.

        Returns:
            {
                "primary_intent": "thread_summary",
                "confidence": 0.85,
                "metadata": {
                    "sender": "Sarah",  # if sender_query
                    "time_range": "last_week"  # if temporal_query
                }
            }
        """
        # Pattern matching + optional LLM classification
        # Return top intent with confidence score
        pass
```

**Alternative: LLM-Based Intent Detection**

For more sophisticated intent detection, use small LLM:

```python
def detect_intent_llm(self, query: str) -> dict:
    """Use LLM to classify intent."""
    prompt = f"""
    Classify the following email query into one of these intents:
    - thread_summary: User wants to summarize an email thread
    - sender_query: User wants emails from specific person
    - temporal_query: User wants recent/time-based emails
    - action_items: User wants tasks/deadlines
    - decision_tracking: User wants decisions made
    - factual_lookup: User wants specific information

    Query: "{query}"

    Return JSON: {{"intent": "...", "confidence": 0.0-1.0, "metadata": {{}}}}
    """
    # Call lightweight model (gpt-3.5-turbo or local model)
    # Parse JSON response
    return intent_data
```

---

### 2. Strategy Selector

**Purpose:** Choose optimal retrieval strategy based on detected intent

```python
class EmailStrategySelector:
    """Selects retrieval strategy based on intent."""

    STRATEGY_MAP = {
        "thread_summary": "thread_retrieval",
        "sender_query": "sender_filtered_retrieval",
        "temporal_query": "temporal_retrieval",
        "action_items": "action_extraction",
        "decision_tracking": "decision_extraction",
        "factual_lookup": "standard_retrieval"
    }

    def select_strategy(self, intent: dict) -> str:
        """
        Select strategy based on intent.

        Args:
            intent: Output from IntentDetector

        Returns:
            Strategy name (e.g., "thread_retrieval")
        """
        primary_intent = intent["primary_intent"]
        confidence = intent["confidence"]

        if confidence < 0.5:
            # Low confidence, use standard retrieval
            return "standard_retrieval"

        return self.STRATEGY_MAP.get(primary_intent, "standard_retrieval")
```

---

### 3. Retrieval Strategies

#### 3.1 Thread Retrieval

**Use Case:** Summarize email discussions, understand conversation flow

**Algorithm:**
1. Semantic search to find most relevant email(s)
2. Extract thread ID (normalized subject)
3. Retrieve ALL emails in thread
4. Sort chronologically
5. Return complete thread(s)

```python
class ThreadRetriever:
    """Retrieves complete email threads."""

    def retrieve(self, query: str, top_threads: int = 2) -> List[Chunk]:
        """
        Retrieve complete email threads.

        Args:
            query: User query
            top_threads: Number of threads to return

        Returns:
            List of chunks representing complete threads, chronologically sorted
        """
        # Stage 1: Find seed emails
        seed_emails = self.semantic_search(query, top_k=10)

        # Stage 2: Group by thread (normalized subject)
        threads = self.group_by_thread(seed_emails)

        # Stage 3: Score threads by relevance
        scored_threads = self.score_threads(threads, query)

        # Stage 4: Expand top threads to full conversation
        complete_threads = []
        for thread_id in scored_threads[:top_threads]:
            thread_emails = self.get_full_thread(thread_id)
            thread_emails.sort(key=lambda e: e.meta['date'])
            complete_threads.extend(thread_emails)

        return complete_threads

    def normalize_subject(self, subject: str) -> str:
        """
        Normalize email subject for thread grouping.

        Examples:
            "Budget Discussion" → "budget discussion"
            "Re: Budget Discussion" → "budget discussion"
            "Fwd: Re: Budget Discussion" → "budget discussion"
        """
        # Remove Re:, Fwd:, etc.
        # Lowercase and strip
        # Remove extra whitespace
        normalized = re.sub(r'^(re:|fwd:|fw:)\s*', '', subject, flags=re.IGNORECASE)
        return normalized.lower().strip()
```

**Benefits:**
- ✅ Complete conversation context
- ✅ Chronological ordering
- ✅ Natural thread boundaries
- ✅ Better LLM understanding

---

#### 3.2 Temporal Retrieval

**Use Case:** "Recent emails about X", "What happened last week?"

**Algorithm:**
1. Parse time range from query
2. Filter emails by date
3. Semantic search within filtered set
4. Sort chronologically (most recent first)

```python
class TemporalRetriever:
    """Retrieves emails filtered by time range."""

    def retrieve(self, query: str, intent_metadata: dict, top_k: int = 15) -> List[Chunk]:
        """
        Retrieve emails in specific time range.

        Args:
            query: User query
            intent_metadata: {"time_range": "last_week"} from intent detector
            top_k: Number of chunks to return

        Returns:
            Chronologically sorted chunks from time range
        """
        # Parse time range
        time_range = self.parse_time_range(intent_metadata.get("time_range"))

        # Filter emails by date
        filtered_emails = self.filter_by_date(
            start_date=time_range["start"],
            end_date=time_range["end"]
        )

        # Semantic search within filtered set
        results = self.semantic_search_filtered(query, filtered_emails, top_k=top_k)

        # Sort by date (most recent first)
        results.sort(key=lambda e: e.meta['date'], reverse=True)

        return results

    def parse_time_range(self, time_expr: str) -> dict:
        """
        Parse time expression to date range.

        Examples:
            "last_week" → {"start": "2025-11-13", "end": "2025-11-20"}
            "yesterday" → {"start": "2025-11-19", "end": "2025-11-19"}
            "this_month" → {"start": "2025-11-01", "end": "2025-11-30"}
        """
        # Use dateparser or custom logic
        pass
```

**Benefits:**
- ✅ Temporal precision
- ✅ "Recent" queries work naturally
- ✅ Time-aware context
- ✅ Filters noise from old emails

---

#### 3.3 Sender-Filtered Retrieval

**Use Case:** "What did Sarah say about X?"

**Algorithm:**
1. Extract sender name from query
2. Filter emails from that sender
3. Semantic search within sender's emails
4. Sort by relevance or date

```python
class SenderRetriever:
    """Retrieves emails from specific sender."""

    def retrieve(self, query: str, intent_metadata: dict, top_k: int = 10) -> List[Chunk]:
        """
        Retrieve emails from specific sender.

        Args:
            query: User query
            intent_metadata: {"sender": "Sarah"} from intent detector
            top_k: Number of chunks to return
        """
        # Extract sender name
        sender_name = intent_metadata.get("sender")

        # Find matching sender in email metadata
        sender_emails = self.filter_by_sender(sender_name)

        # Semantic search within sender's emails
        results = self.semantic_search_filtered(query, sender_emails, top_k=top_k)

        return results

    def filter_by_sender(self, sender_name: str) -> List[Chunk]:
        """
        Filter emails by sender name (fuzzy match).

        Examples:
            "Sarah" matches "Sarah Johnson", "Sarah J", "sarah.j@company.com"
        """
        # Fuzzy matching on sender_name and sender_email
        pass
```

**Benefits:**
- ✅ Person-specific queries
- ✅ Clearer attribution
- ✅ Reduces noise from other senders
- ✅ Better for "who said what" questions

---

#### 3.4 Adaptive Standard Retrieval

**Use Case:** Generic queries, factual lookup

**Algorithm:**
- Standard semantic search
- **But** with adaptive K based on content type:
  - Email content → K=15
  - Document content → K=5

```python
class AdaptiveRetriever:
    """Standard retrieval with adaptive K."""

    def retrieve(self, query: str, doc_type: str = "email") -> List[Chunk]:
        """Adaptive K based on document type."""
        if doc_type == "email":
            top_k = 15  # More context for emails
        else:
            top_k = 5   # Standard for documents

        return self.semantic_search(query, top_k=top_k)
```

---

### 4. Email Orchestrator Agent

**Purpose:** Main orchestration agent that coordinates the entire flow

```python
class EmailOrchestratorAgent:
    """
    Main orchestrator for email retrieval and processing.

    Responsibilities:
    - Detect query intent
    - Select retrieval strategy
    - Route to appropriate retriever
    - Assemble final context
    - Log decisions for debugging
    """

    def __init__(self, project: ProjectManager):
        self.project = project
        self.intent_detector = EmailIntentDetector()
        self.strategy_selector = EmailStrategySelector()

        # Initialize all retrievers
        self.retrievers = {
            "thread_retrieval": ThreadRetriever(project),
            "temporal_retrieval": TemporalRetriever(project),
            "sender_filtered_retrieval": SenderRetriever(project),
            "standard_retrieval": AdaptiveRetriever(project)
        }

    def retrieve(self, query: str) -> dict:
        """
        Main retrieval orchestration.

        Returns:
            {
                "chunks": [...],  # Retrieved chunks
                "intent": {...},  # Detected intent
                "strategy": "thread_retrieval",  # Used strategy
                "metadata": {...}  # Additional info
            }
        """
        # Step 1: Detect intent
        intent = self.intent_detector.detect(query)
        self.log_intent(intent)

        # Step 2: Select strategy
        strategy_name = self.strategy_selector.select_strategy(intent)
        self.log_strategy(strategy_name)

        # Step 3: Route to retriever
        retriever = self.retrievers[strategy_name]
        chunks = retriever.retrieve(
            query=query,
            intent_metadata=intent.get("metadata", {})
        )

        # Step 4: Post-process (deduplicate, rank, etc.)
        chunks = self.post_process(chunks, intent)

        # Return results with metadata
        return {
            "chunks": chunks,
            "intent": intent,
            "strategy": strategy_name,
            "metadata": {
                "chunk_count": len(chunks),
                "date_range": self.get_date_range(chunks),
                "senders": self.get_unique_senders(chunks)
            }
        }

    def log_intent(self, intent: dict):
        """Log detected intent for debugging."""
        logger.info(
            f"Detected intent: {intent['primary_intent']} "
            f"(confidence: {intent['confidence']:.2f})"
        )

    def log_strategy(self, strategy: str):
        """Log selected strategy."""
        logger.info(f"Selected strategy: {strategy}")
```

---

## Implementation Phases

### Phase 1: Intent Detection (Week 1)

**Goals:**
- ✅ Implement pattern-based intent detection
- ✅ Define intent taxonomy
- ✅ Test with sample queries

**Deliverables:**
- `scripts/agents/email_intent_detector.py`
- Unit tests for intent detection
- Sample query dataset with labeled intents

**Effort:** 4-6 hours

---

### Phase 2: Basic Retrieval Strategies (Week 1-2)

**Goals:**
- ✅ Implement Thread Retriever
- ✅ Implement Temporal Retriever
- ✅ Implement Adaptive K
- ✅ Test with real email data

**Deliverables:**
- `scripts/retrieval/email_thread_retriever.py`
- `scripts/retrieval/email_temporal_retriever.py`
- `scripts/retrieval/adaptive_retriever.py`
- Unit tests for each retriever

**Effort:** 8-10 hours

---

### Phase 3: Orchestrator Integration (Week 2)

**Goals:**
- ✅ Implement EmailOrchestratorAgent
- ✅ Integrate with existing RAG pipeline
- ✅ Add logging and debugging
- ✅ Test end-to-end with UI

**Deliverables:**
- `scripts/agents/email_orchestrator.py`
- Integration with `ask_interface.py`
- UI updates to show strategy selection
- End-to-end tests

**Effort:** 6-8 hours

---

### Phase 4: Advanced Features (Week 3+)

**Goals:**
- ✅ LLM-based intent detection (optional upgrade)
- ✅ Sender-filtered retrieval
- ✅ Action item extraction
- ✅ Decision tracking

**Deliverables:**
- Enhanced intent detector with LLM
- Additional specialized retrievers
- Performance benchmarks

**Effort:** 10-12 hours

---

## Testing Strategy

### Unit Tests

**Intent Detection:**
```python
def test_thread_summary_intent():
    detector = EmailIntentDetector()
    result = detector.detect("Summarize the discussion about Primo NDE")
    assert result["primary_intent"] == "thread_summary"
    assert result["confidence"] > 0.7

def test_sender_query_intent():
    detector = EmailIntentDetector()
    result = detector.detect("What did Sarah say about the budget?")
    assert result["primary_intent"] == "sender_query"
    assert result["metadata"]["sender"] == "Sarah"
```

**Thread Retrieval:**
```python
def test_thread_reconstruction():
    retriever = ThreadRetriever(project)
    chunks = retriever.retrieve("Budget discussion", top_threads=1)

    # Check thread completeness
    subjects = [c.meta["subject"] for c in chunks]
    normalized = [retriever.normalize_subject(s) for s in subjects]
    assert len(set(normalized)) == 1  # All same thread

    # Check chronological order
    dates = [c.meta["date"] for c in chunks]
    assert dates == sorted(dates)  # Chronological
```

### Integration Tests

**End-to-End Query:**
```python
def test_thread_summary_e2e():
    orchestrator = EmailOrchestratorAgent(project)
    result = orchestrator.retrieve("Summarize Primo NDE discussion")

    # Check intent detection
    assert result["intent"]["primary_intent"] == "thread_summary"

    # Check strategy selection
    assert result["strategy"] == "thread_retrieval"

    # Check results
    assert len(result["chunks"]) > 5  # Complete thread
    assert all(c.meta["doc_type"] == "outlook_eml" for c in result["chunks"])
```

### Real-World Validation

Test with actual email queries:
- "Summarize the Primo NDE discussion from last week"
- "What did Manuela say about facet behavior?"
- "Recent emails about CSS customization"
- "What action items were mentioned in the migration thread?"

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| **Retrieval Quality** |||
| Thread completeness | 20% (5/25 emails) | 90% (complete threads) |
| Temporal accuracy | N/A | 95% (correct time range) |
| Sender precision | N/A | 90% (correct sender) |
| **Performance** |||
| Intent detection time | N/A | <50ms |
| Retrieval time | ~500ms | <1000ms |
| Total query time | ~2s | <3s |
| **User Experience** |||
| Query success rate | 60% | 90% |
| Answer relevance | 70% | 90% |
| Context completeness | 40% | 85% |

---

## Architecture Decisions

### Why Agentic Approach?

**Alternatives Considered:**
1. **Static rules** - Too rigid, can't adapt to query variations
2. **Pure LLM** - Expensive, slow, less control
3. **Hybrid (chosen)** - Deterministic routing + optional LLM enhancement

**Benefits of Agentic Approach:**
- ✅ Modular (easy to add new strategies)
- ✅ Testable (each component isolated)
- ✅ Debuggable (explicit decision trail)
- ✅ Extensible (can add LLM enhancement later)
- ✅ Cost-effective (pattern matching for most cases)

### Why Intent Detection?

**Without Intent Detection:**
```
Query: "Summarize Primo discussion"
→ Returns 5 random chunks (bad)
```

**With Intent Detection:**
```
Query: "Summarize Primo discussion"
→ Intent: thread_summary
→ Strategy: thread_retrieval
→ Returns complete thread chronologically (good)
```

### Why Multiple Retrieval Strategies?

Different query types have different optimal retrieval patterns:
- **Thread summary** → Need complete conversation
- **Recent emails** → Need temporal filtering
- **What did X say** → Need sender filtering
- **Factual lookup** → Standard semantic search

One-size-fits-all doesn't work for emails!

---

## File Structure

```
scripts/
├── agents/
│   ├── __init__.py
│   ├── email_intent_detector.py      # Intent detection
│   ├── email_orchestrator.py          # Main orchestrator
│   └── README.md                       # Agent documentation
├── retrieval/
│   ├── email_thread_retriever.py      # Thread reconstruction
│   ├── email_temporal_retriever.py    # Time-based filtering
│   ├── email_sender_retriever.py      # Sender-based filtering
│   └── adaptive_retriever.py          # Adaptive K retrieval
└── interface/
    └── ask_interface.py                # Updated with orchestrator

tests/
├── agents/
│   ├── test_intent_detector.py
│   └── test_orchestrator.py
└── retrieval/
    ├── test_thread_retriever.py
    ├── test_temporal_retriever.py
    └── test_sender_retriever.py

docs/
└── EMAIL_AGENTIC_STRATEGY_PLAN.md     # This document
```

---

## Next Steps

### Immediate (This Session)
1. ✅ Merge email-integration to main
2. ✅ Create feature/email-agentic-strategy branch
3. ✅ Create implementation plan (this document)

### Phase 1 (Next Session)
4. Implement EmailIntentDetector
5. Define intent taxonomy and patterns
6. Create test dataset with sample queries
7. Unit tests for intent detection

### Phase 2
8. Implement ThreadRetriever
9. Implement TemporalRetriever
10. Implement AdaptiveRetriever
11. Test with real email data

### Phase 3
12. Implement EmailOrchestratorAgent
13. Integrate with ask_interface.py
14. UI updates to show strategy
15. End-to-end testing

---

## Open Questions

1. **Intent Detection Method:**
   - Start with pattern matching or use LLM from the start?
   - **Recommendation:** Start with patterns, add LLM as Phase 4 enhancement

2. **Thread Grouping:**
   - Use subject normalization or message-ID threading?
   - **Recommendation:** Subject normalization (simpler, works for most cases)

3. **Retrieval Ranking:**
   - How to score threads for relevance?
   - **Recommendation:** Combine semantic score of best email + thread size

4. **UI Integration:**
   - Show strategy selection to user?
   - **Recommendation:** Yes, for transparency and debugging

5. **Fallback Strategy:**
   - What if intent detection has low confidence?
   - **Recommendation:** Fall back to adaptive standard retrieval

---

## Related Documentation

- **Proposal:** docs/EMAIL_PROMPTING_PROPOSAL.md (retrieval strategies proposed)
- **Implementation:** docs/EMAIL_PROMPTING_IMPLEMENTATION.md (current prompting)
- **Outlook Integration:** docs/PHASE5_COMPLETION_SUMMARY.md (email extraction)

---

## Conclusion

This agentic approach transforms the RAG system from **static retrieval** to **intelligent orchestration**, enabling:

✅ **Intent-driven retrieval** - Different strategies for different query types
✅ **Thread awareness** - Complete conversations instead of fragments
✅ **Temporal intelligence** - "Recent" queries work naturally
✅ **Sender attribution** - "What did X say" queries work perfectly
✅ **Modular architecture** - Easy to extend and test
✅ **Production-ready** - Built on solid foundation from Phase 1-5

The orchestrator agent acts as the **intelligent router** that understands what the user wants and selects the best retrieval strategy to satisfy that need.

**Ready to begin Phase 1!** 🚀

---

**Last Updated:** 2025-11-20
**Status:** Planning Complete - Ready for Implementation
**Branch:** feature/email-agentic-strategy
