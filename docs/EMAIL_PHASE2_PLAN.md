# Phase 2: Intent Detection & Specialized Retrievers - Implementation Plan

**Status:** 🚧 In Progress
**Start Date:** 2025-11-22
**Dependencies:** Phase 1 Complete ✅

---

## Overview

Phase 2 transforms email retrieval from generic semantic search into **intent-aware, specialized retrieval** that understands what users are asking for and routes queries to the most appropriate retrieval strategy.

### The Problem
Current system (Phase 1) uses one-size-fits-all semantic search:
- "emails from John" → Generic semantic search (not optimized for sender)
- "recent discussions about FOLIO" → Generic search (ignores time signal)
- "summarize the migration thread" → Returns chunks, not thread structure

### The Solution
Phase 2 provides:
1. **Intent Classification** - Understand what user wants
2. **Specialized Retrievers** - Different strategies per intent
3. **Dynamic Top-K** - Adjust result count based on intent
4. **Orchestration** - Route queries intelligently

---

## Architecture

```
User Query
    ↓
Intent Classifier
    ↓
    ├─ sender_query → SenderRetriever (filter by sender) → 12 chunks
    ├─ temporal_query → TemporalRetriever (time-based + recency) → 15 chunks
    ├─ thread_summary → ThreadRetriever (group by thread) → 20 chunks
    ├─ factual_lookup → MultiAspectRetriever (semantic only) → 5 chunks
    └─ multi_aspect → MultiAspectRetriever (all signals) → 10 chunks
    ↓
EmailOrchestratorAgent
    ↓
Retrieved Chunks (ranked by relevance)
```

---

## Component 1: Intent Classifier

**Purpose:** Classify user query into one of 5 intent types

**Location:** `scripts/email/agents/intent_classifier.py`

### Intent Types

#### 1. `sender_query`
**Pattern:** User wants emails from/to specific person(s)

**Examples:**
- "emails from John Smith"
- "what did Sarah say about configuration?"
- "messages from john.doe@university.edu"
- "replies from the project manager"

**Signals:**
- Keywords: "from", "by", "sent by", "wrote", "said"
- Person names (proper nouns)
- Email addresses
- Sender-related verbs

---

#### 2. `temporal_query`
**Pattern:** User wants emails from specific time period

**Examples:**
- "emails from last week"
- "recent discussions about migration"
- "what happened in November?"
- "latest updates on the project"

**Signals:**
- Time expressions: "last week", "yesterday", "recent", "latest"
- Date references: "November", "Q3", "2024"
- Temporal keywords: "before", "after", "during", "since"
- Recency indicators: "new", "latest", "recent"

---

#### 3. `thread_summary`
**Pattern:** User wants to understand an entire discussion thread

**Examples:**
- "summarize the migration discussion"
- "what's the thread about facet customization?"
- "recap of the FOLIO conversation"
- "overview of the NDE implementation discussion"

**Signals:**
- Keywords: "summarize", "summary", "thread", "discussion", "conversation"
- Aggregation words: "overview", "recap", "what happened"
- Thread-related: "entire", "whole", "complete"

---

#### 4. `factual_lookup`
**Pattern:** User wants specific answer to a "how-to" or technical question

**Examples:**
- "how do I hide facet values in Primo?"
- "what is the CSS syntax for NDE?"
- "where is the configuration file?"
- "who is the contact for FOLIO migration?"

**Signals:**
- Question words: "how", "what", "where", "who", "when", "why"
- Technical nouns: "syntax", "configuration", "command", "setting"
- Instructional: "steps", "guide", "tutorial", "how-to"
- Definite article: "the" (seeking specific answer)

---

#### 5. `multi_aspect` (Default)
**Pattern:** Complex query requiring multiple retrieval signals

**Examples:**
- "recent emails from John about facet customization"
- "what did the team discuss last week about migration?"
- "Sarah's latest messages about CSS issues"

**Signals:**
- Combination of 2+ intent types
- Complex query structure
- Fallback when no clear single intent

---

### Implementation Strategy

**Two-Stage Approach:**

#### Stage 1: Pattern-Based (Fast, Deterministic)
```python
class PatternBasedIntentClassifier:
    """
    Rule-based classifier using regex and keyword matching.
    Fast, deterministic, no API calls.
    """

    PATTERNS = {
        'sender_query': [
            r'\b(from|by|sent by|wrote)\s+[A-Z][a-z]+',
            r'\b(emails?|messages?)\s+(from|by)\b',
            r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',  # Email address
        ],
        'temporal_query': [
            r'\b(last|past|previous)\s+(week|month|year|day)',
            r'\b(recent|latest|new)\b',
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(January|February|...|December)\b',
        ],
        'thread_summary': [
            r'\b(summarize|summary|recap|overview)\b',
            r'\b(thread|discussion|conversation)\s+(about|on)\b',
            r'\b(entire|whole|complete)\s+(thread|discussion)\b',
        ],
        'factual_lookup': [
            r'^\s*(how|what|where|who|when|why)\b',
            r'\b(configure|setup|install|customize)\b',
            r'\b(syntax|command|setting|parameter)\b',
        ],
    }

    def classify(self, query: str) -> IntentResult:
        scores = {intent: 0 for intent in self.PATTERNS}

        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    scores[intent] += 1

        # Return intent with highest score, or 'multi_aspect' if tie
        max_score = max(scores.values())
        if max_score == 0:
            return IntentResult('multi_aspect', confidence=0.3)

        winners = [k for k, v in scores.items() if v == max_score]
        if len(winners) > 1:
            return IntentResult('multi_aspect', confidence=0.7)

        return IntentResult(winners[0], confidence=0.8)
```

#### Stage 2: LLM-Based (Fallback, High Accuracy)
```python
class LLMIntentClassifier:
    """
    LLM-based classifier for ambiguous queries.
    Slower, costs API tokens, but higher accuracy.
    """

    PROMPT_TEMPLATE = """
    Classify this email search query into ONE intent category:

    Query: "{query}"

    Intent Categories:
    1. sender_query - User wants emails from/by specific person(s)
    2. temporal_query - User wants emails from specific time period
    3. thread_summary - User wants to understand entire discussion thread
    4. factual_lookup - User wants specific answer to technical question
    5. multi_aspect - Complex query requiring multiple signals

    Return ONLY the intent name (e.g., "sender_query") and confidence (0-1).

    Examples:
    - "emails from John" → sender_query (0.95)
    - "recent discussions" → temporal_query (0.90)
    - "how do I configure facets?" → factual_lookup (0.95)
    - "John's recent messages about CSS" → multi_aspect (0.85)
    """

    def classify(self, query: str) -> IntentResult:
        # Call LLM with prompt
        response = self.llm_client.complete(
            prompt=self.PROMPT_TEMPLATE.format(query=query),
            temperature=0.1,  # Low temp for deterministic classification
            max_tokens=50
        )

        # Parse response: "sender_query (0.95)"
        intent, confidence = self._parse_response(response)
        return IntentResult(intent, confidence=confidence)
```

#### Combined Classifier
```python
class IntentClassifier:
    """Combines pattern-based and LLM-based classifiers."""

    def __init__(self, use_llm_fallback: bool = True):
        self.pattern_classifier = PatternBasedIntentClassifier()
        self.llm_classifier = LLMIntentClassifier() if use_llm_fallback else None

    def classify(self, query: str) -> IntentResult:
        # Try pattern-based first
        result = self.pattern_classifier.classify(query)

        # If low confidence and LLM available, use LLM
        if result.confidence < 0.6 and self.llm_classifier:
            result = self.llm_classifier.classify(query)

        return result
```

---

## Component 2: Specialized Retrievers

### Base Retriever Interface
```python
from abc import ABC, abstractmethod
from typing import List
from scripts.chunking.models import Chunk

class BaseEmailRetriever(ABC):
    """Base class for all email retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict = None
    ) -> List[Chunk]:
        """Retrieve relevant chunks for query."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return retriever name for logging."""
        pass
```

---

### 2.1 SenderRetriever

**Purpose:** Filter/boost results by sender email or name

**Strategy:**
1. Parse query for sender information (name or email)
2. Search FAISS for semantic relevance
3. **Re-rank** results: boost chunks from target sender(s)
4. Return top-K after re-ranking

**Implementation:**
```python
class SenderRetriever(BaseEmailRetriever):
    """Retrieves emails from specific sender(s)."""

    def __init__(self, faiss_manager, metadata_manager):
        self.faiss = faiss_manager
        self.metadata = metadata_manager
        self.sender_extractor = SenderExtractor()  # Parses sender from query

    def retrieve(self, query: str, top_k: int = 12, filters: dict = None) -> List[Chunk]:
        # Extract sender info from query
        senders = self.sender_extractor.extract(query)  # ['John Smith', 'john@...']

        # Get initial semantic results (top_k * 3 for re-ranking)
        semantic_results = self.faiss.search(query, top_k=top_k * 3)

        # Re-rank: boost chunks from target senders
        ranked_results = []
        for chunk in semantic_results:
            score = chunk.meta['similarity']

            # Boost if sender matches
            sender_name = chunk.meta.get('sender_name', '').lower()
            sender_email = chunk.meta.get('sender', '').lower()

            for target_sender in senders:
                if (target_sender.lower() in sender_name or
                    target_sender.lower() in sender_email):
                    score += 0.3  # Boost sender matches
                    break

            chunk.meta['final_score'] = score
            ranked_results.append(chunk)

        # Sort by final score and return top-K
        ranked_results.sort(key=lambda c: c.meta['final_score'], reverse=True)
        return ranked_results[:top_k]

    def get_name(self) -> str:
        return "SenderRetriever"
```

**Sender Extraction:**
```python
class SenderExtractor:
    """Extracts sender names/emails from query."""

    EMAIL_PATTERN = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    NAME_PATTERN = r'\b(from|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'

    def extract(self, query: str) -> List[str]:
        senders = []

        # Extract email addresses
        emails = re.findall(self.EMAIL_PATTERN, query)
        senders.extend(emails)

        # Extract person names
        names = re.findall(self.NAME_PATTERN, query, re.IGNORECASE)
        senders.extend([name[1] for name in names])  # Get captured group

        return senders
```

---

### 2.2 TemporalRetriever

**Purpose:** Retrieve emails from specific time period with recency decay

**Strategy:**
1. Parse query for time constraints
2. Search FAISS for semantic relevance
3. **Filter** by date range (if specified)
4. **Re-rank** with recency decay (newer = higher score)

**Implementation:**
```python
import datetime
from dateutil import parser as date_parser

class TemporalRetriever(BaseEmailRetriever):
    """Retrieves emails based on time constraints."""

    def __init__(self, faiss_manager, metadata_manager):
        self.faiss = faiss_manager
        self.metadata = metadata_manager
        self.time_extractor = TimeRangeExtractor()

    def retrieve(self, query: str, top_k: int = 15, filters: dict = None) -> List[Chunk]:
        # Extract time range from query
        time_range = self.time_extractor.extract(query)  # (start_date, end_date)

        # Get initial semantic results
        semantic_results = self.faiss.search(query, top_k=top_k * 3)

        # Filter by date range (if specified)
        if time_range:
            start_date, end_date = time_range
            filtered = [
                c for c in semantic_results
                if self._in_date_range(c.meta.get('date'), start_date, end_date)
            ]
        else:
            filtered = semantic_results

        # Re-rank with recency decay
        now = datetime.datetime.now()
        for chunk in filtered:
            semantic_score = chunk.meta['similarity']

            # Calculate recency score (exponential decay)
            email_date = self._parse_date(chunk.meta.get('date'))
            if email_date:
                days_ago = (now - email_date).days
                recency_score = math.exp(-days_ago / 30)  # Decay over 30 days
            else:
                recency_score = 0.5  # Unknown date

            # Combine scores (70% semantic, 30% recency)
            chunk.meta['final_score'] = 0.7 * semantic_score + 0.3 * recency_score

        # Sort and return
        filtered.sort(key=lambda c: c.meta['final_score'], reverse=True)
        return filtered[:top_k]

    def _in_date_range(self, date_str, start, end):
        date = self._parse_date(date_str)
        if not date:
            return False
        return start <= date <= end

    def _parse_date(self, date_str):
        try:
            return date_parser.parse(str(date_str))
        except:
            return None

    def get_name(self) -> str:
        return "TemporalRetriever"
```

**Time Range Extraction:**
```python
class TimeRangeExtractor:
    """Extracts time constraints from query."""

    RELATIVE_PATTERNS = {
        r'\blast\s+week\b': lambda: (datetime.datetime.now() - datetime.timedelta(days=7), datetime.datetime.now()),
        r'\blast\s+month\b': lambda: (datetime.datetime.now() - datetime.timedelta(days=30), datetime.datetime.now()),
        r'\byesterday\b': lambda: (datetime.datetime.now() - datetime.timedelta(days=1), datetime.datetime.now()),
        r'\brecent\b': lambda: (datetime.datetime.now() - datetime.timedelta(days=14), datetime.datetime.now()),
    }

    def extract(self, query: str):
        # Check relative patterns
        for pattern, range_func in self.RELATIVE_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                return range_func()

        # Check absolute dates (e.g., "November 2024")
        # ... (implementation for parsing absolute dates)

        return None  # No time constraint found
```

---

### 2.3 ThreadRetriever

**Purpose:** Reconstruct email threads and retrieve entire conversations

**Strategy:**
1. Identify thread from query or initial search
2. Group chunks by `subject` or `thread_id` metadata
3. Retrieve all chunks from same thread
4. Return chronologically sorted

**Implementation:**
```python
class ThreadRetriever(BaseEmailRetriever):
    """Retrieves entire email threads."""

    def __init__(self, faiss_manager, metadata_manager):
        self.faiss = faiss_manager
        self.metadata = metadata_manager

    def retrieve(self, query: str, top_k: int = 20, filters: dict = None) -> List[Chunk]:
        # Get initial results to identify thread
        seed_results = self.faiss.search(query, top_k=3)

        if not seed_results:
            return []

        # Use top result to identify thread
        seed_chunk = seed_results[0]
        thread_subject = self._normalize_subject(seed_chunk.meta.get('subject', ''))

        # Get all chunks from same thread
        all_chunks = self.metadata.get_all_chunks(doc_type='outlook_eml')
        thread_chunks = [
            c for c in all_chunks
            if self._normalize_subject(c.meta.get('subject', '')) == thread_subject
        ]

        # Sort by date
        thread_chunks.sort(key=lambda c: c.meta.get('date', ''))

        # Return up to top_k chunks from thread
        return thread_chunks[:top_k]

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject line (remove Re:, Fwd:, etc.)"""
        normalized = re.sub(r'^\s*(Re|Fwd|RE|FW):\s*', '', subject, flags=re.IGNORECASE)
        return normalized.strip().lower()

    def get_name(self) -> str:
        return "ThreadRetriever"
```

---

### 2.4 MultiAspectRetriever

**Purpose:** Combine multiple retrieval signals for complex queries

**Strategy:**
1. Run semantic search (FAISS)
2. Re-rank with multiple signals:
   - Semantic similarity (primary)
   - Recency decay (time-based boost)
   - Sender frequency (boost from frequent contributors)
   - Thread depth (boost root messages vs replies)

**Implementation:**
```python
class MultiAspectRetriever(BaseEmailRetriever):
    """Combines multiple retrieval signals."""

    def __init__(self, faiss_manager, metadata_manager):
        self.faiss = faiss_manager
        self.metadata = metadata_manager
        self.sender_stats = self._compute_sender_stats()  # Pre-compute sender authority

    def retrieve(self, query: str, top_k: int = 10, filters: dict = None) -> List[Chunk]:
        # Get initial semantic results
        semantic_results = self.faiss.search(query, top_k=top_k * 3)

        # Re-rank with multiple signals
        now = datetime.datetime.now()
        for chunk in semantic_results:
            # 1. Semantic score (40%)
            semantic_score = chunk.meta['similarity']

            # 2. Recency score (25%)
            email_date = self._parse_date(chunk.meta.get('date'))
            if email_date:
                days_ago = (now - email_date).days
                recency_score = math.exp(-days_ago / 30)
            else:
                recency_score = 0.5

            # 3. Sender authority (20%)
            sender = chunk.meta.get('sender', '')
            authority_score = self.sender_stats.get(sender, 0.5)  # 0-1 based on freq

            # 4. Thread position (15%)
            subject = chunk.meta.get('subject', '')
            is_root = not re.match(r'^\s*(Re|Fwd):', subject, re.IGNORECASE)
            thread_score = 1.0 if is_root else 0.7

            # Weighted combination
            chunk.meta['final_score'] = (
                0.40 * semantic_score +
                0.25 * recency_score +
                0.20 * authority_score +
                0.15 * thread_score
            )

        # Sort and return
        semantic_results.sort(key=lambda c: c.meta['final_score'], reverse=True)
        return semantic_results[:top_k]

    def _compute_sender_stats(self):
        """Pre-compute sender authority (email frequency)."""
        all_chunks = self.metadata.get_all_chunks(doc_type='outlook_eml')
        sender_counts = {}

        for chunk in all_chunks:
            sender = chunk.meta.get('sender', '')
            sender_counts[sender] = sender_counts.get(sender, 0) + 1

        # Normalize to 0-1 range
        max_count = max(sender_counts.values()) if sender_counts else 1
        return {
            sender: count / max_count
            for sender, count in sender_counts.items()
        }

    def get_name(self) -> str:
        return "MultiAspectRetriever"
```

---

## Component 3: Dynamic Top-K

**Purpose:** Adjust number of retrieved chunks based on query intent

**Strategy:**
```python
TOP_K_BY_INTENT = {
    'sender_query': 12,      # Show sender's message history
    'temporal_query': 15,    # Cover time range adequately
    'thread_summary': 20,    # Need full thread context
    'factual_lookup': 5,     # Precise answer, fewer chunks
    'multi_aspect': 10,      # Balanced default
}

def get_top_k(intent: str) -> int:
    return TOP_K_BY_INTENT.get(intent, 10)
```

---

## Component 4: EmailOrchestratorAgent

**Purpose:** Route queries to appropriate retriever based on intent

**Location:** `scripts/email/agents/email_orchestrator.py`

**Implementation:**
```python
class EmailOrchestratorAgent:
    """Routes email queries to specialized retrievers."""

    def __init__(self, project_root: Path, config: dict):
        self.project_root = project_root
        self.config = config

        # Initialize components
        self.intent_classifier = IntentClassifier(use_llm_fallback=True)

        # Initialize retrievers
        faiss_manager = FaissManager(project_root)
        metadata_manager = MetadataManager(project_root)

        self.retrievers = {
            'sender_query': SenderRetriever(faiss_manager, metadata_manager),
            'temporal_query': TemporalRetriever(faiss_manager, metadata_manager),
            'thread_summary': ThreadRetriever(faiss_manager, metadata_manager),
            'factual_lookup': MultiAspectRetriever(faiss_manager, metadata_manager),
            'multi_aspect': MultiAspectRetriever(faiss_manager, metadata_manager),
        }

    def retrieve(self, query: str) -> List[Chunk]:
        """Main retrieval method."""

        # 1. Classify intent
        intent_result = self.intent_classifier.classify(query)
        intent = intent_result.intent
        confidence = intent_result.confidence

        # 2. Get appropriate retriever
        retriever = self.retrievers[intent]

        # 3. Determine top-K
        top_k = get_top_k(intent)

        # 4. Retrieve chunks
        chunks = retriever.retrieve(query, top_k=top_k)

        # 5. Add metadata for logging
        for chunk in chunks:
            chunk.meta['intent'] = intent
            chunk.meta['intent_confidence'] = confidence
            chunk.meta['retriever'] = retriever.get_name()

        return chunks
```

---

## Integration with PipelineRunner

**Location:** `scripts/pipeline/runner.py` - `step_retrieve()` method

**Change:**
```python
def step_retrieve(self, query: str, top_k: int = 5):
    """Retrieve relevant chunks for query."""

    # Check if this is an email project
    has_email_docs = any(
        doc.metadata.get('doc_type') == 'outlook_eml'
        for doc in self.raw_docs
    )

    if has_email_docs:
        # Use Phase 2 EmailOrchestratorAgent
        orchestrator = EmailOrchestratorAgent(
            project_root=self.project.root,
            config=self.config
        )
        self.retrieved_chunks = orchestrator.retrieve(query)
        yield f"🎯 Retrieved {len(self.retrieved_chunks)} chunks using intent-aware retrieval"
    else:
        # Use standard retrieval for non-email projects
        retrieval_manager = RetrievalManager(
            project_root=self.project.root,
            config=self.config,
            top_k=top_k
        )
        self.retrieved_chunks = retrieval_manager.retrieve(query)
        yield f"Retrieved {len(self.retrieved_chunks)} chunks"
```

---

## Testing Strategy

### Test Dataset
Continue using Primo_List_2 (115 emails, 791 chunks) for validation

### Test Queries

**Sender Queries:**
1. "emails from Victoria Castro"
2. "what did Erin Nettifee say about FOLIO?"
3. "messages from primo@exlibrisusers.org"

**Temporal Queries:**
1. "recent discussions about NDE"
2. "emails from last week"
3. "what happened in November 2024?"

**Thread Summary:**
1. "summarize the migration discussion"
2. "thread about facet customization"
3. "conversation about Research Assistant"

**Factual Lookup:**
1. "how do I hide facet values in Primo using CSS?"
2. "what is the difference between Research Assistant and Ask Anything?"
3. "problems migrating from Ex Libris to EBSCO FOLIO?"

### Success Metrics

**Intent Classification:**
- Accuracy > 90% (validated manually on 20 test queries)
- Confidence > 0.7 for correct classifications

**Retrieval Quality:**
- Sender queries: Top 5 results from correct sender (precision)
- Temporal queries: Top 5 results within time range (precision)
- Thread queries: Complete thread retrieved (recall)
- Factual queries: Answer in top 3 results (MRR)

**Performance:**
- Intent classification: < 100ms (pattern-based), < 1s (LLM fallback)
- Retrieval: < 500ms for up to 20 chunks
- End-to-end: < 2s per query

---

## Implementation Timeline

### Week 1: Core Components
- ✅ Day 1: Phase 2 plan (this document)
- 🔲 Day 2: Intent Classifier (pattern-based + LLM)
- 🔲 Day 3: SenderRetriever + TemporalRetriever
- 🔲 Day 4: ThreadRetriever + MultiAspectRetriever
- 🔲 Day 5: EmailOrchestratorAgent

### Week 2: Integration & Testing
- 🔲 Day 6: Integrate with PipelineRunner
- 🔲 Day 7: Test on Primo_List_2 (all intent types)
- 🔲 Day 8: Tune retriever parameters
- 🔲 Day 9: Performance optimization
- 🔲 Day 10: Documentation

---

## Dependencies

**Python Packages:**
```toml
# Add to pyproject.toml
python-dateutil = "^2.8.2"  # For time range parsing
```

**Existing Components:**
- Phase 1 cleaning (quote removal, signature detection) ✅
- Semantic chunking ✅
- FAISS indexing ✅
- Metadata management ✅

---

## Success Criteria

Phase 2 is complete when:
1. ✅ Intent classifier achieves > 90% accuracy on test set
2. ✅ All 4 specialized retrievers implemented and tested
3. ✅ EmailOrchestratorAgent routes queries correctly
4. ✅ Retrieval quality improves vs Phase 1 baseline (measured by precision@5)
5. ✅ Performance meets targets (< 2s end-to-end)
6. ✅ Integration with PipelineRunner complete
7. ✅ Documentation complete (this plan + integration doc)

---

## Next Steps After Phase 2

**Phase 3:** Context Assembly & Ranking
- Thread deduplication (merge redundant chunks)
- Advanced multi-signal ranking
- Context window optimization for LLM

**Phase 4:** Quality & Monitoring
- Answer validation pipeline
- User feedback loop
- Performance dashboard

---

## References

- **Roadmap:** `docs/EMAIL_RAG_ENHANCEMENT_ROADMAP.md`
- **Phase 1:** `docs/EMAIL_PHASE1_INTEGRATION.md`
- **Base Retrieval:** `scripts/retrieval/retrieval_manager.py`

---

**Status:** Ready to implement
**Next Action:** Build Intent Classifier (`scripts/email/agents/intent_classifier.py`)
