# Email RAG Enhancement Roadmap: Evolution to State-of-the-Art

**Date:** 2025-11-22
**Status:** ROADMAP - Integration Plan
**Purpose:** Evolve current Email RAG module to 2025 state-of-the-art architecture
**Source:** Integration of `EMAIL_RAG_ARCHITECTURE.md` + `Enhancing Email Data for RAG.md`

---

## Executive Summary

### Current State → Future State

**What We Have (Foundation):**
- ✅ Email loading (EML, MSG, MBOX, Outlook connector)
- ✅ Basic thread detection (header-based)
- ✅ Agentic orchestration started (EmailOrchestratorAgent, IntentDetector)
- ✅ Specialized retrievers (Sender, Temporal, Thread, Multi-Aspect)
- ✅ Email cleaning (basic regex-based)
- ✅ Paragraph-based chunking
- ✅ Dense vector search (FAISS)

**What We're Adding (Evolution):**
- 🎯 **GraphRAG**: Knowledge graph of people, entities, threads
- 🎯 **Parent-Child Indexing**: Thread-aware retrieval pattern
- 🎯 **Semantic Chunking**: Topic-aware boundaries
- 🎯 **Hybrid Search**: Dense + Sparse (BM25) with RRF
- 🎯 **ML-Based Cleaning**: Signature/disclaimer detection
- 🎯 **Quote Deduplication**: Remove reply chains
- 🎯 **Enhanced Agents**: Router, query transformation, HyDE
- 🎯 **Multimodal**: Attachment processing (PDFs, images)
- 🎯 **Privacy & Security**: PII redaction, RBAC

### Architectural Vision

```
┌─────────────────────────────────────────────────────────────┐
│                  AGENTIC ORCHESTRATION LAYER                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Router Agent │→ │Query Transform│→ │ Result Synth │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    RETRIEVAL LAYER (Hybrid)                 │
│  ┌─────────────────┐    ┌────────────────┐                 │
│  │  Vector Search  │ +  │  Graph Search  │                 │
│  │ (Semantic/Dense)│    │  (Structural)  │                 │
│  └─────────────────┘    └────────────────┘                 │
│           ↓                      ↓                          │
│  ┌─────────────────┐    ┌────────────────┐                 │
│  │ Sparse (BM25)   │    │ Community      │                 │
│  │ (Keyword/Exact) │    │ Summaries      │                 │
│  └─────────────────┘    └────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│                    INDEXING LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Parent-Child  │  │Knowledge     │  │Multimodal    │      │
│  │Vector Index  │  │Graph (Neo4j) │  │Embeddings    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    INGESTION LAYER                          │
│  Thread → Clean → Dedupe → Chunk → Embed → Index           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ML Signature  │  │Quote         │  │Semantic      │      │
│  │Detection     │  │Deduplication │  │Chunking      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Enhancement Plan

### Stage 1: Ingestion Pipeline Enhancement

**Current:** Basic email loading → Basic cleaning → Paragraph chunking
**Target:** Advanced cleaning → Thread-aware deduplication → Semantic chunking

#### 1.1 Advanced Email Cleaning

**What:** ML-based signature/disclaimer detection + quote deduplication

**Why:** Current regex-based cleaning misses complex signatures and indexes quoted replies multiple times, causing "vector collapse" where boilerplate text dominates semantic space.

**Implementation:**

```python
# scripts/email/ingestion/advanced_cleaner.py

from transformers import pipeline
from difflib import SequenceMatcher

class AdvancedEmailCleaner:
    """ML-based email cleaning pipeline."""

    def __init__(self):
        # Fine-tuned DistilBERT for signature detection
        self.signature_detector = pipeline(
            "text-classification",
            model="huggingface/signature-detection-model"  # Fine-tune on our data
        )

    def clean(self, email: Email) -> Email:
        """Clean email using ML and overlap detection."""

        # Step 1: ML-based signature detection
        segments = self.split_into_segments(email.body)
        clean_segments = []

        for segment in segments:
            classification = self.signature_detector(segment)
            if classification['label'] != 'signature':
                clean_segments.append(segment)

        email.body = '\n\n'.join(clean_segments)

        # Step 2: Quote deduplication (MinHash)
        if email.in_reply_to:
            parent_email = self.load_parent(email.in_reply_to)
            email.body = self.remove_quoted_text(email.body, parent_email.body)

        return email

    def remove_quoted_text(self, current: str, parent: str) -> str:
        """Remove text quoted from parent email using overlap detection."""

        # Split into lines
        current_lines = current.split('\n')
        parent_lines = parent.split('\n')

        # Find matching sequences using SequenceMatcher
        matcher = SequenceMatcher(None, current_lines, parent_lines)

        # Remove lines with high similarity to parent
        unique_lines = []
        for i, line in enumerate(current_lines):
            # Check if this line is part of a quoted block
            is_quote = False
            for block in matcher.get_matching_blocks():
                if block.a <= i < block.a + block.size and block.size > 2:
                    is_quote = True
                    break

            if not is_quote:
                unique_lines.append(line)

        return '\n'.join(unique_lines)
```

**Integration Point:** `scripts/ingestion/email_loader.py` - add cleaning step before creating RawDoc

**Dependencies:**
- `transformers` library
- Fine-tuned signature detection model (can start with Hugging Face dataset)

**Timeline:** 1 week
**Priority:** HIGH (foundational quality improvement)

---

#### 1.2 Semantic Chunking

**What:** Replace paragraph-based chunking with semantic boundary detection

**Why:** Emails often contain multiple topics in single paragraph or split topics across paragraphs. Semantic chunking creates coherent, topic-focused chunks.

**Implementation:**

```python
# scripts/chunking/semantic_chunker.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticChunker:
    """Semantic boundary-based chunking for emails."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold

    def chunk(self, text: str) -> List[str]:
        """Split text at semantic boundaries."""

        # Split into sentences
        sentences = self.split_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Embed each sentence
        embeddings = self.model.encode(sentences)

        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i+1]]
            )[0][0]
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold
        chunks = []
        current_chunk = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                # Topic shift detected - create new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i+1]]
            else:
                # Same topic - continue chunk
                current_chunk.append(sentences[i+1])

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (handle email-specific patterns)."""
        import re

        # Handle common email patterns
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize whitespace

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        return [s.strip() for s in sentences if s.strip()]
```

**Integration Point:** `scripts/chunking/chunker_v3.py` - add semantic strategy option

**Config:**
```yaml
# configs/chunk_rules.yaml
outlook_eml:
  strategy: "semantic"  # NEW: semantic boundary detection
  similarity_threshold: 0.7
  min_tokens: 20
  max_tokens: 300
```

**Timeline:** 1 week
**Priority:** HIGH (directly improves retrieval quality)

---

### Stage 2: Parent-Child Indexing Pattern

**What:** Store full threads as "parent" documents, individual emails/chunks as "children"

**Why:** Solves context fragmentation - when a chunk is retrieved, we return the full thread context to the LLM.

**Implementation:**

```python
# scripts/email/indexing/parent_child.py

class ParentChildIndexer:
    """Thread-aware indexing with parent-child pattern."""

    def __init__(self, vector_db, document_store):
        self.vector_db = vector_db  # FAISS for child chunks
        self.document_store = document_store  # MongoDB/file for parent threads

    def index_thread(self, thread: EmailThread):
        """Index a complete email thread using parent-child pattern."""

        # Step 1: Store parent (full thread) in document store
        parent_id = f"thread_{thread.thread_id}"
        parent_doc = {
            'id': parent_id,
            'thread_id': thread.thread_id,
            'subject': thread.subject,
            'participants': thread.get_all_senders(),
            'date_range': (thread.first_date, thread.last_date),
            'full_content': thread.get_chronological_content(),  # Complete thread
            'email_count': len(thread.emails)
        }
        self.document_store.save(parent_doc)

        # Step 2: Create child chunks from individual emails
        chunks = []
        for email in thread.emails:
            email_chunks = self.chunk_email(email)

            for chunk in email_chunks:
                # Link child to parent
                chunk.meta['parent_id'] = parent_id
                chunk.meta['parent_type'] = 'thread'
                chunk.meta['email_id'] = email.id
                chunk.meta['sender'] = email.sender
                chunk.meta['date'] = email.date
                chunks.append(chunk)

        # Step 3: Embed and index child chunks
        for chunk in chunks:
            chunk.embedding = self.embed(chunk.text)
            self.vector_db.add(chunk)

        return parent_id, len(chunks)

    def retrieve_with_context(self, query: str, top_k: int = 5):
        """Retrieve child chunks and expand to parent context."""

        # Step 1: Vector search for relevant child chunks
        child_chunks = self.vector_db.search(query, top_k=top_k)

        # Step 2: Get unique parent IDs
        parent_ids = set(chunk.meta['parent_id'] for chunk in child_chunks)

        # Step 3: Retrieve full parent documents
        parents = []
        for parent_id in parent_ids:
            parent = self.document_store.get(parent_id)

            # Attach relevant child chunks
            parent['matched_chunks'] = [
                c for c in child_chunks if c.meta['parent_id'] == parent_id
            ]
            parents.append(parent)

        return parents  # Return full threads with highlighted relevant chunks
```

**Data Structure:**

```
Document Store (MongoDB/Files):
┌────────────────────────────────────────┐
│ Parent: thread_abc123                  │
│ ├─ thread_id: abc123                   │
│ ├─ subject: "Q3 Budget Discussion"     │
│ ├─ full_content: [email1, email2, ...] │
│ └─ participants: [Alice, Bob, Carol]   │
└────────────────────────────────────────┘

Vector DB (FAISS):
┌─────────────────────────────────────────┐
│ Child Chunk 1                           │
│ ├─ text: "I approve the budget..."     │
│ ├─ embedding: [0.1, 0.2, ...]          │
│ └─ parent_id: "thread_abc123"          │
├─────────────────────────────────────────┤
│ Child Chunk 2                           │
│ ├─ text: "When can we proceed..."      │
│ ├─ embedding: [0.3, 0.1, ...]          │
│ └─ parent_id: "thread_abc123"          │
└─────────────────────────────────────────┘
```

**Integration Point:** New indexing path in `scripts/pipeline/runner.py` - step_index_parent_child()

**Timeline:** 2 weeks
**Priority:** HIGH (architectural foundation for GraphRAG)

---

### Stage 3: GraphRAG Implementation

**What:** Build knowledge graph of emails, people, entities, and relationships

**Why:** Enables structural queries ("Who knows about X?", "Communication sentiment over time", "Key stakeholders") that vector search cannot answer.

#### 3.1 Knowledge Graph Schema

```python
# scripts/email/graph/schema.py

"""
Email Knowledge Graph Schema

NODES:
- Person: email participants
- Email: individual emails
- Thread: conversation threads
- Entity: extracted topics/concepts
- Attachment: file attachments

RELATIONSHIPS:
- (Person)-[SENT]->(Email)
- (Email)-[RECEIVED_BY]->(Person)
- (Email)-[PART_OF]->(Thread)
- (Email)-[REPLIES_TO]->(Email)
- (Email)-[MENTIONS]->(Entity)
- (Email)-[HAS_ATTACHMENT]->(Attachment)
- (Person)-[COLLABORATES_WITH]->(Person)  # Derived
- (Person)-[IN_COMMUNITY]->(Community)     # Computed
"""

class EmailGraphSchema:
    """Neo4j schema for email knowledge graph."""

    PERSON_PROPERTIES = {
        'email_address': 'string',
        'name': 'string',
        'department': 'string',
        'role': 'string',
        'email_count': 'int',
        'centrality_score': 'float'  # PageRank
    }

    EMAIL_PROPERTIES = {
        'id': 'string',
        'subject': 'string',
        'body': 'string',
        'date': 'datetime',
        'embedding_summary': 'vector',  # Avg embedding for filtering
        'has_attachments': 'boolean',
        'sentiment': 'float'
    }

    THREAD_PROPERTIES = {
        'id': 'string',
        'subject_normalized': 'string',
        'start_date': 'datetime',
        'end_date': 'datetime',
        'email_count': 'int',
        'participant_count': 'int',
        'community_id': 'string'
    }
```

#### 3.2 Graph Construction Pipeline

```python
# scripts/email/graph/builder.py

from neo4j import GraphDatabase

class EmailGraphBuilder:
    """Build email knowledge graph from processed emails."""

    def __init__(self, neo4j_uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    def build_graph(self, emails: List[Email]):
        """Construct graph from email dataset."""

        with self.driver.session() as session:
            # Create person nodes
            self.create_person_nodes(session, emails)

            # Create email nodes
            self.create_email_nodes(session, emails)

            # Create thread nodes
            threads = self.reconstruct_threads(emails)
            self.create_thread_nodes(session, threads)

            # Create relationships
            self.create_sent_relationships(session, emails)
            self.create_reply_relationships(session, emails)
            self.create_thread_relationships(session, emails)

            # Extract and link entities
            self.extract_and_link_entities(session, emails)

            # Compute derived relationships
            self.compute_collaboration_graph(session)
            self.compute_community_detection(session)

    def create_person_nodes(self, session, emails):
        """Create Person nodes from email participants."""

        # Get unique senders/receivers
        people = set()
        for email in emails:
            people.add((email.sender_email, email.sender_name))
            for recipient in email.recipients:
                people.add((recipient['email'], recipient['name']))

        for email_addr, name in people:
            session.run("""
                MERGE (p:Person {email: $email})
                SET p.name = $name,
                    p.email_count = 0
                """,
                email=email_addr,
                name=name
            )

    def create_email_nodes(self, session, emails):
        """Create Email nodes."""

        for email in emails:
            session.run("""
                CREATE (e:Email {
                    id: $id,
                    subject: $subject,
                    body: $body,
                    date: datetime($date),
                    has_attachments: $has_attachments
                })
                """,
                id=email.id,
                subject=email.subject,
                body=email.body[:1000],  # Truncate for storage
                date=email.date.isoformat(),
                has_attachments=len(email.attachments) > 0
            )

    def create_sent_relationships(self, session, emails):
        """Create (Person)-[SENT]->(Email) relationships."""

        for email in emails:
            session.run("""
                MATCH (p:Person {email: $sender})
                MATCH (e:Email {id: $email_id})
                CREATE (p)-[:SENT]->(e)
                SET p.email_count = p.email_count + 1
                """,
                sender=email.sender_email,
                email_id=email.id
            )

            # RECEIVED_BY relationships
            for recipient in email.recipients:
                session.run("""
                    MATCH (p:Person {email: $recipient})
                    MATCH (e:Email {id: $email_id})
                    CREATE (e)-[:RECEIVED_BY]->(p)
                    """,
                    recipient=recipient['email'],
                    email_id=email.id
                )

    def compute_community_detection(self, session):
        """Apply Leiden algorithm for community detection."""

        session.run("""
            CALL gds.leiden.write('email-network', {
                writeProperty: 'community_id',
                relationshipTypes: ['SENT', 'RECEIVED_BY']
            })
            """)

        # Generate community summaries using LLM
        communities = session.run("""
            MATCH (p:Person)
            RETURN DISTINCT p.community_id AS community_id
            """)

        for community in communities:
            self.generate_community_summary(session, community['community_id'])

    def generate_community_summary(self, session, community_id):
        """Generate LLM summary of community topics."""

        # Get all emails in community
        result = session.run("""
            MATCH (p:Person {community_id: $community_id})-[:SENT]->(e:Email)
            RETURN e.subject AS subject, e.body AS body
            LIMIT 50
            """,
            community_id=community_id
        )

        emails_text = "\n\n".join([
            f"Subject: {r['subject']}\n{r['body'][:500]}"
            for r in result
        ])

        # LLM summarization
        summary = self.llm_summarize(emails_text)

        # Store summary
        session.run("""
            MERGE (c:Community {id: $community_id})
            SET c.summary = $summary
            """,
            community_id=community_id,
            summary=summary
        )
```

#### 3.3 Hybrid Graph+Vector Retrieval

```python
# scripts/email/retrieval/hybrid_graph_vector.py

class HybridGraphVectorRetriever:
    """Combine graph and vector retrieval."""

    def __init__(self, graph_db, vector_db):
        self.graph = graph_db
        self.vector = vector_db

    def retrieve(self, query: str, intent: str):
        """Route query to appropriate index based on intent."""

        if intent == "structural":
            # Example: "Who did Alice email about Project X?"
            return self.graph_retrieve(query)

        elif intent == "semantic":
            # Example: "Emails about budget concerns"
            return self.vector_retrieve(query)

        elif intent == "global":
            # Example: "What are the main topics discussed?"
            return self.community_retrieve(query)

        else:
            # Hybrid: combine both
            return self.hybrid_retrieve(query)

    def graph_retrieve(self, query: str):
        """Structural query via Cypher."""

        # Extract entities from query
        entities = self.extract_entities(query)  # NER

        # Example Cypher query
        with self.graph.session() as session:
            result = session.run("""
                MATCH (p:Person {name: $person})-[:SENT]->(e:Email)
                      -[:MENTIONS]->(entity:Entity {name: $entity})
                RETURN e
                ORDER BY e.date DESC
                LIMIT 10
                """,
                person=entities['person'],
                entity=entities['entity']
            )

            return [record['e'] for record in result]

    def community_retrieve(self, query: str):
        """Retrieve pre-computed community summaries."""

        # Embed query
        query_embedding = self.vector.embed(query)

        # Find closest communities
        with self.graph.session() as session:
            result = session.run("""
                MATCH (c:Community)
                RETURN c.id AS id, c.summary AS summary
                """)

            communities = list(result)

        # Rank communities by semantic similarity to query
        ranked = sorted(
            communities,
            key=lambda c: cosine_sim(query_embedding, self.vector.embed(c['summary'])),
            reverse=True
        )

        return ranked[:5]
```

**Integration Point:** New retrieval strategy in `scripts/retrieval/retrieval_manager.py`

**Dependencies:**
- Neo4j database
- `neo4j` Python driver
- Graph Data Science (GDS) library for Leiden algorithm

**Timeline:** 3-4 weeks
**Priority:** MEDIUM-HIGH (major architectural addition)

---

### Stage 4: Hybrid Search (Dense + Sparse)

**What:** Combine semantic (dense vectors) with keyword (BM25 sparse) retrieval

**Why:** Dense vectors miss exact identifiers (invoice numbers, error codes). Sparse retrieval captures these but misses semantic similarity. Hybrid gets both.

**Implementation:**

```python
# scripts/email/retrieval/hybrid_search.py

from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearchEngine:
    """Hybrid dense (semantic) + sparse (BM25) retrieval with RRF fusion."""

    def __init__(self, vector_index, sparse_index=None):
        self.vector_index = vector_index  # FAISS
        self.sparse_index = sparse_index or self.build_bm25_index()

    def build_bm25_index(self):
        """Build BM25 index from corpus."""

        # Get all documents
        docs = self.vector_index.get_all_documents()

        # Tokenize for BM25
        tokenized_docs = [doc.text.lower().split() for doc in docs]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs)

        return bm25

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """Hybrid search with RRF (Reciprocal Rank Fusion).

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
        """

        # Dense retrieval (semantic)
        dense_results = self.vector_index.search(query, top_k=100)

        # Sparse retrieval (BM25)
        query_tokens = query.lower().split()
        sparse_scores = self.sparse_index.get_scores(query_tokens)
        sparse_results = np.argsort(sparse_scores)[::-1][:100]

        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        scores = {}

        # Score from dense retrieval
        for rank, result in enumerate(dense_results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (k + rank + 1)

        # Score from sparse retrieval
        for rank, doc_idx in enumerate(sparse_results):
            doc_id = self.vector_index.get_doc_id(doc_idx)
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k documents
        top_docs = [self.vector_index.get_document(doc_id) for doc_id, _ in ranked[:top_k]]

        return top_docs
```

**Integration Point:** `scripts/retrieval/retrieval_manager.py` - add hybrid strategy

**Config:**
```yaml
# project config
retrieval:
  strategy: "hybrid"  # NEW: hybrid dense+sparse
  hybrid_alpha: 0.7   # Weight: 70% semantic, 30% keyword
  sparse_method: "bm25"
```

**Timeline:** 1 week
**Priority:** MEDIUM (improves precision for exact matches)

---

### Stage 5: Enhanced Agentic Workflows

**What:** Router agent, query transformation (HyDE, Multi-Query), corrective RAG

**Why:** Static retrieval often fails. Agents can plan, reformulate, and iterate to find the right answer.

**Implementation:**

```python
# scripts/email/agents/router_agent.py

from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langgraph.graph import StateGraph

class EmailRAGRouter:
    """Agentic router for email RAG queries."""

    def __init__(self, vector_retriever, graph_retriever, llm):
        self.vector = vector_retriever
        self.graph = graph_retriever
        self.llm = llm

        # Build LangGraph workflow
        self.workflow = self.build_workflow()

    def build_workflow(self):
        """Build state machine for agentic retrieval."""

        workflow = StateGraph()

        # States
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("vector_search", self.vector_search)
        workflow.add_node("graph_search", self.graph_search)
        workflow.add_node("community_search", self.community_search)
        workflow.add_node("hyde_transform", self.hyde_transform)
        workflow.add_node("multi_query", self.multi_query_expansion)
        workflow.add_node("synthesize", self.synthesize_results)
        workflow.add_node("evaluate", self.evaluate_confidence)
        workflow.add_node("rewrite_query", self.rewrite_query)

        # Edges (transitions)
        workflow.set_entry_point("classify_intent")

        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "vector": "vector_search",
                "graph": "graph_search",
                "global": "community_search",
                "complex": "multi_query"
            }
        )

        workflow.add_edge("vector_search", "evaluate")
        workflow.add_edge("graph_search", "synthesize")
        workflow.add_edge("community_search", "synthesize")
        workflow.add_edge("multi_query", "synthesize")

        workflow.add_conditional_edges(
            "evaluate",
            self.check_confidence,
            {
                "high": "synthesize",
                "low": "rewrite_query"  # Corrective RAG
            }
        )

        workflow.add_edge("rewrite_query", "hyde_transform")
        workflow.add_edge("hyde_transform", "vector_search")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def classify_intent(self, state):
        """Classify query intent using LLM."""

        prompt = f"""Classify this email query into one of these categories:
        - vector: Semantic search (e.g., "emails about budget concerns")
        - graph: Structural query (e.g., "who emailed Alice about Project X?")
        - global: High-level summary (e.g., "what topics were discussed?")
        - complex: Multi-part query (e.g., "compare marketing plans 2023 vs 2024")

        Query: {state['query']}

        Category:"""

        intent = self.llm.invoke(prompt).strip().lower()
        state['intent'] = intent
        return state

    def hyde_transform(self, state):
        """Generate hypothetical document (HyDE)."""

        prompt = f"""Generate a hypothetical email that would answer this query:

        Query: {state['query']}

        Write a realistic email (2-3 paragraphs) that contains the answer."""

        hypothetical_email = self.llm.invoke(prompt)
        state['transformed_query'] = hypothetical_email
        return state

    def multi_query_expansion(self, state):
        """Expand complex query into sub-queries."""

        prompt = f"""Break this complex query into 2-4 simpler sub-queries:

        Query: {state['query']}

        Sub-queries (one per line):"""

        sub_queries = self.llm.invoke(prompt).strip().split('\n')
        state['sub_queries'] = sub_queries

        # Execute sub-queries in parallel
        results = []
        for sq in sub_queries:
            results.extend(self.vector.search(sq, top_k=5))

        state['results'] = results
        return state

    def evaluate_confidence(self, state):
        """Evaluate retrieval confidence."""

        results = state.get('results', [])

        if not results:
            state['confidence'] = 'low'
        elif results[0].similarity < 0.6:
            state['confidence'] = 'low'
        else:
            state['confidence'] = 'high'

        return state

    def rewrite_query(self, state):
        """Rewrite query if initial retrieval failed (Corrective RAG)."""

        prompt = f"""The initial query didn't find good results. Rewrite it to be more specific:

        Original: {state['query']}
        Results found: {len(state.get('results', []))} (low relevance)

        Rewritten query:"""

        rewritten = self.llm.invoke(prompt).strip()
        state['query'] = rewritten
        state['iteration'] = state.get('iteration', 0) + 1

        # Prevent infinite loops
        if state['iteration'] > 2:
            state['confidence'] = 'high'  # Force exit

        return state
```

**Integration Point:** `scripts/pipeline/runner.py` - replace step_retrieve() with agentic workflow

**Timeline:** 2-3 weeks
**Priority:** MEDIUM (improves query success rate)

---

### Stage 6: Multimodal & Security

#### 6.1 Attachment Processing

**What:** Extract and embed PDFs, images, Excel files from email attachments

**Implementation:**

```python
# scripts/email/multimodal/attachment_processor.py

from PIL import Image
import pytesseract
from transformers import CLIPModel, CLIPProcessor

class AttachmentProcessor:
    """Process email attachments (PDFs, images, Excel)."""

    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def process(self, attachment: Attachment) -> ProcessedAttachment:
        """Process attachment based on type."""

        if attachment.type == 'application/pdf':
            return self.process_pdf(attachment)
        elif attachment.type.startswith('image/'):
            return self.process_image(attachment)
        elif attachment.type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            return self.process_excel(attachment)
        else:
            return None

    def process_image(self, attachment: Attachment) -> ProcessedAttachment:
        """Extract text (OCR) and create multimodal embedding."""

        image = Image.open(attachment.path)

        # OCR for text
        text = pytesseract.image_to_string(image)

        # CLIP embedding (image + text in same vector space)
        inputs = self.clip_processor(
            text=text if text else "image",
            images=image,
            return_tensors="pt",
            padding=True
        )

        outputs = self.clip_model(**inputs)
        embedding = outputs.image_embeds[0].detach().numpy()

        return ProcessedAttachment(
            id=attachment.id,
            type='image',
            text=text,
            embedding=embedding,
            metadata={'dimensions': image.size}
        )
```

**Timeline:** 2 weeks
**Priority:** LOW-MEDIUM (nice-to-have, not critical)

#### 6.2 PII Redaction & RBAC

**What:** Detect and redact PII at ingestion, enforce access control at retrieval

**Implementation:**

```python
# scripts/email/security/pii_redaction.py

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIRedactor:
    """Detect and redact PII from emails."""

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> tuple[str, dict]:
        """Redact PII and return mapping for potential restoration."""

        # Detect PII entities
        results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=['PHONE_NUMBER', 'EMAIL_ADDRESS', 'CREDIT_CARD', 'SSN']
        )

        # Anonymize with consistent tokens
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "PHONE_NUMBER": {"type": "replace", "new_value": "<PHONE>"},
                "EMAIL_ADDRESS": {"type": "replace", "new_value": "<EMAIL>"},
                "CREDIT_CARD": {"type": "replace", "new_value": "<CARD>"},
                "SSN": {"type": "replace", "new_value": "<SSN>"}
            }
        )

        return anonymized.text, {r.entity_type: r.start for r in results}

# scripts/email/security/rbac.py

class RBACFilter:
    """Role-based access control for email retrieval."""

    def filter_by_acl(self, chunks: List[Chunk], user_groups: List[str]):
        """Filter chunks based on user's group memberships."""

        accessible = []
        for chunk in chunks:
            # Check if user has access
            chunk_acl = chunk.meta.get('acl_groups', ['public'])

            if 'public' in chunk_acl or any(g in user_groups for g in chunk_acl):
                accessible.append(chunk)

        return accessible
```

**Integration Point:** `scripts/ingestion/email_loader.py` (PII redaction), `scripts/retrieval/retrieval_manager.py` (RBAC)

**Timeline:** 1-2 weeks
**Priority:** HIGH (critical for production deployment with sensitive data)

---

## Implementation Phases

### Phase 1: Foundation Enhancements (3-4 weeks)
**Focus:** Improve data quality and chunking

- ✅ ML-based signature detection
- ✅ Quote deduplication
- ✅ Semantic chunking
- ✅ PII redaction

**Outcome:** Cleaner data, better chunk boundaries

### Phase 2: Indexing Evolution (4-5 weeks)
**Focus:** Structural foundations

- ✅ Parent-Child indexing pattern
- ✅ Hybrid search (Dense + BM25)
- ✅ Enhanced metadata schema

**Outcome:** Thread-aware retrieval, better precision

### Phase 3: GraphRAG Integration (4-6 weeks)
**Focus:** Knowledge graph

- ✅ Neo4j setup
- ✅ Graph schema and construction
- ✅ Community detection
- ✅ Hybrid graph+vector retrieval

**Outcome:** Structural queries, global summarization

### Phase 4: Agentic Orchestration (3-4 weeks)
**Focus:** Intelligent routing and planning

- ✅ Router agent (LangGraph)
- ✅ Query transformation (HyDE, Multi-Query)
- ✅ Corrective RAG
- ✅ Intent classification

**Outcome:** Adaptive retrieval, higher success rate

### Phase 5: Multimodal & Security (2-3 weeks)
**Focus:** Production-readiness

- ✅ Attachment processing (PDFs, images)
- ✅ RBAC enforcement
- ✅ Audit logging

**Outcome:** Complete, production-ready system

---

## Integration with Existing System

### Current Architecture Mapping

**What We Keep:**
```
scripts/
├─ ingestion/email_loader.py      ✅ Keep (enhance with cleaning)
├─ chunking/chunker_v3.py          ✅ Keep (add semantic strategy)
├─ embeddings/unified_embedder.py  ✅ Keep (no changes)
├─ agents/email/
│   ├─ email_orchestrator.py       ✅ Keep (integrate with router)
│   ├─ intent_detector.py          ✅ Keep (enhance)
│   ├─ sender_retriever.py         ✅ Keep
│   ├─ temporal_retriever.py       ✅ Keep
│   └─ thread_retriever.py         ✅ Keep (upgrade to parent-child)
└─ retrieval/retrieval_manager.py  ✅ Keep (add hybrid + graph strategies)
```

**What We Add:**
```
scripts/email/
├─ ingestion/
│   └─ advanced_cleaner.py         🆕 ML-based cleaning
├─ chunking/
│   └─ semantic_chunker.py         🆕 Semantic boundaries
├─ indexing/
│   └─ parent_child.py             🆕 Thread-aware indexing
├─ graph/
│   ├─ schema.py                   🆕 Neo4j schema
│   ├─ builder.py                  🆕 Graph construction
│   └─ queries.py                  🆕 Cypher templates
├─ retrieval/
│   ├─ hybrid_search.py            🆕 Dense+Sparse fusion
│   └─ hybrid_graph_vector.py      🆕 Graph+Vector routing
├─ agents/
│   ├─ router_agent.py             🆕 LangGraph workflow
│   └─ query_transformer.py        🆕 HyDE, Multi-Query
├─ multimodal/
│   └─ attachment_processor.py     🆕 PDF/image processing
└─ security/
    ├─ pii_redaction.py            🆕 Presidio integration
    └─ rbac.py                     🆕 Access control
```

### Migration Strategy

**No Breaking Changes:**
- All enhancements are additive
- Existing email projects continue to work
- New features opt-in via configuration

**Config-Driven Activation:**
```yaml
# data/projects/Primo_List/config.yml

email_enhancements:
  # Phase 1
  advanced_cleaning: true
  semantic_chunking: true
  pii_redaction: false  # Opt-in

  # Phase 2
  parent_child_indexing: true
  hybrid_search:
    enabled: true
    alpha: 0.7

  # Phase 3
  graph_rag:
    enabled: true
    neo4j_uri: "bolt://localhost:7687"

  # Phase 4
  agentic_workflows:
    enabled: true
    router: "langgraph"

  # Phase 5
  multimodal:
    enabled: false  # Opt-in
```

---

## Success Metrics

### Retrieval Quality
- **Hit Rate**: % of queries finding relevant email (target: >90%)
- **MRR (Mean Reciprocal Rank)**: Avg position of first relevant result (target: >0.7)
- **Context Precision**: % of retrieved chunks actually relevant (target: >80%)

### Operational Efficiency
- **Ingestion Speed**: Emails processed per second (target: >100 emails/sec)
- **Query Latency**: Time from query to results (target: <2 seconds for hybrid, <5 seconds for graph)
- **Storage Efficiency**: Deduplication ratio (target: >30% reduction from quote removal)

### User Experience
- **Answer Quality**: LLM-as-judge faithfulness score (target: >0.85)
- **Query Success Rate**: % of queries yielding actionable answer (target: >85%)
- **Zero-Result Rate**: % of queries with no results (target: <5%)

---

## Conclusion

This roadmap transforms our email RAG system from a functional baseline to a state-of-the-art architecture by integrating:

1. **Advanced Ingestion**: ML-based cleaning, quote deduplication
2. **Semantic Chunking**: Topic-aware boundaries
3. **Parent-Child Indexing**: Thread context preservation
4. **GraphRAG**: Structural reasoning and global summarization
5. **Hybrid Search**: Dense+Sparse fusion
6. **Agentic Workflows**: Adaptive retrieval with planning
7. **Security**: PII redaction, RBAC

**Timeline:** 16-23 weeks (4-6 months) across 5 phases
**Risk:** Medium (architectural additions, not rewrites)
**ROI:** High (transforms email archives from static storage to intelligent assistant)

The system remains **backward compatible** - existing projects work unchanged, new features activate via configuration. This ensures we can deploy incrementally and validate each enhancement before proceeding to the next.

---

**Next Action:** Review roadmap → Prioritize phases → Begin Phase 1 implementation
**Document Status:** READY FOR REVIEW
