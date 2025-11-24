# Architecture Overview

**Version:** 1.1.0
**Last Updated:** 2025-11-23

This document provides a technical overview of the Multi-Source RAG Platform architecture, components, and data flow.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Email Agentic Strategy](#email-agentic-strategy)
5. [Production Features](#production-features-v110)
6. [Design Decisions](#design-decisions)

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  (Streamlit UI / CLI)                                          │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE RUNNER                            │
│  • Orchestrates workflow                                        │
│  • Validation (v1.1.0)                                         │
│  • Smart fallback (v1.1.0)                                     │
└───┬─────────┬─────────┬──────────┬──────────┬─────────┬────────┘
    │         │         │          │          │         │
    ▼         ▼         ▼          ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│INGEST  │ │CHUNK   │ │ENRICH  │ │EMBED   │ │RETRIEVE│ │  ASK   │
│        │ │        │ │(opt)   │ │        │ │        │ │        │
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    │          │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Raw     │ │Chunks  │ │Enhanced│ │FAISS   │ │Retrieved││Final  │
│Docs    │ │TSV     │ │Chunks  │ │Index   │ │Chunks   │ │Answer  │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

### Component Layers

**Layer 1: Interface**
- Streamlit UI (`scripts/ui/ui_v3.py`)
- CLI (`app/cli.py`)

**Layer 2: Orchestration**
- PipelineRunner (`scripts/pipeline/runner.py`)
- Pipeline Validator (`scripts/pipeline/validator.py`) - NEW v1.1.0
- Email Orchestrator (`scripts/agents/email_orchestrator.py`)

**Layer 3: Processing**
- Ingestion (`scripts/ingestion/`)
- Chunking (`scripts/chunking/`)
- Embedding (`scripts/embeddings/`)
- Retrieval (`scripts/retrieval/`)

**Layer 4: Data Storage**
- FAISS indices (vector search)
- Metadata JSONL (chunk metadata)
- TSV files (intermediate chunks)

---

## Core Components

### 1. Ingestion Manager

**Location:** `scripts/ingestion/manager.py`

**Purpose:** Loads documents and extracts text + metadata.

**Key Features:**
- Multi-format support (PDF, DOCX, PPTX, XLSX, emails)
- Loader registry pattern
- Deduplication by content hash
- Image extraction (optional)

**Flow:**
```python
IngestionManager
  ├─ Scans input/raw/ recursively
  ├─ Matches file extension → Loader
  ├─ Loader extracts (content, metadata)
  ├─ Creates RawDoc objects
  └─ Deduplicates by SHA256(content + images)
```

**Output:** `List[RawDoc]` in memory

---

### 2. Chunker (v3)

**Location:** `scripts/chunking/chunker_v3.py`

**Purpose:** Splits documents into semantic chunks for retrieval.

**Key Features:**
- Paragraph-based splitting (blank line boundaries)
- Token-aware merging (min/max tokens)
- Document-type specific rules (`configs/chunk_rules.yaml`)
- Image-aware (chunks with images bypass min threshold)

**Flow:**
```python
Chunker
  ├─ Splits on blank lines (paragraphs)
  ├─ Loads ChunkRule for doc_type
  ├─ Merges paragraphs to meet min_tokens
  ├─ Splits if exceeds max_tokens
  ├─ Adds overlap between chunks
  └─ Creates Chunk objects with metadata
```

**Output:**
- In-memory: `List[Chunk]`
- Disk: `input/chunks_<doc_type>.tsv`

**TSV Format:**
```
chunk_id<TAB>doc_id<TAB>text<TAB>token_count<TAB>meta_json
```

---

### 3. Unified Embedder

**Location:** `scripts/embeddings/unified_embedder.py`

**Purpose:** Generates vector embeddings and indexes them in FAISS.

**Key Features:**
- Multi-provider support (OpenAI, local models via LiteLLM)
- Async batch processing
- Deduplication by content_hash
- Per-doc-type indices

**Flow:**
```python
UnifiedEmbedder
  ├─ Groups chunks by doc_type
  ├─ Checks existing metadata for content_hash (dedup)
  ├─ Batches new chunks
  ├─ Calls embedding API (async)
  ├─ Creates/updates FAISS index per doc_type
  ├─ Appends to metadata JSONL
  └─ Saves FAISS index to disk
```

**Output:**
- `output/faiss/<doc_type>.faiss` - FAISS IndexFlatL2 (3072-dim)
- `output/metadata/<doc_type>_metadata.jsonl` - One line per chunk

---

### 4. Retrieval Manager

**Location:** `scripts/retrieval/retrieval_manager.py`

**Purpose:** Searches FAISS indices and returns relevant chunks.

**Key Features:**
- Late fusion (combines text + image results)
- Multi-doc-type search
- Similarity scoring (L2 distance → cosine)
- Metadata enrichment

**Flow:**
```python
RetrievalManager
  ├─ Embeds query (same model as chunks)
  ├─ Searches each doc_type FAISS index
  ├─ Retrieves top-K from each index
  ├─ Combines results (late fusion)
  ├─ Loads full metadata from JSONL
  ├─ Computes similarity scores
  └─ Returns sorted List[Chunk]
```

**Output:** `List[Chunk]` with `meta["similarity"]`

---

### 5. Prompt Builder

**Location:** `scripts/prompting/prompt_builder.py`

**Purpose:** Builds prompts for LLM with retrieved chunks as context.

**Key Features (v1.1.0):**
- Multi-template support (default, v2, email)
- **Strategy-based selection** (NEW)
- Auto-detection (email vs document chunks)
- Language detection (Hebrew/English)
- Email-specific formatting (sender, subject, date)

**Prompt Strategies:**
```yaml
prompt_strategy: auto    # Auto-detect (>50% emails → email template)
prompt_strategy: email   # Force email template
prompt_strategy: default # Force default template
prompt_strategy: v2      # Force v2 (structured, step-by-step)
```

**Flow:**
```python
PromptBuilder(config=config)  # Loads strategy from config
  ├─ Determines strategy (explicit or from config)
  ├─ Formats chunks (email or document style)
  ├─ Selects template based on strategy
  ├─ Inserts context + query
  └─ Returns formatted prompt
```

---

## Data Flow

### Complete Pipeline Flow

```
1. INGEST
   Raw files → RawDoc objects (in memory)

2. CHUNK
   RawDocs → Chunk objects (in memory)
           → chunks_<doc_type>.tsv (disk)

3. EMBED
   Chunks → Embeddings (via API)
         → FAISS indices (disk)
         → metadata JSONL (disk)

4. RETRIEVE
   Query → Query embedding
        → FAISS search
        → Retrieved chunks (in memory)

5. ASK
   Query + Chunks → Prompt
                  → LLM API
                  → Answer
```

### File Organization

```
data/projects/<project_name>/
│
├── input/
│   ├── raw/                          # Original files
│   │   ├── pdf/
│   │   ├── docx/
│   │   ├── eml/
│   │   └── outlook_eml/
│   │       └── emails.outlook_eml    # JSONL format (multi-doc)
│   │
│   ├── chunks_pdf.tsv                # Intermediate chunks
│   ├── chunks_docx.tsv
│   ├── chunks_outlook_eml.tsv
│   │
│   └── cache/images/                 # Extracted images
│
├── output/
│   ├── faiss/                        # Vector indices
│   │   ├── pdf.faiss
│   │   ├── docx.faiss
│   │   └── outlook_eml.faiss
│   │
│   ├── metadata/                     # Chunk metadata
│   │   ├── pdf_metadata.jsonl
│   │   ├── docx_metadata.jsonl
│   │   └── outlook_eml_metadata.jsonl
│   │
│   ├── .last_config.yml              # Validator state (v1.1.0)
│   └── .last_metadata.json           # Validator metadata (v1.1.0)
│
├── logs/
│   ├── app/                          # Subsystem logs
│   │   ├── app.log
│   │   ├── retrieval.log
│   │   └── prompt.log
│   │
│   └── runs/                         # Per-query logs
│       └── <run_id>/
│           ├── app.log
│           ├── prompt.txt
│           ├── response.txt
│           ├── chunks.jsonl
│           └── metadata.json
│
└── config.yml                        # Project configuration
```

---

## Email Agentic Strategy

**Location:** `scripts/agents/email_orchestrator.py`, `scripts/agents/email_strategy_selector.py`

**Purpose:** Intelligent query routing based on intent detection.

### Intent Types

```python
IntentType:
  - factual_lookup      # Simple fact finding
  - sender_query        # Find emails from specific people
  - temporal_query      # Time-based queries
  - thread_summary      # Conversation threads
  - multi_aspect        # Complex multi-part
  - action_decision     # Tasks and decisions
  - aggregation_query   # Topic summaries
```

### Specialized Retrievers

**1. SenderRetriever**
- Matches sender name/email
- Uses fuzzy matching
- Boosts chunks from target sender

**2. TemporalRetriever**
- Parses date expressions ("last week", "in June")
- Filters by date range
- Sorts by recency

**3. ThreadRetriever**
- Identifies email threads (subject matching)
- Returns conversation context
- Preserves chronological order

**4. MultiAspectRetriever**
- Combines multiple strategies
- Weighted scoring
- Handles complex queries

### Dynamic Top-K Adjustment

```python
TOP_K_ADJUSTMENTS = {
    'factual_lookup': 10,      # Default
    'sender_query': 12,         # Need more to find all from sender
    'temporal_query': 15,       # Recent may have many results
    'thread_summary': 20,       # Full conversation context
    'multi_aspect': 15,         # Complex needs broader coverage
    'action_decision': 12,      # Find all relevant tasks
    'aggregation_query': 20,    # Summary needs comprehensive view
}
```

### Flow

```
Query → EmailStrategySelector
          ├─ Detects intents (LLM-based)
          ├─ Selects retriever(s)
          └─ Adjusts top-K

       → EmailOrchestratorAgent
          ├─ Executes retrieval strategy
          ├─ Combines results
          └─ Returns chunks

       → PromptBuilder
          ├─ Uses email template (if applicable)
          └─ Formats email metadata

       → LLM → Answer
```

---

## Production Features (v1.1.0)

### 1. Pipeline Validator

**Location:** `scripts/pipeline/validator.py`

**Purpose:** Pre-flight checks before pipeline execution to prevent data corruption.

**Validations:**
- **Embedding dimension mismatch** (ERROR - blocks execution)
- **Chunking rule changes** (WARNING - allows with recommendation)
- **Data additions** (INFO - confirms deduplication will work)
- **Data removals** (WARNING - old data remains)
- **Model changes** (INFO or WARNING depending on impact)

**State Tracking:**
```
output/.last_config.yml       # Config snapshot
output/.last_metadata.json    # File counts, timestamps
```

**Integration:**
```python
# In PipelineRunner.run_steps()
validator = PipelineValidator(project)
report = validator.validate()

if not validator.print_report(report):
    # Blocked by errors
    return

# Proceed with pipeline...

# After success:
validator.save_current_state()
```

---

### 2. Smart Disk Fallback

**Location:** `scripts/pipeline/runner.py`

**Purpose:** Enables independent step execution by auto-loading from disk.

**Helper Methods:**
```python
_load_chunks_from_disk() → List[Chunk]
_count_raw_files() → int
_has_faiss_indices() → bool
```

**Modified Steps:**

**step_chunk():**
```python
if not self.raw_docs:
    if self._count_raw_files() > 0:
        # Auto-run ingest
        yield from self.step_ingest()
    else:
        # Show clear error
        yield "❌ No raw documents available."
        yield "   Options: ..."
        return
```

**step_embed():**
```python
if not chunk_files:
    if self._count_raw_files() > 0:
        # Auto-run ingest + chunk
        yield from self.step_ingest()
        yield from self.step_chunk()
    else:
        yield "❌ No chunks available."
        return
```

**step_retrieve():**
```python
if not self._has_faiss_indices():
    if chunk_files:
        # Auto-run embed
        yield from self.step_embed()
    elif self._count_raw_files() > 0:
        # Auto-run full pipeline
        yield from self.step_ingest()
        yield from self.step_chunk()
        yield from self.step_embed()
    else:
        yield "❌ No FAISS indices available."
        return
```

**Benefits:**
- Run steps independently in UI
- Survives UI refreshes (data persists on disk)
- Clear error messages with options
- Automatic recovery

---

### 3. Config Respect

**Location:** `scripts/prompting/prompt_builder.py`

**Purpose:** Respect `prompt_strategy` setting in config.

**Before v1.1.0:**
```python
# ALWAYS auto-detected, ignored config
if email_chunk_count > total / 2:
    use_email_template()
```

**After v1.1.0:**
```python
# Read from config
strategy = config.get('llm', {}).get('prompt_strategy', 'auto')

if strategy == 'auto':
    # Auto-detect
    if email_chunk_count > total / 2:
        use_email_template()
elif strategy == 'email':
    # Force email template
    use_email_template()
elif strategy in ['default', 'v2']:
    # Use specified template
    use_template(strategy)
```

**Configuration:**
```yaml
llm:
  prompt_strategy: email  # Will be respected!
```

---

## Design Decisions

### Why Per-Doc-Type FAISS Indices?

**Decision:** Separate FAISS index for each document type (pdf.faiss, outlook_eml.faiss, etc.)

**Rationale:**
- Different doc types may need different processing
- Easier to debug/inspect
- Can optimize per doc type
- Supports doc-type specific features

**Trade-off:**
- Slightly more complex retrieval (search multiple indices)
- More files to manage
- **Benefit outweighs cost** for flexibility

---

### Why TSV for Chunks (Not JSON)?

**Decision:** Store intermediate chunks in TSV format

**Rationale:**
- Human-readable (can inspect with less/cat)
- Simple parsing (no nested structures needed)
- Compact (no JSON overhead)
- Fast to read/write

**Alternative considered:** JSONL
- More flexible but overkill for simple tabular data

---

### Why Async Batch Embedding?

**Decision:** Use async batch API for embedding

**Rationale:**
- 10x faster than sequential
- Lower API costs (batching reduces overhead)
- Better resource utilization

**Implementation:**
```python
# Batch 64 chunks at a time
async with aiohttp.ClientSession() as session:
    tasks = [
        embed_batch(session, batch)
        for batch in chunks[::64]
    ]
    results = await asyncio.gather(*tasks)
```

---

### Why Deduplication by Content Hash?

**Decision:** Use SHA256(text + image_paths) for deduplication

**Rationale:**
- Exact match detection (no false positives)
- Fast lookup (hash comparison)
- Works across re-runs
- Saves API costs

**Alternative considered:** Fuzzy matching
- Too slow, too many false positives
- Content hash is deterministic and reliable

---

### Why LLM-Based Intent Detection?

**Decision:** Use LLM to detect query intent (v1.1.0)

**Rationale:**
- More flexible than regex patterns
- Understands natural language
- Can detect multiple intents
- Evolves with query complexity

**Trade-off:**
- Adds API call (small cost)
- Slight latency (~200ms)
- **Benefit:** Much better retrieval quality

**Alternative considered:** Rule-based
- Too rigid, misses edge cases
- Requires constant maintenance

---

### Why Multi-Template Prompt System?

**Decision:** Support multiple prompt templates with strategy selection (v1.1.0)

**Rationale:**
- Email queries need different formatting (sender, subject, date)
- Training docs need structured format (step-by-step)
- General docs need simple format
- Users need control (not just auto-detection)

**Implementation:**
- `default`: Simple, straightforward
- `v2`: Structured, step-by-step for training
- `email`: Rich email metadata

**Config:**
```yaml
prompt_strategy: auto | email | default | v2
```

---

## Cross-Platform Considerations

### WSL2 + Outlook Integration

**Challenge:** Outlook runs on Windows, but app runs in WSL2.

**Solution:**
```
WSL2 (RAG App)
     ↓ (Python script)
Windows (Helper)
     ↓ (COM interface)
Outlook (Desktop App)
```

**Implementation:**
- `outlook_wsl_client.py` - Spawns Windows Python process
- `outlook_helper_utils.py` - COM automation on Windows
- Communication via JSON files

---

## Security & Privacy

**Data Storage:**
- All documents stored locally (no cloud storage)
- FAISS indices on disk (not transmitted)
- Only chunk text sent to OpenAI API (for embedding/LLM)

**API Keys:**
- Stored in `.env` file (not in git)
- Never logged
- Passed via environment variables

**Logs:**
- Sanitized (no API keys or sensitive data)
- Structured JSON (easy to parse/audit)

---

## Performance Characteristics

**Embedding:**
- 100 chunks: ~10 seconds (async batch)
- 1,000 chunks: ~60 seconds
- 10,000 chunks: ~10 minutes

**Retrieval:**
- FAISS search: <100ms for millions of vectors
- Metadata lookup: ~50ms per chunk
- Total: ~200-500ms

**Query:**
- Retrieval: ~500ms
- LLM: 2-5 seconds
- Total: 2.5-5.5 seconds

---

## Future Enhancements

**Planned:**
- Hierarchical categorization (broad → specific topics)
- Real-time email sync (automated updates)
- Cross-project search
- Web-based deployment (Docker)
- Enhanced UI (cleaner, more intuitive)

**Under Consideration:**
- Local LLM support (Ollama)
- Advanced caching (reduce API calls)
- Distributed FAISS (scale to 100M+ documents)
- Graph-based retrieval (relationship awareness)

---

**For more details:**
- Implementation: See source code in `scripts/`
- Usage: See [USER_GUIDE.md](USER_GUIDE.md)
- Deployment: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
