# Multi-Source RAG Platform - User Guide

**Version:** 1.1.0
**Last Updated:** 2025-11-23
**For:** End users and system administrators

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Common Workflows](#common-workflows)
5. [Understanding Results](#understanding-results)
6. [Configuration Guide](#configuration-guide)
7. [Best Practices](#best-practices)
8. [Advanced Features](#advanced-features)

---

## Introduction

### What is the Multi-Source RAG Platform?

A **Retrieval-Augmented Generation (RAG)** system that helps you search and ask questions across multiple document sources using natural language. Think of it as a smart search assistant that can:

- Search through PDFs, Word documents, PowerPoint presentations, Excel spreadsheets, and emails
- Answer questions using the content from your documents
- Cite sources so you know where the information came from
- Handle both simple lookups and complex multi-document queries

### Who is this for?

- **Library systems administrators** managing technical documentation
- **IT professionals** searching through support emails and documentation
- **Researchers** working with large document collections
- **Teams** needing centralized knowledge search

### Key Features

✅ **Multi-format support:** PDF, DOCX, PPTX, XLSX, CSV, TXT, emails (EML, MSG, MBOX, Outlook)
✅ **Natural language queries:** Ask questions in plain English/Hebrew
✅ **Smart retrieval:** Intent-based strategies for better results
✅ **Email-specific features:** Temporal queries, sender-based search, thread analysis
✅ **Production-ready:** Validation, smart recovery, configurable behavior

---

## Prerequisites

### System Requirements

- **Operating System:** Linux, macOS, or Windows (with WSL2 for Outlook integration)
- **Python:** 3.10 or higher
- **Memory:** 8GB RAM minimum (16GB recommended for large datasets)
- **Disk Space:** 5GB+ (depends on document collection size)
- **Internet:** Required for OpenAI API calls (embedding and LLM)

### Required Accounts

1. **OpenAI API Key** - For embeddings and language model
   - Get one at: https://platform.openai.com/api-keys
   - Recommended models:
     - Embedding: `text-embedding-3-large` (3072 dimensions)
     - LLM: `gpt-4o-mini` or `gpt-4o`

2. **Microsoft Account** (Optional) - For Outlook email integration
   - Only needed if ingesting emails from Outlook

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd Multi-Source_RAG_Platform

# 2. Install with Poetry (recommended)
poetry install
poetry shell

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

For detailed installation instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

---

## Quick Start

### Your First Query in 5 Minutes

**1. Create a Project**

```bash
# Using Streamlit UI (recommended)
streamlit run scripts/ui/ui_v3.py

# Or using CLI
poetry run python -m app.cli create-project \
  --name "My First Project" \
  --description "Testing the RAG system"
```

**2. Add Documents**

Place your documents in the project's input directory:
```
data/projects/My_First_Project/input/raw/
├── pdf/
│   └── document1.pdf
├── docx/
│   └── report.docx
└── eml/
    └── email1.eml
```

**3. Run the Pipeline**

In the UI:
1. Go to "Pipeline" tab
2. Select steps: `Ingest` → `Chunk` → `Embed`
3. Click "Run Pipeline"
4. Wait for completion (shows progress)

**4. Ask a Question**

In the "Query" tab:
1. Enter your question: *"What are the main findings?"*
2. Click "Submit"
3. View the answer with source citations

**That's it!** You've successfully queried your documents.

---

## Common Workflows

### Workflow 1: Adding Documents to Existing Project

**Scenario:** You already have a project and want to add more documents.

**Steps:**

1. **Add files** to `data/projects/<project_name>/input/raw/`
2. **Run pipeline steps:**
   - `Ingest` → processes new files
   - `Chunk` → creates searchable chunks
   - `Embed` → generates embeddings and indexes them
3. **Validation happens automatically** ✨
   - System detects new files
   - Shows: *"Added 50 new file(s)"*
   - Deduplication skips unchanged content
   - Only new chunks get embedded

**Important:** The new **Pipeline Validator** (v1.1.0) automatically checks for issues:
- ✅ Detects config changes
- ✅ Warns about potential problems
- ✅ Blocks execution if critical errors found
- ✅ Shows exactly what will happen

### Workflow 2: Ingesting Outlook Emails

**Scenario:** Search through your Outlook emails (Windows + WSL2 required).

**Steps:**

1. **Open UI** and go to "Data Sources" tab
2. **Configure Outlook:**
   - Account: `your.email@example.com`
   - Folder path: `Inbox > Project Emails`
   - Days back: `30`
   - Max emails: `500` (or leave blank for all)
3. **Click "Extract Emails"**
   - System connects to Outlook (via Windows COM)
   - Extracts emails and saves to project
   - Shows progress: *"Extracted 237 emails"*
4. **Run pipeline:**
   - `Ingest` → loads email file
   - `Chunk` → creates email chunks with metadata
   - `Embed` → indexes emails

**Email-Specific Features:**
- Sender-based queries: *"Show me emails from John Smith"*
- Temporal queries: *"What did we discuss last week?"*
- Thread analysis: Finds related conversation chains
- Action/decision extraction: Identifies tasks and decisions

### Workflow 3: Running Steps Independently

**Scenario:** You only want to re-run chunking after changing chunk rules.

**New in v1.1.0:** Smart disk fallback enables independent step execution! 🎉

**Steps:**

1. **Modify chunking rules** in `configs/chunk_rules.yaml`
2. **Run only "Chunk" step** in UI
3. **System auto-recovers:**
   ```
   ⚠️  No raw documents in memory, but found 150 raw file(s) on disk
       Running 'ingest' step first...
   ✅ Ingested 150 documents
   📚 Starting chunking...
   ✅ Chunking complete. Total chunks: 523
   ```

**Benefits:**
- No need to run full pipeline every time
- Survives UI refreshes
- Clear error messages with options
- Smart recovery from disk

### Workflow 4: Changing Configuration

**Scenario:** You want to change the prompt template or embedding model.

**⚠️ Important:** The new **Pipeline Validator** will check for issues!

**Safe Changes:**
```yaml
# config.yml

llm:
  model: gpt-4o              # Can change freely
  temperature: 0.4           # Can adjust
  max_tokens: 400           # Can adjust
  prompt_strategy: email    # NEW in v1.1.0 - actually works now!
```

**Changes That Require Cleanup:**
```yaml
embedding:
  model: text-embedding-ada-002  # ⚠️ Different dimensions!
```

**What happens:**
```
❌ CRITICAL ERRORS - Cannot proceed:

  [EMBEDDING] Embedding model dimension mismatch
    Old: text-embedding-3-large (3072d)
    New: text-embedding-ada-002 (1536d)
    Impact: FAISS indices are incompatible with new model
    Fix: Delete existing indices and metadata:
      rm -r output/faiss/
      rm -r output/metadata/

================================================================================
❌ Pipeline execution BLOCKED
================================================================================
```

**The validator prevents disasters!** Follow the recommended fix before proceeding.

### Workflow 5: Querying Your Documents

**Simple Queries:**
```
"What is the deadline for the project?"
"Who is responsible for the API integration?"
"What are the system requirements?"
```

**Email-Specific Queries:**
```
"What topics were discussed last week?"
"Show me emails from Jane about the bug reports"
"What decisions were made in the last month?"
"Find action items from recent emails"
```

**Multi-Document Queries:**
```
"Summarize the key findings across all reports"
"What are the common issues mentioned in bug reports?"
"Compare the approaches mentioned in different documents"
```

---

## Understanding Results

### Query Response Format

When you submit a query, you receive:

**1. Retrieved Chunks**
```
Retrieved 10 chunks:
  [1] Source: report.pdf (Page 3) - Score: 0.892
  [2] Source: email_from_john.eml (2025-11-15) - Score: 0.867
  [3] Source: presentation.pptx (Slide 5) - Score: 0.845
  ...
```

**2. Generated Answer**
```
Based on the documents, the project deadline is December 15, 2025 [report.pdf].
The API integration is assigned to John Smith [email_from_john.eml].
```

**3. Source Citations**
- Citations in `[brackets]` link to source documents
- Check the retrieved chunks for full context
- Similarity scores show relevance (0.0 - 1.0, higher is better)

### Email-Specific Formatting

For email queries, results include rich metadata:

```
Email #1:
From: John Smith <john.smith@example.com>
Subject: API Integration Update
Date: 2025-11-15

Content:
The API integration is progressing well. Expected completion by November 30th.
Action items:
- Complete authentication module by Nov 20
- Deploy to staging environment Nov 25
```

### Understanding Similarity Scores

- **0.9 - 1.0:** Highly relevant (exact match or very close)
- **0.8 - 0.9:** Very relevant (strong match)
- **0.7 - 0.8:** Relevant (good match)
- **0.6 - 0.7:** Somewhat relevant (partial match)
- **< 0.6:** Weakly relevant (consider refining query)

---

## Configuration Guide

### Project Configuration (`config.yml`)

Located at: `data/projects/<project_name>/config.yml`

#### LLM Settings

```yaml
llm:
  model: gpt-4o-mini           # LLM model to use
  temperature: 0.4             # 0.0 = deterministic, 1.0 = creative
  max_tokens: 400              # Maximum response length
  prompt_strategy: auto        # NEW: 'auto', 'email', 'default', 'v2'
  provider: openai             # Provider (openai, litellm, etc.)
```

**Prompt Strategies (NEW in v1.1.0):**
- `auto` - Auto-detect based on content (default)
- `email` - Force email template (use for email-only projects)
- `default` - Force original default template
- `v2` - Force enhanced v2 template (structured, step-by-step)

**When to use each:**
- Email projects → `email` for consistent email formatting
- Training/documentation → `v2` for structured answers
- General purpose → `auto` for smart detection

#### Embedding Settings

```yaml
embedding:
  model: text-embedding-3-large  # Embedding model
  provider: litellm               # Provider
  mode: batch                     # 'batch' or 'streaming'
  use_async_batch: true           # Async batch processing
  skip_duplicates: true           # Skip already-embedded chunks
  embed_batch_size: 64            # Batch size for API calls
```

**⚠️ Warning:** Changing `model` with different dimensions breaks FAISS indices! Validator will catch this.

#### Chunking Rules

Located at: `configs/chunk_rules.yaml`

```yaml
pdf:
  strategy: by_paragraph
  min_tokens: 50
  max_tokens: 300
  overlap: 20

outlook_eml:
  strategy: by_email_block
  min_tokens: 20
  max_tokens: 300
  overlap: 5
```

**Changing chunk rules:** Validator will warn about mixed chunk sizes.

### Global Configuration (`.env`)

```bash
# API Keys
OPENAI_API_KEY=sk-...

# Optional: LiteLLM configuration
LITELLM_BASE_URL=https://api.openai.com/v1
```

---

## Best Practices

### 1. Document Organization

**✅ Do:**
- Organize files by type: `raw/pdf/`, `raw/docx/`, `raw/eml/`
- Use descriptive filenames
- Keep source files in `input/raw/` for reproducibility

**❌ Don't:**
- Mix unrelated documents in one project
- Delete source files after ingestion
- Use special characters in filenames

### 2. Query Optimization

**✅ Do:**
- Be specific: *"What are the API authentication requirements?"*
- Use temporal context for emails: *"last week"*, *"in June"*
- Ask one question at a time
- Mention document types if relevant: *"according to the manual"*

**❌ Don't:**
- Ask vague questions: *"Tell me about the project"*
- Combine multiple unrelated questions
- Expect answers not in your documents

### 3. Pipeline Execution

**✅ Do:**
- Run validation before large changes (automatic in v1.1.0)
- Use smart fallback (run steps independently)
- Monitor logs for warnings
- Save your config changes

**❌ Don't:**
- Skip validation (`skip_validation=True` unless debugging)
- Change embedding dimensions without cleanup
- Mix chunk sizes from different runs

### 4. Email Projects

**✅ Do:**
- Set `prompt_strategy: email` for email-only projects
- Use temporal queries for recent discussions
- Extract regularly (weekly/monthly) for freshness
- Filter by folder to reduce noise

**❌ Don't:**
- Ingest personal/private emails without permission
- Extract entire mailbox (use date limits)
- Forget to update after major email discussions

### 5. Cost Optimization

**✅ Do:**
- Use deduplication (enabled by default)
- Start with smaller models: `gpt-4o-mini` for testing
- Batch operations when possible
- Use `text-embedding-3-large` (better quality, same cost as ada-002)

**❌ Don't:**
- Re-embed unchanged content (deduplication handles this)
- Use `gpt-4o` for simple queries (overkill)
- Process same documents multiple times without cleanup

---

## Advanced Features

### 1. Email Agentic Strategy (v1.1.0)

**What it does:** Automatically detects query intent and uses specialized retrieval strategies.

**Intent Types:**
- `factual_lookup` - Simple fact finding
- `sender_query` - Find emails from specific people
- `temporal_query` - Recent discussions, time-based
- `thread_summary` - Conversation threads
- `multi_aspect` - Complex multi-part queries
- `action_decision` - Find tasks and decisions
- `aggregation_query` - Topic summaries (with categorization)

**How to use:** Just ask naturally! The system auto-detects intent.

**Example:**
```
Query: "What did John discuss last week about the API?"

Auto-detected intent: temporal_query + sender_query
Strategy: Combines temporal filter + sender match
Top-K: Adjusted to 15 (from default 10)
Result: Emails from John in the last 7 days about API
```

### 2. Dynamic Top-K Adjustment

**What it does:** System automatically adjusts how many chunks to retrieve based on query intent.

**Default Top-K:**
- `factual_lookup`: 10 chunks
- `sender_query`: 12 chunks
- `temporal_query`: 15 chunks
- `thread_summary`: 20 chunks
- `multi_aspect`: 15 chunks

**Benefits:** Better results without manual tuning.

### 3. LLM-Enhanced Features

**Action Item Extraction:**
```
Query: "What are my action items?"

System extracts:
- Complete authentication module by Nov 20
- Deploy to staging Nov 25
- Review API documentation
```

**Decision Extraction:**
```
Query: "What decisions were made?"

System identifies:
- Decided to use REST API (Nov 10)
- Approved budget increase (Nov 12)
- Postponed feature X to Phase 2 (Nov 15)
```

### 4. Email Categorization (Beta)

**Status:** Phase 1-2 complete, categories discovered

**What it does:** Auto-categorizes emails into topics using BERTopic + K-means clustering.

**Example Categories:**
- Research Support
- UI Customization
- Technical Issues
- Product Updates

**How to use:** Currently in testing phase. Will enable aggregation queries like:
```
"What topics were discussed in the last month?"
→ Shows summary by category
```

### 5. Image Support (Optional)

**What it does:** Extracts images from PDFs/PPTX and adds AI-generated descriptions.

**Configuration:**
```yaml
agents:
  enable_image_insight: true
  image_agent_model: gpt-4o
```

**Use case:** When visual content matters (charts, diagrams, screenshots).

---

## Troubleshooting

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

**Quick fixes:**

**"No raw documents available"**
→ Add files to `input/raw/` or let smart fallback auto-recover

**"FAISS indices incompatible"**
→ Validator detected dimension mismatch. Follow the fix command shown.

**"Chunk rules have changed"**
→ Warning only. Delete `input/chunks_*.tsv` if you want clean chunks.

**"No results for query"**
→ Try broader query, check if documents are embedded, verify spelling

---

## Getting Help

**Documentation:**
- Quick Start: [README_QUICKSTART.md](README_QUICKSTART.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Deployment: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- FAQ: [FAQ.md](FAQ.md)

**Support:**
- GitHub Issues: Report bugs and request features
- Check logs: `data/projects/<project>/logs/` for diagnostics

---

## What's Next?

**After your first successful query:**
1. Explore different query types (temporal, sender-based, aggregation)
2. Adjust configuration for your use case
3. Set up automated email sync (if using Outlook)
4. Optimize chunk rules for your document types

**Advanced users:**
5. Experiment with different prompt strategies
6. Fine-tune retrieval parameters
7. Integrate with your own applications via CLI/API

---

**Version History:**
- **v1.1.0** (2025-11-23): Production readiness (validator, smart fallback, config respect), Email agentic strategy
- **v1.0.0** (2025-11): Initial release with multi-format support, Outlook integration

**Happy querying!** 🚀
