# Frequently Asked Questions (FAQ)

**Version:** 1.1.0
**Last Updated:** 2025-11-23

---

## General Questions

### What is a RAG system?

**RAG** = **Retrieval-Augmented Generation**

It's a technique that combines:
1. **Retrieval:** Finding relevant information from your documents
2. **Generation:** Using an LLM (like GPT-4) to generate answers based on retrieved information

**Why it's better than just using ChatGPT:**
- Searches YOUR documents, not the internet
- Cites sources (tells you where information came from)
- Works with private/confidential documents
- More accurate for domain-specific questions

---

### Who is this for?

**Perfect for:**
- IT teams managing technical documentation
- Library systems administrators
- Support teams with large email histories
- Researchers working with document collections
- Teams needing centralized knowledge search

**Not ideal for:**
- Simple file search (use grep/find instead)
- Real-time collaboration (use Google Docs/SharePoint)
- One-off questions (just use ChatGPT directly)

---

### How is this different from ChatGPT?

| Feature | ChatGPT | This RAG System |
|---------|---------|-----------------|
| **Knowledge source** | Internet (up to cutoff date) | YOUR documents only |
| **Privacy** | Data sent to OpenAI | Documents stay local |
| **Citations** | No sources | Shows exact sources |
| **Domain knowledge** | General | Your specific content |
| **Cost** | $20/month unlimited | Pay-per-use (~$0.01/query) |

**Best use case:** When you need answers from specific documents with source citations.

---

## Setup & Installation

### What are the system requirements?

**Minimum:**
- Python 3.10+
- 8GB RAM
- 5GB disk space
- Internet connection (for API calls)

**Recommended:**
- Python 3.11+
- 16GB RAM
- 50GB+ disk space (for large document collections)
- OpenAI API key

---

### How do I get an OpenAI API key?

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-proj-...`)
5. Add to `.env` file:
   ```
   OPENAI_API_KEY=sk-proj-your-key-here
   ```

**Cost:** Pay-as-you-go
- Embedding: ~$0.13 per 1M tokens
- LLM: ~$0.15-$0.60 per 1M tokens (gpt-4o-mini)
- Typical: $0.10-0.20 per 100 documents, $0.01 per query

---

### Do I need a GPU?

**No!** This system doesn't require a GPU.

- Embeddings use OpenAI API (runs on their servers)
- LLM calls use OpenAI API (runs on their servers)
- FAISS (vector search) runs on CPU (very fast)

**Alternative:** You CAN use local embedding models (sentence-transformers) if you want to avoid API costs, but this requires more setup.

---

### Can I run this offline?

**Partially:**
- Once documents are embedded, you can search offline
- But queries need internet (LLM API calls)

**Fully offline option:**
- Use local embedding model (sentence-transformers)
- Use local LLM (Ollama, LM Studio)
- Requires significant setup and lower quality results

---

## Usage Questions

### What file formats are supported?

**Documents:**
- PDF (`.pdf`)
- Word (`.docx`)
- PowerPoint (`.pptx`)
- Excel (`.xlsx`)
- CSV (`.csv`)
- Plain text (`.txt`)

**Emails:**
- EML (`.eml`)
- MSG (`.msg`)
- MBOX (`.mbox`)
- Outlook (via connector)

---

### How many documents can I process?

**Limits:**
- **Technical:** No hard limit
- **Practical:** Depends on RAM and disk space

**Guidelines:**
- 100 documents: Works great on 8GB RAM
- 1,000 documents: 16GB RAM recommended
- 10,000+ documents: 32GB RAM, consider distributed setup

**Cost scales linearly:** 10x documents ≈ 10x cost

---

### How long does it take to process documents?

**First-time processing:**
- 10 documents: ~30 seconds
- 100 documents: ~2-3 minutes
- 1,000 documents: ~20-30 minutes

**Re-processing (with deduplication):**
- Only new/changed documents are processed
- Usually 10-20% of initial time

**Query time:**
- Simple query: 2-5 seconds
- Complex query: 5-10 seconds

---

### Can I query in languages other than English?

**Yes!** The system supports:

**Hebrew:** Full support (prompts detect language automatically)
```
שאלה: מהם הממצאים העיקריים?
תשובה: (in Hebrew)
```

**Other languages:** Works but not optimized
- GPT-4o supports 50+ languages
- Embeddings work cross-lingually
- May need to add language hint to query

---

### How accurate are the answers?

**Accuracy depends on:**

1. **Document quality** (clear, well-written → better results)
2. **Embedding model** (text-embedding-3-large recommended)
3. **Chunk size** (default 300 tokens works well)
4. **Query phrasing** (specific queries → better results)

**Typical accuracy:**
- Factual lookups: 90-95% accurate
- Complex reasoning: 80-85% accurate
- Ambiguous queries: 60-70% accurate

**Important:** System can only answer based on document content. It won't invent information.

---

## Features & Capabilities

### What is "intent detection"? (v1.1.0)

**Intent detection** automatically determines what kind of query you're asking and uses the best retrieval strategy.

**Example:**
```
Query: "What did John discuss last week about the API?"

Detected intents:
- temporal_query (last week)
- sender_query (John)

Actions:
- Filters emails from last 7 days
- Looks for sender "John"
- Retrieves 15 chunks (instead of default 10)
```

**You don't need to do anything special** - just ask naturally!

---

### What is the "validator"? (v1.1.0)

**Pipeline Validator** is a safety feature that checks for problems before running the pipeline.

**It detects:**
- Embedding dimension mismatches (would break FAISS)
- Chunk rule changes (would create mixed chunk sizes)
- Data additions/removals
- Model changes

**Example:**
```
You: Change embedding model and try to embed

Validator: ❌ BLOCKED
  Reason: Dimension mismatch (3072d → 1536d)
  Fix: Delete output/faiss/ and output/metadata/
```

**Benefits:** Prevents disasters, saves time and money.

---

### What is "smart fallback"? (v1.1.0)

**Smart fallback** lets you run pipeline steps independently.

**Before v1.1.0:**
```
You: Run "Chunk" step only
System: ❌ No raw documents. Run 'ingest' first.
(You have to remember to run steps in order)
```

**With v1.1.0:**
```
You: Run "Chunk" step only
System: ⚠️  No raw docs in memory, found 150 on disk
        Auto-running 'ingest' first...
        ✅ Ingested 150 documents
        ✅ Chunking complete
```

**Benefits:** Survives UI refreshes, more flexible, better UX.

---

### Can I search multiple projects at once?

**No, not currently.** Each query searches one project.

**Workaround:**
1. Create a "master" project
2. Copy documents from multiple projects into master
3. Query the master project

**Future feature:** Cross-project search is on the roadmap.

---

## Email-Specific Questions

### Can I connect to Outlook?

**Yes!** But only on **Windows** (with WSL2 for the app itself).

**Requirements:**
- Windows 10/11
- Outlook desktop app installed and configured
- WSL2 with the RAG platform installed

**How it works:**
1. UI runs in WSL2
2. Helper script connects to Windows Outlook via COM
3. Extracts emails and saves to project
4. Then you can search emails like any other documents

**See:** Outlook integration guide in [USER_GUIDE.md](USER_GUIDE.md#workflow-2-ingesting-outlook-emails)

---

### Can I search by email sender?

**Yes!** Use sender-based queries:

```
"Show me emails from john.smith@example.com"
"What did Jane discuss about the bug?"
"Find emails from the support team"
```

**The system automatically:**
- Detects sender intent
- Matches sender name/email
- Retrieves relevant emails
- Highlights sender in results

---

### Can I search by date/time?

**Yes!** Use temporal queries:

```
"What was discussed last week?"
"Show me emails from June 2025"
"Recent discussions about the API"
"What happened yesterday?"
```

**The system automatically:**
- Detects temporal intent
- Parses date expressions
- Filters by date range
- Adjusts top-K (retrieves more results for time-based queries)

---

### Can I find action items in emails?

**Yes!** Use action/decision queries:

```
"What are my action items?"
"What decisions were made last month?"
"Find tasks assigned to me"
```

**The system uses LLM to extract:**
- Action items (todos, tasks)
- Decisions (approvals, conclusions)
- Deadlines
- Assignments

---

## Technical Questions

### What is FAISS?

**FAISS** = **Facebook AI Similarity Search**

It's a library for fast vector similarity search.

**In this system:**
- Your document chunks → converted to vectors (embeddings)
- Vectors stored in FAISS index
- Query → converted to vector
- FAISS finds most similar chunk vectors
- Results returned in milliseconds

**Why FAISS?**
- Very fast (searches millions of vectors in <100ms)
- Memory efficient
- Production-tested (used by Facebook/Meta)

---

### What embedding model should I use?

**Recommended:** `text-embedding-3-large`
- 3072 dimensions
- Best quality
- Same cost as ada-002 ($0.13/1M tokens)

**Budget option:** `text-embedding-3-small`
- 1536 dimensions
- Good quality
- Half the cost

**Don't use:** `text-embedding-ada-002`
- Older model
- Lower quality
- Same cost as 3-large

---

### What LLM model should I use?

**For most users:** `gpt-4o-mini`
- Fast
- Cheap ($0.15/1M input tokens)
- Good quality

**For best quality:** `gpt-4o`
- Highest quality
- More expensive ($2.50/1M input tokens)
- Slower

**For testing:** `gpt-3.5-turbo`
- Cheapest
- Lower quality
- Fast

---

### How is my data stored?

**Documents:**
- Original files: `input/raw/` (preserved)
- Chunks: `input/chunks_*.tsv` (TSV format)
- Embeddings: `output/faiss/*.faiss` (binary FAISS format)
- Metadata: `output/metadata/*_metadata.jsonl` (JSONL format)

**Security:**
- All data stored locally on your machine
- Only text sent to OpenAI API (for embedding/LLM)
- No document storage on OpenAI servers

**Backup:**
- Back up `data/projects/<project>/` directory
- Contains all data needed to recreate project

---

### Can I use a different LLM provider?

**Yes!** The system supports multiple providers via LiteLLM.

**Supported:**
- OpenAI (default)
- Anthropic (Claude)
- Azure OpenAI
- Google (PaLM)
- Local models (Ollama, LM Studio)

**Configuration:**
```yaml
# config.yml
llm:
  provider: litellm
  model: claude-3-5-sonnet  # or gpt-4o, etc.
```

**See:** LiteLLM docs for full provider list

---

### What is "deduplication"?

**Deduplication** prevents re-processing unchanged content.

**How it works:**
1. Each chunk gets a `content_hash` (SHA256)
2. Before embedding, system checks if hash already exists
3. If exists → skip (reuse existing embedding)
4. If new → embed and add to index

**Benefits:**
- Saves API costs
- Faster re-processing
- Automatic (no configuration needed)

**Example:**
```
100 documents processed
→ 500 chunks created
→ 50 chunks are duplicates (same content)
→ Only 450 chunks embedded
→ Save 10% on API costs
```

---

## Cost & Performance

### How much does it cost?

**Setup (one-time):**
- 100 documents: $0.10 - $0.20
- 1,000 documents: $1.00 - $2.00
- 10,000 documents: $10.00 - $20.00

**Queries:**
- Simple query: $0.005 - $0.01
- Complex query: $0.01 - $0.02

**Monthly (typical usage):**
- Small team (100 queries/month): $1-2
- Medium team (1000 queries/month): $10-20
- Heavy usage (10,000 queries/month): $100-200

**Compared to:**
- ChatGPT Plus: $20/month (but no source citations or private docs)
- Enterprise RAG: $500-5000/month (but more features)

---

### Can I reduce costs?

**Yes! Several ways:**

**1. Use deduplication** (default)
```yaml
embedding:
  skip_duplicates: true
```

**2. Use cheaper models**
```yaml
llm:
  model: gpt-4o-mini  # vs gpt-4o
```

**3. Reduce top-K**
```
# Retrieve fewer chunks
top_k: 5  # vs 10
```

**4. Batch operations**
```yaml
embedding:
  use_async_batch: true
  embed_batch_size: 64
```

**5. Use local models** (advanced)
- Local embeddings: sentence-transformers
- Local LLM: Ollama
- Zero API costs, but lower quality

---

### Why is the first run slow?

**First run downloads models/libraries:**
- Sentence transformers (if using local embeddings)
- NLTK data (for text processing)
- Can take 2-5 minutes

**Subsequent runs are fast:**
- Models cached locally
- Only new data processed
- Deduplication skips unchanged content

---

## Troubleshooting

### Where do I find error logs?

**Application logs:**
```
data/projects/<project_name>/logs/app/
```

**Per-run logs:**
```
data/projects/<project_name>/logs/runs/<run_id>/
```

**See:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed diagnostics

---

### Why can't I find specific information?

**Common reasons:**

1. **Information not in documents** (most common)
2. Query too vague
3. Query has typos
4. Documents not embedded yet
5. Wrong project selected

**Try:**
- Rephrase query more specifically
- Check documents actually contain the information
- Verify pipeline ran successfully
- Check similarity scores (should be >0.7)

---

### How do I reset a project?

**Full reset:**
```bash
# Delete everything except source files
rm -r data/projects/MyProject/input/chunks_*.tsv
rm -r data/projects/MyProject/output/
rm -r data/projects/MyProject/logs/

# Re-run pipeline
# Ingest → Chunk → Embed
```

**Keep embeddings, reset queries:**
```bash
# Just delete logs
rm -r data/projects/MyProject/logs/runs/
```

---

## Still Have Questions?

**Documentation:**
- [USER_GUIDE.md](USER_GUIDE.md) - Comprehensive guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Installation help

**Support:**
- GitHub Issues: Report bugs, request features
- Check existing issues first: Someone may have already solved your problem

---

**Not answered here?** [Create a GitHub issue](https://github.com/your-repo/issues/new) with your question!
