# Troubleshooting Guide

**Version:** 1.1.0
**Last Updated:** 2025-11-23

This guide covers common issues and their solutions. Issues are organized by pipeline stage and severity.

---

## Table of Contents

1. [Pipeline Validation Errors](#pipeline-validation-errors)
2. [Pipeline Execution Issues](#pipeline-execution-issues)
3. [Smart Fallback Messages](#smart-fallback-messages)
4. [Query Issues](#query-issues)
5. [Outlook Integration Issues](#outlook-integration-issues)
6. [Performance Issues](#performance-issues)
7. [Configuration Issues](#configuration-issues)
8. [Diagnostic Steps](#diagnostic-steps)

---

## Pipeline Validation Errors

### ❌ Embedding Model Dimension Mismatch

**Error Message:**
```
❌ CRITICAL ERRORS - Cannot proceed:

  [EMBEDDING] Embedding model dimension mismatch
    Old: text-embedding-3-large (3072d)
    New: text-embedding-ada-002 (1536d)
    Impact: FAISS indices are incompatible with new model
    Fix: Delete existing indices and metadata:
      rm -r data/projects/<project>/output/faiss/
      rm -r data/projects/<project>/output/metadata/
```

**Cause:** You changed the embedding model to one with different dimensions. FAISS indices are incompatible.

**Solution:**
```bash
# 1. Back up your project (optional but recommended)
cp -r data/projects/MyProject data/projects/MyProject.backup

# 2. Delete indices and metadata
rm -r data/projects/MyProject/output/faiss/
rm -r data/projects/MyProject/output/metadata/

# 3. Run pipeline again (will re-embed all chunks)
# Ingest → Chunk → Embed
```

**Prevention:** Don't change embedding models mid-project, or be prepared to re-embed everything.

---

### ⚠️ Chunking Rules Have Changed

**Warning Message:**
```
⚠️  WARNINGS - Proceed with caution:

  [CHUNKING] Chunking rules have changed since last run
    Old: Last modified: 1732356789.123
    New: Current modified: 1732357890.456
    Impact: Will create chunks with different sizes, mixed with old chunks
    Recommendation: Delete existing chunks to avoid mixed sizes:
      rm data/projects/<project>/input/chunks_*.tsv
```

**Cause:** You modified `configs/chunk_rules.yaml` after already chunking documents.

**Impact:** New chunks will have different sizes than old chunks, potentially affecting search quality.

**Solution (Clean Start):**
```bash
# 1. Delete old chunks
rm data/projects/MyProject/input/chunks_*.tsv

# 2. Re-run chunking
# Chunk step will create fresh chunks with new rules

# 3. Re-embed (embeddings depend on chunks)
# Run Embed step
```

**Solution (Keep Old Chunks):**
- You can proceed if you're okay with mixed chunk sizes
- New documents will use new rules
- Old chunks remain unchanged
- Generally works fine, but not ideal

---

### ✓ Added New Raw Files

**Info Message:**
```
✓ INFORMATION:

  [DATA] Added 400 new raw file(s)
    Previous: 600 files
    Current: 1000 files
    Impact: New files will be ingested, chunked, and embedded
    Note: Deduplication will skip unchanged content (via content_hash).
          Only truly new chunks will be embedded.
```

**What this means:** Everything is fine! The system detected new files and will process only what's new.

**No action needed.** The pipeline will handle deduplication automatically.

---

### ⚠️ Removed Raw Files

**Warning Message:**
```
⚠️  WARNINGS:

  [DATA] Removed 30 raw file(s)
    Old: 100 files
    New: 70 files
    Impact: Old chunks/embeddings from removed files will remain
    Recommendation: Consider deleting old data if files were intentionally removed:
      rm data/projects/<project>/input/chunks_*.tsv
      rm -r data/projects/<project>/output/faiss/
      rm -r data/projects/<project>/output/metadata/
```

**Cause:** You deleted some source files from `input/raw/`.

**Impact:** The chunks and embeddings from those deleted files are still in the index.

**Solution (Clean Slate):**
```bash
# Delete all processed data
rm data/projects/MyProject/input/chunks_*.tsv
rm -r data/projects/MyProject/output/faiss/
rm -r data/projects/MyProject/output/metadata/

# Re-run pipeline
# Only remaining files will be processed
```

**Solution (Keep Existing Data):**
- If you're fine with old chunks remaining, you can proceed
- Old chunks won't cause errors, just outdated results

---

## Pipeline Execution Issues

### ❌ No Raw Documents Available

**Error Message:**
```
❌ No raw documents available.
   Options:
   1. Run 'ingest' step first
   2. Add files to data/projects/MyProject/input/raw/ directory
```

**Cause:** No files in `input/raw/` and `step_chunk()` was called directly.

**Solution:**
```bash
# Option 1: Add files
cp your-documents/* data/projects/MyProject/input/raw/pdf/

# Option 2: Run ingest first
# In UI: Select "Ingest" → "Run Pipeline"
```

**Note:** In v1.1.0, smart fallback should handle this automatically. If you see this, the fallback didn't find raw files.

---

### ⚠️ Auto-Recovery: Smart Fallback

**Message:**
```
⚠️  No raw documents in memory, but found 150 raw file(s) on disk
    Running 'ingest' step first...
🚀 Starting ingestion...
✅ Ingested 150 documents
```

**What this means:** This is the smart fallback feature working correctly! **No action needed.**

**What happened:**
1. You ran "Chunk" step independently
2. System found no raw docs in memory
3. System checked disk, found raw files
4. System auto-ran "Ingest" first
5. System proceeded with chunking

**This is a feature, not an error.**

---

### ❌ No FAISS Indices Available for Retrieval

**Error Message:**
```
❌ No FAISS indices available for retrieval.
   Options:
   1. Run 'ingest', 'chunk', and 'embed' steps first
   2. Add files to data/projects/MyProject/input/raw/ directory
```

**Cause:** Trying to query before running the pipeline.

**Solution:**
```bash
# Run full pipeline in order:
1. Ingest → processes raw files
2. Chunk → creates chunks
3. Embed → generates FAISS indices

# Then you can query
```

**Or let smart fallback handle it:** Just run "Embed" or "Retrieve" and the system will auto-run prerequisites.

---

### ❌ OpenAI API Error: Rate Limit

**Error Message:**
```
openai.RateLimitError: Rate limit exceeded
```

**Cause:** Too many API requests too quickly.

**Solutions:**

**1. Wait and Retry (easiest)**
```python
# The system has automatic retries built in
# Just wait 30-60 seconds and try again
```

**2. Reduce Batch Size**
```yaml
# config.yml
embedding:
  embed_batch_size: 32  # Reduce from 64
```

**3. Use Async Batch Mode** (default in v1.1.0)
```yaml
embedding:
  use_async_batch: true  # Default
  mode: batch
```

---

### ❌ OpenAI API Error: Invalid API Key

**Error Message:**
```
openai.AuthenticationError: Incorrect API key provided
```

**Cause:** API key is missing, invalid, or expired.

**Solution:**
```bash
# 1. Check .env file exists
ls .env

# 2. Check it contains your key
cat .env
# Should show: OPENAI_API_KEY=sk-proj-...

# 3. Get a valid key from OpenAI
# https://platform.openai.com/api-keys

# 4. Update .env
echo "OPENAI_API_KEY=sk-your-new-key" > .env

# 5. Restart application
```

---

## Smart Fallback Messages

These messages indicate the smart fallback system is working. **No action needed** unless you prefer to run steps manually.

### ⚠️ Running Prerequisites Automatically

**Chunk Step:**
```
⚠️  No raw documents in memory, but found 150 raw file(s) on disk
    Running 'ingest' step first...
```
**Meaning:** Auto-running ingest before chunking.

**Embed Step:**
```
⚠️  No chunk files found, but found 150 raw file(s) on disk
    Running 'ingest' and 'chunk' steps first...
```
**Meaning:** Auto-running ingest + chunk before embedding.

**Retrieve Step:**
```
⚠️  No FAISS indices found, but found chunk files
    Running 'embed' step first...
```
**Meaning:** Auto-running embed before retrieval.

**Full Pipeline Auto-Recovery:**
```
⚠️  No indices/chunks found, but found raw files
    Running full pipeline (ingest → chunk → embed)...
```
**Meaning:** Running entire pipeline from scratch.

**These are helpful automation features introduced in v1.1.0.**

---

## Query Issues

### No Results Found

**Symptom:** Query returns empty or very few results.

**Possible Causes & Solutions:**

**1. Documents Not Embedded**
```bash
# Check if FAISS indices exist
ls data/projects/MyProject/output/faiss/
# Should show *.faiss files

# If empty, run Embed step
```

**2. Query Too Specific**
```
# Too specific:
"What is the exact deadline for Phase 2.3 of the API integration project?"

# Better:
"What is the deadline for the API project?"
```

**3. Typos in Query**
```
# Check spelling
"deeadline" → "deadline"
"integraion" → "integration"
```

**4. Information Not in Documents**
```
# The RAG system can only find information that exists in your documents
# It won't invent or guess information
```

---

### Low Similarity Scores

**Symptom:** All results have scores < 0.6

**Meaning:** Query doesn't match well with any documents.

**Solutions:**

**1. Rephrase Query**
```
# Original: "Tell me about the thing John mentioned"
# Better: "What did John discuss about the API integration?"
```

**2. Check Document Content**
```bash
# Are your documents actually related to the query?
# Try a query you KNOW the answer is in the documents
```

**3. Verify Embeddings**
```bash
# Check embedding model in config.yml
# text-embedding-3-large is recommended
```

---

### Wrong Language in Answer

**Symptom:** Asked in Hebrew, got English answer (or vice versa).

**Cause:** All prompt templates have language detection, but it might fail for very short queries.

**Solution:**
```
# Add language hint to your query:
"(in Hebrew) מהם הממצאים העיקריים?"
"(in English) What are the main findings?"
```

---

### Email-Specific Query Not Working

**Symptom:** Queries like "emails from John" or "last week" don't work well.

**Solutions:**

**1. Verify Email Project Configuration**
```yaml
# config.yml
llm:
  prompt_strategy: email  # Force email template
```

**2. Use Email-Specific Phrasing**
```
# Good:
"Show me emails from john.smith@example.com about the API"
"What was discussed last week?"
"Find decisions from November"

# Bad:
"John's stuff"
"Recent things"
```

**3. Check Email Metadata**
```bash
# Verify emails have proper metadata
# Check: data/projects/MyProject/output/metadata/outlook_eml_metadata.jsonl

# Should have: sender, subject, date fields
```

---

## Outlook Integration Issues

### Outlook Not Found (WSL2)

**Error Message:**
```
Error: Outlook application not found or not accessible
```

**Cause:** Running in WSL2 but Outlook is on Windows host.

**Solution:**
```bash
# 1. Verify you're using outlook_wsl_client.py (not direct COM)
# This is the default in v1.1.0

# 2. Ensure Windows Outlook is running
# Open Outlook on Windows

# 3. Check WSL2 can access Windows
# Run: explorer.exe .
# Should open Windows File Explorer

# 4. Verify helper script exists
ls scripts/connectors/outlook_wsl_client.py
```

---

### Permission Denied Accessing Outlook

**Error Message:**
```
Access denied when accessing Outlook COM object
```

**Cause:** Security settings or Outlook trust center blocking access.

**Solution:**

**1. Enable Programmatic Access**
```
1. Open Outlook on Windows
2. File → Options → Trust Center → Trust Center Settings
3. Programmatic Access → Never warn about suspicious activity (UNSAFE - only for testing)
4. Click OK
```

**Warning:** This is a security risk. Only enable for development/testing.

**2. Run Outlook as Administrator**
```
1. Close Outlook
2. Right-click Outlook icon → Run as Administrator
3. Try extraction again
```

---

### Slow Email Extraction

**Symptom:** Extracting 1000 emails takes > 10 minutes.

**Solutions:**

**1. Limit Extraction**
```yaml
# config.yml
sources:
  outlook:
    max_emails: 500      # Limit to 500
    days_back: 30        # Only last 30 days
```

**2. Use Folder Filters**
```yaml
sources:
  outlook:
    folder_path: "Inbox > Project Emails"  # Specific folder only
```

**3. Check Windows Performance**
```
# Outlook extraction runs on Windows side
# Check Windows Task Manager for Outlook.exe CPU usage
# Close other Windows applications
```

---

## Performance Issues

### Slow Queries (> 10 seconds)

**Causes & Solutions:**

**1. Too Many Chunks Retrieved**
```yaml
# Check top_k setting
# Default is 10, which is usually fine

# In UI query page:
# Advanced options → Top K → Set to 10 or less
```

**2. Large Documents/Chunks**
```yaml
# Check chunk sizes
# configs/chunk_rules.yaml
# max_tokens: 300 is recommended
```

**3. LLM Model Too Large**
```yaml
# config.yml
llm:
  model: gpt-4o-mini  # Faster than gpt-4o
```

---

### High API Costs

**Monitoring Costs:**
```
# Check OpenAI dashboard:
# https://platform.openai.com/usage

# Costs breakdown:
# - Embedding: ~$0.13 per 1M tokens (text-embedding-3-large)
# - LLM: ~$0.15/$0.60 per 1M tokens (gpt-4o-mini input/output)
```

**Reducing Costs:**

**1. Use Deduplication** (default in v1.1.0)
```yaml
embedding:
  skip_duplicates: true  # Don't re-embed unchanged chunks
```

**2. Reduce Top-K**
```
# Fewer retrieved chunks = smaller prompts = lower LLM costs
top_k: 5-10 (default: 10)
```

**3. Use Smaller Models**
```yaml
llm:
  model: gpt-4o-mini     # $0.15/1M vs gpt-4o $2.50/1M
```

**4. Batch Processing**
```yaml
embedding:
  use_async_batch: true  # More efficient API usage
  embed_batch_size: 64
```

---

## Configuration Issues

### Config Changes Not Taking Effect

**Symptom:** Changed `config.yml` but system still uses old settings.

**Solutions:**

**1. Restart UI**
```bash
# Stop Streamlit (Ctrl+C)
# Restart
streamlit run scripts/ui/ui_v3.py
```

**2. Check Correct Config File**
```bash
# Make sure you're editing the project config, not global
data/projects/MyProject/config.yml  # ✅ Correct
configs/config.yml                  # ❌ Wrong (template)
```

**3. Verify YAML Syntax**
```bash
# Use a YAML validator
python -c "import yaml; yaml.safe_load(open('data/projects/MyProject/config.yml'))"
```

---

### Prompt Strategy Not Working (v1.1.0)

**Symptom:** Set `prompt_strategy: email` but still getting default template.

**Check:**
```bash
# 1. Verify you're on v1.1.0 or later
git log --oneline | head -1

# 2. Check logs for strategy confirmation
cat data/projects/MyProject/logs/app/prompt.log | grep strategy

# Should show:
# "strategy": "email"
```

**If still not working:**
```bash
# The fix was in commit: "Issue 1: Config Respect"
# Make sure you pulled latest changes
git pull origin feature/email-categorization
```

---

## Diagnostic Steps

### 1. Check Logs

**Application logs:**
```bash
# Main app log
cat data/projects/MyProject/logs/app/app.log | tail -100

# Specific subsystem logs
cat data/projects/MyProject/logs/app/prompt.log
cat data/projects/MyProject/logs/app/retrieval.log
cat data/projects/MyProject/logs/app/embedding.log
```

**Per-run logs:**
```bash
# Latest run
ls -lt data/projects/MyProject/logs/runs/ | head -1

# Check run logs
cat data/projects/MyProject/logs/runs/<run_id>/app.log
```

---

### 2. Verify File Structure

**Expected structure:**
```
data/projects/MyProject/
├── config.yml                    # Project config
├── input/
│   ├── raw/                      # Source documents
│   │   ├── pdf/
│   │   ├── docx/
│   │   └── eml/
│   ├── chunks_pdf.tsv            # Generated chunks
│   ├── chunks_outlook_eml.tsv
│   └── cache/images/             # Extracted images
├── output/
│   ├── faiss/                    # FAISS indices
│   │   ├── pdf.faiss
│   │   └── outlook_eml.faiss
│   ├── metadata/                 # Chunk metadata
│   │   ├── pdf_metadata.jsonl
│   │   └── outlook_eml_metadata.jsonl
│   ├── .last_config.yml          # Validator state
│   └── .last_metadata.json       # Validator metadata
└── logs/                         # All logs
```

---

### 3. Test with Known-Good Data

**Create test project:**
```bash
# 1. Create simple test project
# 2. Add ONE small PDF (1-2 pages)
# 3. Run pipeline
# 4. Query: "What is this document about?"

# If this works, your setup is fine
# If this fails, check prerequisites
```

---

### 4. Check System Resources

**Memory:**
```bash
# Linux/macOS
free -h

# Should have 2GB+ free for small projects
# 8GB+ for large email projects
```

**Disk Space:**
```bash
df -h

# FAISS indices can be large (100MB+ for 1000+ documents)
```

**Python Version:**
```bash
python --version
# Should be 3.10 or higher
```

---

## Getting Further Help

**If none of these solutions work:**

1. **Check GitHub Issues**
   - Search for similar problems
   - Create new issue with:
     - Error message (full traceback)
     - Steps to reproduce
     - Logs (attach relevant log files)
     - Config (sanitized, no API keys)

2. **Provide Diagnostic Info**
   ```bash
   # System info
   uname -a
   python --version
   poetry --version

   # Project info
   ls -la data/projects/MyProject/

   # Error logs
   cat data/projects/MyProject/logs/app/app.log | tail -100
   ```

3. **Check Documentation**
   - [USER_GUIDE.md](USER_GUIDE.md) - Detailed usage guide
   - [FAQ.md](FAQ.md) - Common questions
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design
   - [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Setup help

---

**Most issues are solved by:**
1. ✅ Restarting the UI
2. ✅ Checking the logs
3. ✅ Verifying file structure
4. ✅ Running validator (automatic in v1.1.0)
5. ✅ Following the fix recommendations

**Good luck!** 🛠️
