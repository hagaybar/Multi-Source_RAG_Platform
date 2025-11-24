# Quick Start - Multi-Source RAG Platform

**Get your first answer in 5 minutes!** ⚡

---

## Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- 8GB+ RAM

---

## 1. Install (2 minutes)

```bash
# Clone and install
git clone <repository-url>
cd Multi-Source_RAG_Platform
poetry install
poetry shell

# Set up API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here
```

---

## 2. Launch UI (30 seconds)

```bash
streamlit run scripts/ui/ui_v3.py
```

Browser opens automatically at `http://localhost:8501`

---

## 3. Create Project (1 minute)

**In the UI:**

1. Go to "Project Management" tab
2. Click "Create New Project"
3. Enter:
   - **Name:** `My_Test_Project`
   - **Description:** `Testing RAG system`
4. Click "Create"

---

## 4. Add Documents (30 seconds)

**Add your documents to:**
```
data/projects/My_Test_Project/input/raw/
```

**Supported formats:**
- PDFs → `raw/pdf/`
- Word docs → `raw/docx/`
- Emails (.eml) → `raw/eml/`
- PowerPoint → `raw/pptx/`
- Excel → `raw/xlsx/`

**Example:**
```bash
cp ~/Documents/my-report.pdf data/projects/My_Test_Project/input/raw/pdf/
```

**Or use the UI:** "Data Upload" tab → Drag & drop files

---

## 5. Run Pipeline (1 minute)

**In the UI → "Pipeline" tab:**

1. Select steps:
   - ☑️ Ingest
   - ☑️ Chunk
   - ☑️ Embed
2. Click **"Run Pipeline"**
3. Watch progress:
   ```
   ✅ Validation passed
   🚀 Starting ingestion...
   ✅ Ingested 5 documents
   📚 Starting chunking...
   ✅ Chunking complete. Total chunks: 87
   🧬 Starting embedding...
   ✅ Embedded and indexed all chunks
   ```

**⏱️ Time:** ~30-60 seconds for 5-10 documents

---

## 6. Ask Your First Question! (30 seconds)

**In the UI → "Query" tab:**

1. Enter a question:
   ```
   What are the main findings in the report?
   ```
2. Click **"Submit"**
3. Get your answer with source citations:
   ```
   Based on the documents, the main findings include:
   1. Performance improved by 35% [report.pdf]
   2. User satisfaction increased to 4.8/5 [survey.pdf]
   3. Cost reduction of $50,000 annually [budget.xlsx]
   ```

**🎉 Success!** You've queried your first documents.

---

## Next Steps

### Try Different Queries

**Simple lookups:**
```
"What is the project deadline?"
"Who is responsible for testing?"
```

**For email projects:**
```
"What topics were discussed last week?"
"Show me emails from John about the bug"
```

**Complex queries:**
```
"Summarize the key decisions across all documents"
"What are the common issues mentioned?"
```

### Add More Documents

Just drop more files in `input/raw/` and run the pipeline again:
- **Smart deduplication** skips unchanged content
- **Only new files** get processed
- **Cost-efficient** - no wasted API calls

### Learn More

📖 **Full Guide:** [USER_GUIDE.md](USER_GUIDE.md)
🔧 **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
❓ **FAQ:** [FAQ.md](FAQ.md)
🏗️ **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Common Issues

**"Module not found" error**
```bash
poetry install  # Re-install dependencies
poetry shell    # Activate virtual environment
```

**"OpenAI API key not found"**
```bash
# Check .env file exists and contains:
OPENAI_API_KEY=sk-your-actual-key-here
```

**"No documents found"**
```bash
# Check files are in the right location:
ls data/projects/My_Test_Project/input/raw/
# Should show your documents
```

**Pipeline takes too long**
- First run takes longer (downloading models)
- Subsequent runs are faster (reuses embeddings)
- 100 documents ≈ 2-3 minutes

---

## What Just Happened?

1. **Ingest:** Loaded your documents and extracted text
2. **Chunk:** Split documents into searchable pieces (~200 tokens each)
3. **Embed:** Created vector embeddings for semantic search (3072-dim)
4. **Index:** Stored embeddings in FAISS (fast similarity search)
5. **Query:** Found relevant chunks + generated answer with GPT-4o

**Cost:** ~$0.10-0.20 for 100 documents (first time), then $0.01 per query

---

## Pro Tips

✅ **Start small:** Test with 5-10 documents first
✅ **Use descriptive filenames:** Easier to find source documents
✅ **One project per topic:** Don't mix unrelated documents
✅ **Check logs:** `data/projects/<name>/logs/` if something goes wrong

---

**Need help?** Check [USER_GUIDE.md](USER_GUIDE.md) or [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Ready to deploy?** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

**Happy querying!** 🚀
