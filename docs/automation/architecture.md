# ARCHITECTURE – RAG-GP Prototype

_Last updated: 2025-06-28_

---

## 🎯 Objective
Build a modular, local-first Retrieval-Augmented Generation (RAG) system that can:
- Ingest mixed-format internal documents
- Chunk them by type-aware logic
- Embed them locally or via OpenAI
- Retrieve relevant context
- Answer queries with cited sources in ≤ 5 s

---

## 🧱 Core Pipeline Components

```text
CLI / Streamlit
   ┃
   ┃  ┌──────────────┐
   ┃  │ Ingestion    │ ← IngestionManager + per-type loader
   ┃  └────┬─────────┘
   ┃       ▼
   ┃  ┌──────────────┐
   ┃  │ Chunking     │ ← Rules per doc_type (paragraph, slide, etc.)
   ┃  └────┬─────────┘
   ┃       ▼
   ┃  ┌──────────────┐
   ┃  │ Embedding    │ ← UnifiedEmbedder (local/BGE or OpenAI/batch)
   ┃  └────┬─────────┘
   ┃       ▼
   ┃  ┌──────────────┐
   ┃  │ Retrieval    │ ← RetrievalManager + late fusion
   ┃  └────┬─────────┘
   ┃       ▼
   ┃  ┌──────────────┐
   ┃  │ Prompt + LLM │ ← prompt_builder (planned), gateway (TBD)
   ┃  └──────────────┘
```

---

## 🧩 Component Breakdown

### 1. Ingestion Layer
- `IngestionManager` detects file types and dispatches to:
  - `pdf.py`, `docx_loader.py`, `pptx.py`, `csv.py`, `xlsx.py`, `email_loader.py`
- Each loader returns text + metadata per segment
- Extensible via registry (LOADER_REGISTRY)

### 2. Chunking Layer
- Central: `chunker_v3.py` applies rules from `chunk_rules.yaml`
- Strategy options: `by_paragraph`, `by_slide`, `by_sheet`, `by_row`, etc.
- Maintains doc_id, chunk_id, metadata in output TSV
- Token overlap logic prevents context loss

### 3. Embedding Layer
- `UnifiedEmbedder` supports:
  - Local (e.g., BGE): via `bge_embedder.py`
  - Remote: via `litellm_embedder.py` or `batch_embedder.py`
- Deduplication via SHA-256 hashes
- Outputs: FAISS index + JSONL metadata

### 4. Retrieval Layer
- `RetrievalManager` loads per-type FAISS indexes
- Supports fusion strategy (`late_fusion.py`)
- Filters by doc_type, date, project, etc.
- Score-ranked top-K results sent to prompting stage

### 5. Prompt + LLM (Planned)
- `prompt_builder.py` will generate QA prompt templates
- `gateway.py` (TBD) will connect to LLM endpoint (OpenAI / local)
- Will concatenate text + OCR image context where available

### 6. CLI + Streamlit Interface
- `app/cli.py` provides entrypoints:
  - `ingest`, `chunk`, `embed`, `retrieve`, `ask`
- Streamlit app (TBD) will visualize:
  - Retrieved chunks
  - FAISS scores
  - Logs + input mapping

---

## 🧾 Data Flow Summary

```
input/raw/*            → ingestion → RawDoc list
RawDoc list            → chunking  → chunks_*.tsv
chunks_*.tsv           → embedding → *.faiss + *.jsonl
query + config         → retrieval → top-k chunk set
chunks + prompt config → ask()     → answer + citations
```

---

## 🧪 Logging & Config
- `LoggerManager` handles per-module logs: `ingestion.log`, `chunker.log`, etc.
- `project_manager.py` resolves paths based on config.yml structure
- Configurable:
  - embedding provider
  - chunk rules
  - async vs sync mode

---

## 🧠 Extensibility Hooks
- Prompt + LLM layer is modular
- Retrieval agent rerankers planned
- OCR integration supports `image_ocr_text` field
- Email threading and XLSX summarization pending

---

## 🧰 Project Status Integration
- See `MANIFEST.md` for module list
- See `AUDIT_CHECKLIST.md` for implementation status
- Follows 8-week roadmap as defined in `roadmap.txt`

---

## 📌 Upcoming Additions
- `ask()` CLI command
- Streamlit prototype
- RerankerAgent + SynthesizerAgent
- Table summarization
- OCR screenshot captioning

