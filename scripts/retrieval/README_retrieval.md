# Retrieval Layer – RAG Pipeline

This module implements the **retrieval system** for a modular, local-first RAG platform. It is designed for flexibility, extensibility, and future integration with AI agents and UI tools.

---

## 🔍 Overview

The retrieval system handles querying across multiple data types (PDF, DOCX, XLSX, etc.), each with its own FAISS index. It supports late-fusion ranking, score normalization, and future agent-driven dynamic strategy selection.

---

## 📁 File Structure

| File | Purpose | Connected To |
|------|---------|---------------|
| `retrieval_manager.py` | Orchestration layer. Manages retrieval strategies and result fusion. | CLI, UI, Agents |
| `base.py` | BaseRetriever abstract class and FaissRetriever implementation. | RetrievalManager |
| `strategies.py` | Strategy plug-ins: late fusion, agentic rerank, etc. | RetrievalManager |
| `utils.py` | Helpers for score normalization, sorting, deduplication. | Strategies |
| `embedder.py` *(optional)* | Shared embedding logic to reuse model across embedding + querying. | FaissRetriever |
| `app/cli.py` *(modify)* | Adds `retrieve` command to CLI. | RetrievalManager |
| `agent_router.py` *(future)* | Agent-based dynamic routing (e.g. pick best strategy). | RetrievalManager, LLMs |

---

## 🧠 Design Principles

- **Modular**: Each retrieval strategy is a pluggable function.
- **Multi-type Aware**: Supports late fusion over FAISS indexes per doc type.
- **Strategy-Driven**: Select strategies via code, config, or agents.
- **Agent-Ready**: RetrievalManager is callable by an orchestration agent.
- **Extensible**: Can add BM25, hybrid or metadata-based retrievers.

---

## 🚀 RetrievalManager

Core class that handles:
- Loading per-type retrievers (e.g., `FaissRetriever`)
- Selecting and applying a retrieval strategy
- Returning final ranked `List[Chunk]`

---

## 🔌 Strategies

Available strategies (in `strategies.py`):
- `late_fusion`: Retrieve top-K from each type, normalize, re-rank globally
- `per_type_top_k`: Raw K-per-type (no fusion)
- `agentic_rerank`: Stub for LLM-based reranking (future)

---

## 🧪 CLI Usage

After implementation, run:

```bash
python -m app.cli retrieve \
    --query "alma analytics letters" \
    --project-dir data/projects/demo_project
```

---

## 🔄 Future Work

- `agent_router.py`: Agentic coordination of strategies
- UI support in Streamlit
- Metadata filters and time-range constraints
- BM25 and hybrid retrievers