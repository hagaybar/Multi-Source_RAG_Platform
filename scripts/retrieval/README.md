# Retrieval Layer – RAG Pipeline

This module implements the **retrieval system** for a modular, local-first RAG platform. It is designed for flexibility, extensibility, and future integration with AI agents and UI tools.

---

## 🔍 Overview

The retrieval system handles querying across multiple data types (PDF, DOCX, XLSX, etc.), each with its own FAISS index. It supports late-fusion ranking, score normalization, and future agent-driven dynamic strategy selection.

---

## 📁 File Structure

| File / Directory | Purpose | Connected To |
|------|---------|---------------|
| `retrieval_manager.py` | Orchestration layer. Manages retrieval strategies and result fusion. | CLI, UI, Agents, `scripts.embeddings.embedder_registry` |
| `base.py` | BaseRetriever abstract class and FaissRetriever implementation. | RetrievalManager, `scripts.embeddings.embedder_registry` |
| `strategies/` | Directory containing retrieval strategy implementations (e.g., `late_fusion.py`). | `strategy_registry.py` |
| `strategies/strategy_registry.py` | Registers available retrieval strategies. | RetrievalManager, strategy files in `strategies/` |
| `scripts/utils/chunk_utils.py` | Contains utility functions like `deduplicate_chunks`. | RetrievalManager |
| `scripts/embeddings/embedder_registry.py` | Provides embedding models for queries. | RetrievalManager, FaissRetriever |
| `app/cli.py` *(relevant section)* | Adds `retrieve` command to CLI, utilizing `RetrievalManager`. | RetrievalManager |
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

Retrieval strategies are defined in the `scripts/retrieval/strategies/` directory and registered in `scripts/retrieval/strategies/strategy_registry.py`.

Available strategies:
- `late_fusion` (defined in `scripts/retrieval/strategies/late_fusion.py`): Retrieve top-K from each document type, then sort all candidates globally by similarity score. Score normalization can be added as a future enhancement to this or other strategies.
- `per_type_top_k` *(example, not currently implemented)*: Raw K-per-type (no fusion or global ranking).
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
- BM25 and hybrid retrievers\n*Added note about strategy registry.*
