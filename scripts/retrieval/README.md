# Retrieval Module

This module implements the retrieval system for the RAG platform. It is designed to be a flexible and extensible system that can query across multiple data types (e.g., text, images) and apply different retrieval strategies.

## Overview

The retrieval system is responsible for finding the most relevant information from the indexed data based on a user's query. It uses a `RetrievalManager` to orchestrate the process, which includes embedding the query, fetching candidate chunks from various retrievers, and applying a strategy to rank and fuse the results.

## File Structure

- **`retrieval_manager.py`**: The central orchestration layer. It manages all the retrievers (for different data types) and applies retrieval strategies to produce the final, ranked list of results.
- **`base.py`**: Defines the `BaseRetriever` abstract class, which provides a common interface for all retrievers. It also includes the `FaissRetriever`, a concrete implementation for retrieving text chunks from a FAISS index.
- **`image_retriever.py`**: Implements the `ImageRetriever` class, which is specialized for retrieving image chunks from a FAISS index.
- **`strategies/`**: A directory containing different retrieval strategies.
  - **`late_fusion.py`**: Implements a late-fusion strategy, which queries each retriever independently, combines the results, and then sorts them by similarity score.
  - **`strategy_registry.py`**: A registry that maps strategy names to their corresponding implementation functions. This allows for easy selection and extension of retrieval strategies.

## Design Principles

- **Modular and Extensible**: The system is designed to be easily extended with new retrievers (e.g., for different data types or indexing technologies) and new retrieval strategies.
- **Multi-Modal**: The architecture supports retrieving information from multiple modalities (e.g., text and images) and fusing the results.
- **Strategy-Driven**: The retrieval process is driven by a selected strategy, which can be chosen based on the specific use case or even dynamically by an agent.

## Core Components

### RetrievalManager

The `RetrievalManager` is the main entry point for the retrieval system. It performs the following key functions:
- Loads and initializes all available retrievers (e.g., `FaissRetriever` for different document types, `ImageRetriever`).
- Embeds the user's query using the configured embedding model.
- Optionally translates the query to English for multilingual support.
- Invokes the selected retrieval strategy to get a list of candidate chunks.
- Fuses text and image results, promoting text chunks that are associated with relevant images.
- Deduplicates the final list of chunks before returning it.

### Retrievers

- **`FaissRetriever`**: A retriever for text data stored in a FAISS index. It takes a query vector and returns a list of the most similar text `Chunk`s.
- **`ImageRetriever`**: A retriever for image data. It searches a FAISS index of image embeddings and returns a list of the most relevant `ImageChunk`s.

### Strategies

- **`late_fusion`**: This is the default strategy. It works by:
  1. Querying each registered retriever (e.g., for DOCX, PDF, images) independently with the same query vector.
  2. Collecting all the returned chunks into a single list.
  3. Sorting the combined list globally based on the similarity score.
  4. Returning the top-K results.

## Usage

The `RetrievalManager` is typically used within a higher-level component, such as a CLI or a UI. To perform a retrieval, you would first instantiate the `RetrievalManager` with a `ProjectManager` object and then call the `retrieve` method:

```python
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager

# Assuming 'project' is an initialized ProjectManager object
retrieval_manager = RetrievalManager(project)

# Perform a retrieval using the late_fusion strategy
results = retrieval_manager.retrieve(
    query="What are the latest updates on the project?",
    top_k=10,
    strategy="late_fusion"
)

# The 'results' variable now contains a list of the top 10 most relevant chunks.
```