# Embeddings Module

The `scripts/embeddings` folder contains modules responsible for generating and managing text and image embeddings, which are crucial for the Retrieval Augmented Generation (RAG) capabilities of this project. These embeddings are numerical representations of content that capture semantic meaning, enabling tasks like similarity search.

## Core Components

The embedding system is designed to be modular and configurable, allowing for different embedding providers and processing strategies for both text and images.

### 1. `base.py` - Abstract Base Class
*   **`BaseEmbedder`**: An abstract base class (ABC) that defines the common interface for all embedder implementations. It mandates an `encode` method that takes a list of texts and returns a NumPy array of their embeddings.

### 2. `bge_embedder.py` - Local BGE Embedder
*   **`BGEEmbedder(BaseEmbedder)`**: An embedder that uses the `sentence-transformers` library to generate embeddings locally. It is suitable for scenarios where local processing is preferred.

### 3. `litellm_embedder.py` - LiteLLM API Embedder
*   **`LiteLLMEmbedder(BaseEmbedder)`**: An embedder designed to work with LiteLLM-compatible embedding APIs, such as those provided by OpenAI, Ollama, or Together.ai.

### 4. `embedder_registry.py` - Embedder Factory
*   **`get_embedder(project: ProjectManager) -> BaseEmbedder`**: A factory function that instantiates and returns the appropriate embedder based on the project's configuration. It ensures a single instance of the embedder is created and reused.

### 5. `unified_embedder.py` - Text Embedding Orchestrator
*   **`UnifiedEmbedder`**: This is the primary class that orchestrates the embedding generation workflow for text chunks.
    *   **Embedder Selection**: Uses the `embedder_registry` to get the configured embedder.
    *   **Chunk Loading and Deduplication**: Loads text chunks and avoids re-embedding identical content by checking against previously processed chunk hashes.
    *   **Processing Modes**: Supports both synchronous batch processing (for local or standard API embedders) and asynchronous batch processing via OpenAI's batch API for large datasets.
    *   **Storage**: Saves the generated embeddings into a FAISS index and the corresponding metadata into a JSONL file. It creates separate indexes and metadata files for each document type.

### 6. `image_indexer.py` - Image Embedding Orchestrator
*   **`ImageIndexer`**: This class handles the embedding and indexing of images.
    *   **Initialization**: Takes a `ProjectManager` instance and uses the `embedder_registry` to get the configured embedder.
    *   **Embedding**: Takes a list of `ImageChunk` objects, extracts their descriptions, and uses the embedder to generate embeddings for these descriptions.
    *   **Storage**: Saves the image embeddings into a dedicated FAISS index (`image_index.faiss`) and the corresponding metadata (including the image description and path) into a JSONL file (`image_metadata.jsonl`).

## Workflow Overview

1.  The `UnifiedEmbedder` (for text) or `ImageIndexer` (for images) is instantiated with a `ProjectManager`.
2.  Based on the project configuration, the `embedder_registry` provides the appropriate `BaseEmbedder` implementation.
3.  The orchestrator loads the chunks (text or image) to be processed.
4.  It generates embeddings for the new chunks using the selected embedder.
5.  The embeddings are stored in a FAISS index, and the metadata is saved to a corresponding JSONL file.
6.  These FAISS indexes and metadata files are then used by the retrieval system to find relevant information.

This system provides a robust and flexible way to convert both textual and visual content into a searchable vector space, forming a foundational component of the project's information retrieval capabilities.
