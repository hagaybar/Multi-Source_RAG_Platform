# Embeddings (`scripts/embeddings`)

The `scripts/embeddings` folder contains modules responsible for generating and managing text embeddings, which are crucial for the Retrieval Augmented Generation (RAG) capabilities of this project. These embeddings are numerical representations of text that capture semantic meaning, enabling tasks like similarity search.

## Core Components

The embedding system is designed to be modular and configurable, allowing for different embedding providers and processing strategies.

### 1. `base.py` - Abstract Base Class
*   **`BaseEmbedder`**: An abstract base class (ABC) that defines the common interface for all embedder implementations.
    *   It mandates an `encode(self, texts: List[str]) -> np.ndarray` method, which should take a list of text strings and return a NumPy array of their corresponding float32 embeddings.

### 2. `bge_embedder.py` - Local BGE Embedder
*   **`BGEEmbedder(BaseEmbedder)`**: An embedder that uses the `sentence-transformers` library to generate embeddings locally.
    *   It defaults to the "BAAI/bge-large-en" model but can be configured to use other SentenceTransformer-compatible models.
    *   Suitable for scenarios where local processing is preferred or when specific open-source models are required.

### 3. `litellm_embedder.py` - LiteLLM API Embedder
*   **`LiteLLMEmbedder(BaseEmbedder)`**: An embedder designed to work with LiteLLM-compatible embedding APIs, such as those provided by OpenAI, Ollama, Together.ai, and others.
    *   It handles making HTTP requests to the specified API endpoint, including authentication (e.g., API keys).
    *   It parses the API response to extract the embedding vectors.

### 4. `embedder_registry.py` - Embedder Factory
*   **`get_embedder(project: ProjectManager) -> BaseEmbedder`**: A factory function that instantiates and returns the appropriate embedder based on the project's configuration (`project.config`).
    *   It reads the `embedding.provider` setting from the configuration (e.g., "local", "litellm").
    *   If "local", it initializes a `BGEEmbedder` (configurable model).
    *   If "litellm", it initializes a `LiteLLMEmbedder` (configurable endpoint, model, API key).
    *   It ensures a single instance of the embedder is created and reused.

### 5. `unified_embedder.py` - Main Orchestrator
*   **`UnifiedEmbedder`**: This is the primary class that orchestrates the entire embedding generation workflow.
    *   **Initialization**: Takes a `ProjectManager` instance for accessing configuration and managing file paths. It also uses `LoggerManager` for logging.
    *   **Embedder Selection**: Uses `embedder_registry.get_embedder()` to obtain an instance of the configured embedder.
    *   **Chunk Loading**: Loads text chunks to be embedded, typically from `.tsv` files (e.g., `chunks.tsv` or `chunks_*.tsv`) located in the project's input directory. Chunks are expected to be instances of `scripts.chunking.models.Chunk`.
    *   **Deduplication**: Implements a deduplication strategy by calculating content hashes (SHA256) of chunk texts. It checks against previously processed chunk hashes (stored in metadata files) to avoid re-embedding identical content, if `skip_duplicates` is enabled in the configuration.
    *   **Processing Modes**:
        *   **Synchronous Batch**: For local or standard API embedders, it processes chunks in batches for efficiency.
        *   **Asynchronous Batch (OpenAI)**: If configured (`embedding.use_async_batch: true`), it leverages `scripts.api_clients.openai.batch_embedder.BatchEmbedder` to use OpenAI's batch API for potentially faster and more cost-effective embedding of large datasets. In this mode, the local embedder from the registry is bypassed for the OpenAI-specific batch client.
    *   **Storage**:
        *   **FAISS Index**: Saves the generated numerical embeddings into FAISS indexes (`.index` files). FAISS allows for efficient similarity searching over large sets of vectors. Indexes are typically created per `doc_type`.
        *   **Metadata**: Stores metadata associated with each chunk (including the original text, chunk ID, content hash, and any other relevant details from the `Chunk` object) in JSONL files (`.jsonl`). Each line in the file is a JSON object representing a chunk's metadata.
    *   **Output Management**: Organizes output FAISS indexes and metadata files into directories managed by `ProjectManager`, typically within `output/faiss/` and `output/metadata/`, often further subdirectory by `doc_type`.
    *   **Grouping**: Processes chunks grouped by their `doc_type` (an attribute of the `Chunk` metadata), allowing for separate FAISS indexes and metadata files for different types of documents.

## Workflow Overview

1.  The `UnifiedEmbedder` is instantiated with a `ProjectManager`.
2.  Based on project configuration (e.g., `config.yml`), the `embedder_registry` provides the appropriate `BaseEmbedder` implementation (either `BGEEmbedder` for local embeddings or `LiteLLMEmbedder` for API-based embeddings), unless OpenAI async batch mode is selected.
3.  `UnifiedEmbedder` loads text chunks.
4.  It filters out chunks that have already been embedded (deduplication based on content hash).
5.  For new chunks:
    *   If using OpenAI async batch mode, it prepares data and uses `BatchEmbedder` to submit an asynchronous job to OpenAI, then retrieves results.
    *   Otherwise, it uses the selected embedder's `encode` method to generate embeddings for batches of chunk texts.
6.  The generated embeddings are added to a FAISS index, and corresponding metadata (including the original chunk text) is saved to a JSONL file.
7.  These FAISS indexes and metadata files are then used by other parts of the application (e.g., a retrieval system) to find relevant information for RAG.

This system provides a robust and flexible way to convert textual content into a searchable vector space, forming a foundational component of the project's information retrieval capabilities.
\n*Updated overview of embedding utilities.*
