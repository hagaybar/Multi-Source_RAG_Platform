# Scripts

This directory contains the core Python scripts that power the RAG platform. It is organized into several subdirectories, each responsible for a specific part of the data processing and retrieval pipeline.

## Subdirectories

- **`agents`**: Contains AI agents that perform specialized tasks, such as the `ImageInsightAgent` which generates textual descriptions for images. For more details, see `scripts/agents/README.md`.
- **`api_clients`**: Provides clients for interacting with external APIs, such as the OpenAI API for embeddings and completions.
- **`chunking`**: Responsible for splitting raw documents into smaller, more manageable chunks based on a set of configurable rules. For more details, see `scripts/chunking/README.md`.
- **`core`**: Contains the `ProjectManager`, a central class for managing the project's configuration and directory structure. For more details, see `scripts/core/README.md`.
- **`embeddings`**: Handles the generation of vector embeddings for both text and image content. It includes a registry for different embedding models and orchestrators for the embedding process. For more details, see `scripts/embeddings/README.md`.
- **`index`**: Contains scripts for managing and inspecting the vector indexes (e.g., FAISS) used for efficient retrieval. For more details, see `scripts/index/README.md`.
- **`ingestion`**: Provides a suite of loaders for ingesting various file formats, including PDFs, DOCX, XLSX, PPTX, and emails. For more details, see `scripts/ingestion/README.md`.
- **`interface`**: Contains the `AskInterface`, which provides a high-level, unified interface for asking questions to the RAG system.
- **`pipeline`**: Includes the `PipelineRunner`, which orchestrates the entire RAG pipeline, from ingestion to chunking, embedding, and retrieval.
- **`prompting`**: Contains the `PromptBuilder`, which is responsible for constructing the final prompts that are sent to the LLM, incorporating the retrieved context.
- **`retrieval`**: Implements the retrieval system, which finds the most relevant text and image chunks from the indexes based on a user's query. For more details, see `scripts/retrieval/README.md`.
- **`ui`**: Contains the Streamlit-based user interfaces for interacting with the RAG platform.
- **`utils`**: A collection of utility modules that provide helper functions for tasks such as logging, configuration loading, and file conversions. For more details, see `scripts/utils/README.md`.

## Overall Workflow

These scripts work together to form a complete RAG pipeline:

1.  The **`ingestion`** scripts load raw documents from various sources.
2.  The **`chunking`** scripts split these documents into smaller chunks.
3.  The **`agents`** (e.g., the `ImageInsightAgent`) can then be used to enrich these chunks with additional information.
4.  The **`embeddings`** scripts generate vector embeddings for the chunks.
5.  These embeddings are stored in a vector **`index`**.
6.  When a user asks a question, the **`retrieval`** scripts find the most relevant chunks from the index.
7.  The **`prompting`** script constructs a prompt that includes the user's question and the retrieved context.
8.  This prompt is sent to a large language model to generate an answer.
9.  The **`pipeline`** and **`interface`** scripts orchestrate this entire process, and the **`ui`** scripts provide a user-friendly way to interact with the system.
