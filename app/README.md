# App Folder

The `app` folder provides the command-line interface (CLI) for interacting with the core functionalities of this project. It acts as the primary user entry point for processes such as document ingestion, embedding generation, and information retrieval.

## Files

-   `__init__.py`: An empty file that designates the `app` folder as a Python package.
-   `cli.py`: This file defines the CLI commands using the `typer` library, offering a user-friendly way to execute various project operations.

## CLI Commands

The `cli.py` script exposes the following commands:

1.  **`ingest`**
    *   **Description**: Ingests documents from a specified folder. It can optionally chunk these documents into smaller segments.
    *   **Usage**: `python -m app.cli ingest <folder_path> [--chunk]`
    *   **Arguments**:
        *   `folder_path`: (Required) Path to the folder containing documents to ingest.
    *   **Options**:
        *   `--chunk`: (Optional) If provided, enables the chunking of ingested documents.
    *   **Modules Used**:
        *   `scripts.ingestion.manager.IngestionManager`: For handling the document ingestion process.
        *   `scripts.chunking.chunker_v3.split`: For splitting documents into chunks if the `--chunk` option is enabled.
        *   `scripts.core.project_manager.ProjectManager`: For managing project-level configurations and paths.

2.  **`embed`**
    *   **Description**: Generates embeddings for text chunks located in a specified project directory. It reads chunk data (typically from `chunks_<doc_type>.tsv` files), creates embeddings, and stores them (e.g., in a FAISS index) along with metadata.
    *   **Usage**: `python -m app.cli embed <project_dir> [--async-batch]`
    *   **Arguments**:
        *   `project_dir`: (Required) Path to the project directory.
    *   **Options**:
        *   `--async-batch` / `--a-b`: (Optional) If provided, uses OpenAI's asynchronous batch embedding.
    *   **Modules Used**:
        *   `scripts.embeddings.unified_embedder.UnifiedEmbedder`: For creating embeddings from text chunks.
        *   `scripts.core.project_manager.ProjectManager`: For accessing project configuration and paths.

3.  **`retrieve`**
    *   **Description**: Retrieves the top-k most relevant chunks from the indexed documents based on a user query. It supports different retrieval strategies.
    *   **Usage**: `python -m app.cli retrieve <project_path> <query> [--top_k <k>] [--strategy <strategy_name>]`
    *   **Arguments**:
        *   `project_path`: (Required) Path to the RAG project directory.
        *   `query`: (Required) The search query string.
    *   **Options**:
        *   `--top_k <k>`: (Optional) Number of top chunks to return (default: 10).
        *   `--strategy <strategy_name>`: (Optional) Retrieval strategy to use (default: "late_fusion").
    *   **Modules Used**:
        *   `scripts.retrieval.retrieval_manager.RetrievalManager`: For managing the retrieval process.
        *   `scripts.core.project_manager.ProjectManager`: For project context.

4.  **`config`**
    *   **Description**: Prints the configuration values for a specified project directory. This is useful for inspecting project settings, especially embedding configurations.
    *   **Usage**: `python -m app.cli config <project_dir>`
    *   **Arguments**:
        *   `project_dir`: (Required) Path to the project directory.
    *   **Modules Used**:
        *   `scripts.core.project_manager.ProjectManager`: To load and display the project's configuration.

## Integration with the Project

The `app` folder serves as the user-facing layer of the project. It orchestrates calls to various managers and utilities within the `scripts` directory (e.g., `IngestionManager`, `UnifiedEmbedder`, `RetrievalManager`, `ProjectManager`). This separation allows for a clean distinction between the CLI definition and the underlying implementation of core functionalities.
