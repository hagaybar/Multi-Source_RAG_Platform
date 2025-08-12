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

5.  **`ask`**
    *   **Description**: Asks a question to the RAG system. It retrieves relevant context chunks and uses an LLM to generate an answer.
    *   **Usage**: `python -m app.cli ask <project_path> <query> [--top_k <k>] [--temperature <t>] [--max_tokens <m>] [--model_name <model>]`
    *   **Arguments**:
        *   `project_path`: (Required) Path to the RAG project directory.
        *   `query`: (Required) The question to ask the RAG system.
    *   **Options**:
        *   `--top_k <k>`: (Optional) Number of context chunks to retrieve (default: 5).
        *   `--temperature <t>`: (Optional) LLM temperature for response generation (default: 0.7).
        *   `--max_tokens <m>`: (Optional) LLM maximum tokens for the response (default: 500).
        *   `--model_name <model>`: (Optional) OpenAI model to use for generating the answer (default: "gpt-3.5-turbo").
    *   **Modules Used**:
        *   `scripts.retrieval.retrieval_manager.RetrievalManager`: For retrieving context chunks.
        *   `scripts.prompting.prompt_builder.PromptBuilder`: For building the prompt for the LLM.
        *   `scripts.api_clients.openai.completer.OpenAICompleter`: For getting the completion from the LLM.
        *   `scripts.core.project_manager.ProjectManager`: For project context.

6.  **`enrich-images`**
    *   **Description**: Enrich chunks with image summaries using the ImageInsightAgent.
    *   **Usage**: `python -m app.cli enrich-images <project_path> [--doc_type <doc_type>] [--overwrite]`
    *   **Arguments**:
        *   `project_path`: (Required) Path to the project folder.
    *   **Options**:
        *   `--doc_type <doc_type>`: (Optional) Document type to enrich (e.g., pptx, pdf, docx) (default: "pptx").
        *   `--overwrite`: (Optional) Overwrite original TSV instead of saving to /enriched.
    *   **Modules Used**:
        *   `scripts.agents.image_insight_agent.ImageInsightAgent`: For enriching chunks with image summaries.
        *   `scripts.core.project_manager.ProjectManager`: For project context.

7.  **`index-images`**
    *   **Description**: Index enriched image summaries into a dedicated FAISS index (`image_index.faiss`) and metadata file (`image_metadata.jsonl`).
    *   **Usage**: `python -m app.cli index-images <project_path> [--doc_type <doc_type>]`
    *   **Arguments**:
        *   `project_path`: (Required) Path to the RAG project directory.
    *   **Options**:
        *   `--doc_type <doc_type>`: (Optional) The document type to read the enriched chunks from (default: "pptx").
    *   **Modules Used**:
        *   `scripts.embeddings.image_indexer.ImageIndexer`: For indexing the image chunks.
        *   `scripts.core.project_manager.ProjectManager`: For project context.

## Integration with the Project

The `app` folder serves as the user-facing layer of the project. It orchestrates calls to various managers and utilities within the `scripts` directory (e.g., `IngestionManager`, `UnifiedEmbedder`, `RetrievalManager`, `ProjectManager`). This separation allows for a clean distinction between the CLI definition and the underlying implementation of core functionalities.
