# Multi-Source RAG Platform

This project is a sophisticated, local-first Retrieval-Augmented Generation (RAG) platform designed to transform a diverse range of organizational content—including PDFs, Office documents, emails, and images—into an interactive and searchable knowledge base. It is built for anyone who needs to derive reliable answers from in-house documentation without compromising data privacy.

<p align="center">
  <img src="docs/architecture.png" width="700" alt="High-level architecture of the RAG platform"/>
</p>

---

## ✨ Key Features

*   **Multi-Source Ingestion**: Supports a wide variety of file formats, including PDF, DOCX, XLSX, PPTX, CSV, EML, TXT, and images.
*   **Configurable Chunking**: Employs a rule-based chunking system that allows for different strategies (e.g., by paragraph, by slide) to be applied to different document types, ensuring optimal data segmentation.
*   **Flexible Embedding Models**: Easily switch between local, open-source embedding models (via `sentence-transformers`) and powerful API-based models like OpenAI's.
*   **Multi-Modal Retrieval**: Capable of retrieving both text and image-based information. The system can generate textual descriptions for images, making visual content fully searchable.
*   **Advanced Retrieval Strategies**: Uses a late-fusion approach to combine results from multiple sources, ensuring comprehensive and relevant context for every query.
*   **Command-Line Interface**: A powerful CLI for interacting with the platform, allowing you to ingest documents, generate embeddings, and ask questions directly from your terminal.
*   **Local-First and Secure**: All your data, including raw files, indexes, and logs, is stored locally on your machine, ensuring complete privacy and control.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.10 or higher
*   An API key for your chosen LLM and embedding providers (e.g., `OPENAI_API_KEY`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The platform is operated through its command-line interface, which is defined in `app/cli.py`. Here is a typical workflow:

1.  **Ingest and Chunk Documents:**
    ```bash
    python -m app.cli ingest /path/to/your/project --chunk
    ```
    This command will ingest all the documents in the specified project directory and chunk them according to the rules in `configs/chunk_rules.yaml`.

2.  **Generate Embeddings:**
    ```bash
    python -m app.cli embed /path/to/your/project
    ```
    This will generate vector embeddings for all the chunked documents and store them in a local FAISS index.

3.  **Ask a Question:**
    ```bash
    python -m app.cli ask /path/to/your/project "Your question here"
    ```
    The platform will retrieve the most relevant context from your documents and use it to generate a comprehensive answer.

For more detailed information on the available commands and their options, please refer to the `app/README.md` file.

---

## Core Concepts

The platform is built around a modular pipeline that processes your data in several stages:

1.  **Ingestion**: The first step is to ingest your raw documents. The platform provides a suite of loaders that can handle a wide variety of file formats.
2.  **Chunking**: Once ingested, the documents are split into smaller, more manageable chunks. This process is highly configurable and can be tailored to the specific characteristics of each document type.
3.  **Enrichment**: The platform includes an `ImageInsightAgent` that can analyze images and generate textual descriptions for them. This makes visual content searchable and adds another layer of context to your knowledge base.
4.  **Embedding**: The text and image chunks are then converted into numerical representations (embeddings) using a chosen embedding model.
5.  **Indexing**: The embeddings are stored in a local FAISS index, which allows for efficient similarity searches.
6.  **Retrieval**: When you ask a question, the platform uses a late-fusion retrieval strategy to find the most relevant text and image chunks from the index.
7.  **Generation**: The retrieved context is then used to construct a detailed prompt, which is sent to a large language model to generate a final answer.

---

## 🗂️ Project Structure

The project is organized into the following key directories:

-   **`app/`**: Contains the command-line interface for the platform. See `app/README.md`.
-   **`assets/`**: A place for static assets. See `assets/README.md`.
-   **`configs/`**: Home to the `chunk_rules.yaml` file, which defines the chunking strategies for different document types. See `configs/README.md`.
-   **`docs/`**: Contains project-related documentation, including architecture diagrams and planning documents. See `docs/README.md`.
-   **`scripts/`**: The heart of the platform, containing the core logic for ingestion, chunking, embedding, retrieval, and more. See `scripts/README.md` for a high-level overview.
-   **`tests/`**: Contains the test suite for the project.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
