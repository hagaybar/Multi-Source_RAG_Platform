# RAG‑Pipeline Prototype

A **local‑first Retrieval‑Augmented‑Generation platform** for turning mixed organisational content (PDFs, office docs, presentations, emails, images, websites, etc.) into an interactive knowledge base.  
Built for librarians, researchers and other knowledge‑workers who need reliable answers from their in‑house documentation without sending private data to external services.

<p align="center">
  <img src="https://raw.githubusercontent.com/your‑org/rag‑pipeline/main/docs/architecture_simplified.png" width="600" alt="High‑level architecture"/>
</p>

---

## ✨ Key Features (v0.1)

* **Drag‑and‑drop ingestion** for TXT, PDF, PPTX, DOCX, XLSX/CSV, images, email files and public URLs (up to **1 GB** total).
* **Source‑aware chunking** – pluggable rule‑sets per file type.
* **Hybrid embeddings** – choose local HuggingFace models *or* OpenAI API with one click.
* **FAISS vector search** with late‑fusion across source types.
* **Streamlit UI** – create projects, upload data, run queries and view logs without touching the CLI.
* **Answer generation** with citations using your preferred chat‑LLM endpoint.
* 100 % offline data storage – raw files, vectors and logs stay on your machine.

See the full [Roadmap](docs/rag_prototype_roadmap.md) for detailed design and future milestones.

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/your‑org/rag‑pipeline.git
cd rag‑pipeline
poetry install          # or: pip install -r requirements.txt

# 2. Launch Streamlit UI
poetry run streamlit run app/ui_streamlit.py   # default browser opens

# 3. Create a new project in the UI, upload some PDFs, and ask a question!
```

> **Tip:** Prefer local embeddings? Select **bge‑large‑en** under *Settings → Embeddings* before indexing.

---

## 💻 UI Usage

The Streamlit UI provides a user-friendly interface for managing RAG-GP projects.

### Creating a New Project

1.  Navigate to the "Projects" section in the sidebar.
2.  Fill out the "Create New Project" form:
    *   **Project Name:** A unique name for your project.
    *   **Project Description:** An optional description of your project.
    *   **Language:** The primary language of your documents.
    *   **Enable Image Enrichment:** Check this box to enable image analysis features.
    *   **Embedding Model:** Select the embedding model to use for your project.
3.  Click the "Create Project" button.

### Managing a Project

Once you have created a project, you can manage it from the "Projects" section.

*   **Select a Project:** Choose a project from the dropdown menu to view its details.
*   **Configuration Editor:** The `config.yml` file for the selected project is displayed in a text editor. You can make changes to the configuration and save them by clicking the "Save Config" button.
*   **Upload Raw Data:** You can upload raw data files (e.g., .pdf, .docx, .txt) to your project using the file uploader. The files will be saved to the appropriate subdirectory under `data/projects/<project_name>/input/raw/`.
*   **Raw File Repository:** The "Raw File Repository" section displays a list of all the raw data files in your project, grouped by file type.

---

## 🗂️ Folder Structure (excerpt)

```text
rag‑pipeline/
├── app/                 # CLI & Streamlit entry‑points
├── scripts/             # Core library (ingestion, chunking, embeddings…)
├── configs/             # YAML templates for datasets & tasks
├── data/                # Your local datasets, chunks & indexes (git‑ignored)
├── docs/                # Technical docs & design diagrams
└── tests/               # Pytest suite
```

For a full tree and design conventions, check the [Codebase Structure](docs/rag_prototype_roadmap.md#10  repository--codebase-structure).

---

## 🛠️ Requirements

* **Python 3.12**
* **Tesseract OCR** (optional, for image ingestion) – install via your OS package manager.
* For OpenAI or other API models: set `OPENAI_API_KEY` or relevant environment variables.

---

## 🤝 Contributing

Pull requests are welcome! Please read **CONTRIBUTING.md** (to be added) and open an issue before starting major work.

---

## 📜 License

This project is released under the MIT License © 2025 — see [LICENSE](LICENSE) for details.

---

> *Built with ❤️ and lots of coffee by the Library Innovation Lab.*  
> *“Organisational knowledge belongs at your fingertips.”*
