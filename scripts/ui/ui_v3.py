import streamlit as st
import os
from pathlib import Path
import yaml
from scripts.interface.ask_interface import run_ask
from scripts.core.project_manager import ProjectManager

st.set_page_config(page_title="RAG-GP UI", layout="wide")

st.title("📘 RAG-GP Streamlit Interface")

# Sidebar navigation
section = st.sidebar.radio(
    "Navigation",
    [
        "Projects",
        "Data",
        "Pipeline Actions",
        "Utilities / Tools"
    ]
)

# Section: Projects
if section == "Projects":
    st.header("🔧 Projects")
    base_path = Path("data/projects")
    base_path.mkdir(parents=True, exist_ok=True)

    st.subheader("Create New Project")
    new_name = st.text_input("Project Name")
    if st.button("Create Project") and new_name:
        project_dir = base_path / new_name
        if project_dir.exists():
            st.error("Project already exists.")
        else:
            project_dir.mkdir(parents=True)
            config = {
                "project": {"name": new_name},
                "paths": {
                    "input_dir": "input",
                    "output_dir": "output",
                    "logs_dir": "output/logs",
                    "faiss_dir": "output/faiss",
                    "metadata_dir": "output/metadata",
                },
            }
            (project_dir / "config.yml").write_text(yaml.dump(config), encoding="utf-8")
            ProjectManager(project_dir)
            st.success(f"Created project '{new_name}'")

    st.markdown("---")
    projects = [p.name for p in base_path.iterdir() if p.is_dir()]
    if projects:
        selected = st.selectbox("Select Project", projects)
        if selected:
            pm = ProjectManager(base_path / selected)
            st.subheader("Configuration")
            st.json(pm.config)

            st.subheader("Input Files")
            files = [str(p.relative_to(pm.input_dir)) for p in pm.input_dir.glob("**/*") if p.is_file()]
            if files:
                for f in files:
                    st.write(f"- {f}")
            else:
                st.write("(No input files found)")
    else:
        st.info("No projects found. Use the form above to create one.")

# Section: Data
elif section == "Data":
    st.header("📂 Data")
    st.info("This section will allow you to manage raw and processed project data.")
    st.subheader("Ingested Files")
    st.write("(List of raw documents, upload options, etc. will be shown here.)")

# Section: Pipeline Actions
elif section == "Pipeline Actions":
    st.header("🚀 Pipeline Actions")
    st.info("Run individual stages of the pipeline (ingest, chunk, embed, retrieve, ask).")
    st.subheader("Ask a Question")

    base_path = Path("data/projects")
    all_projects = [p for p in base_path.iterdir() if p.is_dir()]

    valid_projects = []
    invalid_projects = []

    for proj in all_projects:
        config_exists = (proj / "config.yml").exists()
        input_folder = proj / "input"
        output_folder = proj / "output"
        faiss_folder = output_folder / "faiss"
        chunks_exist = list(input_folder.glob("chunks_*.tsv"))
        index_exist = list(faiss_folder.glob("*.faiss")) if faiss_folder.exists() else []
        jsonl_exist = list(output_folder.glob("*.jsonl"))

        if config_exists and chunks_exist and index_exist and jsonl_exist:
            valid_projects.append(proj.name)
        else:
            missing = []
            if not config_exists:
                missing.append("config.yml")
            if not chunks_exist:
                missing.append("chunk files (chunks_*.tsv)")
            if not index_exist:
                missing.append("FAISS index files (*.faiss)")
            if not jsonl_exist:
                missing.append("JSONL files (*.jsonl)")
            invalid_projects.append((proj.name, missing))

    all_options = valid_projects + [f"{proj[0]} (invalid)" for proj in invalid_projects]

    if all_options:
        st.write("Select a project to ask a question:")
        selected_project_display = st.selectbox("Project", options=all_options)

        if selected_project_display:
            selected_project = selected_project_display.replace(" (invalid)", "")

            if selected_project in valid_projects:
                question = st.text_input("Enter your question:")
                if st.button("Ask") and question.strip():
                    with st.spinner("Asking the LLM..."):
                        answer, sources = run_ask(
                            project_path=str(base_path / selected_project),
                            query=question
                        )
                    st.markdown("---")
                    st.markdown("### 🤖 Answer")
                    st.write(answer or "(No answer returned.)")
                    st.markdown("### 📄 Sources")
                    if sources:
                        for src in sources:
                            st.write(f"- {src}")
                    else:
                        st.write("(No sources returned.)")
            else:
                missing_items = next((missing for proj_name, missing in invalid_projects if proj_name == selected_project), [])
                st.warning(f"This project is invalid. Missing: {', '.join(missing_items)}")
    else:
        st.warning("No projects found. Please create or validate at least one project.")

# Section: Utilities / Tools
elif section == "Utilities / Tools":
    st.header("🧰 Utilities / Tools")
    st.info("Access debugging tools, FAISS index viewer, or embedding inspection.")
    st.subheader("Debugging & Visualization Tools")
    st.write("(Placeholder for utilities like chunk preview, log inspection, etc.)")
