import streamlit as st
import os
from pathlib import Path

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
    st.info("Here you will be able to create or configure a project.")
    st.subheader("Project Configuration")
    st.write("(Form for editing YAML config, selecting paths, etc. goes here.)")

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

    # Scan projects folder for valid projects
    base_path = Path("data/projects")
    all_projects = [p for p in base_path.iterdir() if p.is_dir()]

    valid_projects = []
    disabled_projects = []
    for proj in all_projects:
        config_exists = (proj / "config.yml").exists()
        input_folder = proj / "input"
        output_folder = proj / "output"
        chunks_exist = list(input_folder.glob("chunks_*.tsv"))
        index_exist = list(output_folder.glob("*.index"))
        jsonl_exist = list(output_folder.glob("*.jsonl"))

        if config_exists and chunks_exist and index_exist and jsonl_exist:
            valid_projects.append(proj.name)
        else:
            disabled_projects.append(proj.name)

    st.write("Select a project to ask a question:")
    selected_project = st.selectbox(
        "Project",
        options=valid_projects + disabled_projects,
        index=0 if valid_projects else None,
        format_func=lambda x: f"{x} (invalid)" if x in disabled_projects else x,
        disabled_options=disabled_projects if hasattr(st, "selectbox") else None
    )

    if selected_project in valid_projects:
        question = st.text_input("Enter your question:")
        if st.button("Ask"):
            st.success(f"Running ask() on project '{selected_project}' with question: {question}")
            st.markdown("---")
            st.markdown("### 🤖 Answer")
            st.write("(Answer from LLM will appear here.)")
            st.markdown("### 📄 Sources")
            st.write("- source_1.pdf, page 3\n- source_2.txt")

# Section: Utilities / Tools
elif section == "Utilities / Tools":
    st.header("🧰 Utilities / Tools")
    st.info("Access debugging tools, FAISS index viewer, or embedding inspection.")
    st.subheader("Debugging & Visualization Tools")
    st.write("(Placeholder for utilities like chunk preview, log inspection, etc.)")
