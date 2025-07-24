import sys
from pathlib import Path

# Ensure the project root (where pyproject.toml lives) is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


import streamlit as st
import os
from scripts.interface.ask_interface import run_ask
from scripts.ui.ui_project_manager import render_project_creation, render_config_editor_v2, render_raw_data_upload, render_raw_file_viewer
from scripts.ui.ui_custom_pipeline import render_custom_pipeline_tab



st.set_page_config(page_title="RAG-GP UI", layout="wide")

st.title("📘 RAG-GP Streamlit Interface")

# Define tab options once
TAB_OPTIONS = ["Projects", "Data", "Pipeline Actions", "Utilities / Tools"]

# Ensure session state key exists on first load
if "section" not in st.session_state:
    st.session_state.section = "Projects"

# Bind the radio to session state using the `key`
section = st.sidebar.radio("Navigation", TAB_OPTIONS, key="section")

# Section: Projects
if section == "Projects":
    st.header("🔧 Projects")
    st.info("Here you will be able to create or configure a project.")

    base_path = Path("data/projects")
    if not base_path.exists():
        base_path.mkdir(parents=True)

    all_projects = [p.name for p in base_path.iterdir() if p.is_dir()]

    if "selected_project" not in st.session_state:
        st.session_state.selected_project = all_projects[0] if all_projects else None

    def on_project_change():
        st.session_state.selected_project = st.session_state.project_selector

    if not all_projects:
        st.warning("No projects found. Please create a new project to begin.")
        render_project_creation()
    else:
        selected_project_name = st.selectbox(
            "Select a Project",
            all_projects,
            key="project_selector",
            on_change=on_project_change,
            index=all_projects.index(st.session_state.selected_project) if st.session_state.selected_project in all_projects else 0
        )

        if selected_project_name:
            project_path = base_path / selected_project_name
            st.subheader(f"Project: {selected_project_name}")

            render_config_editor_v2(project_path) # replace with render_config_editor if you want the old version
            st.markdown("---")
            render_raw_data_upload(project_path)
            st.markdown("---")
            render_raw_file_viewer(project_path)

        st.markdown("---")
        render_project_creation()

# Section: Data
elif section == "Data":
    st.header("📂 Data")
    st.info("This section will allow you to manage raw and processed project data.")
    st.subheader("Ingested Files")
    st.write("(List of raw documents, upload options, etc. will be shown here.)")

# Section: Pipeline Actions
elif section == "Pipeline Actions":
    render_custom_pipeline_tab()

# Section: Utilities / Tools
elif section == "Utilities / Tools":
    st.header("🧰 Utilities / Tools")
    st.info("Access debugging tools, FAISS index viewer, or embedding inspection.")
    st.subheader("Debugging & Visualization Tools")
    st.write("(Placeholder for utilities like chunk preview, log inspection, etc.)")
