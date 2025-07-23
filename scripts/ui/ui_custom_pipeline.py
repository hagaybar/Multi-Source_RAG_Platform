# scripts/ui/ui_custom_pipeline.py

import streamlit as st
from pathlib import Path
from scripts.core.project_manager import ProjectManager

def render_custom_pipeline_tab():
    st.header("🧪 Custom Pipeline Runner")
    st.info("Run selected steps of the RAG pipeline with safety checks.")

    # Step 1: Project selector
    base_path = Path("data/projects")
    if not base_path.exists():
        st.warning("No projects folder found.")
        return

    all_projects = [p for p in base_path.iterdir() if p.is_dir()]
    if not all_projects:
        st.warning("No projects available.")
        return

    project_names = [p.name for p in all_projects]
    selected_name = st.selectbox("Select a project", project_names)
    project_path = base_path / selected_name
    project = ProjectManager(project_path)

    st.markdown("---")

    # Step 2: Step selection
    st.markdown("### ✅ Select Steps to Run")
    step_choices = [
        "ingest", "chunk", "enrich", "index_images",
        "embed", "retrieve", "ask"
    ]
    selected_steps = st.multiselect("Pipeline Steps", step_choices, default=[])

    # Step 3: Optional query and options
    query = ""
    if "retrieve" in selected_steps or "ask" in selected_steps:
        query = st.text_input("Query (required for retrieve/ask):", "")

    st.markdown("---")
    if st.button("🚀 Run Selected Steps"):
        st.write("✅ Steps selected:", selected_steps)
        st.write("Query:", query)
        st.write("🔧 [Validation and execution will follow in next steps]")
