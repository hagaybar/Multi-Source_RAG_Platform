# scripts/ui/ui_custom_pipeline.py

import streamlit as st
from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.ui.validation_helpers import validate_steps
from scripts.pipeline.runner import PipelineRunner



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

    st.write("Loaded project at:", project.root_dir)
    st.markdown("==== testing 123 ====")


    st.markdown("---")

    # Step 2: Step selection
    st.markdown("### ✅ Select Steps to Run")
    # Derive step names from PipelineRunner.step_<name> methods
    step_methods = [m for m in dir(PipelineRunner) if m.startswith("step_")]
    raw_steps   = [m.replace("step_", "") for m in step_methods]

    logical_order = [
    "ingest",
    "chunk",
    "embed",
    "enrich",
    "index",
    "index_images",
    "retrieve",
    "ask",
    ]
    ordered = []
    # include steps in your preferred order, if they exist
    for name in logical_order:
        if name in raw_steps:
            ordered.append(name)
    
    # then append any extra steps you didn’t anticipate, alphabetically
    extras = sorted(s for s in raw_steps if s not in ordered)
    step_choices = ordered + extras


    selected_steps = st.multiselect(
    "Pipeline Steps", 
    step_choices, 
    default=[])
    # DEBUG: inspect what the multiselect is returning
    st.write("🔍 Debug – selected_steps:", selected_steps)

    # Step 3: Optional query and options
    query = ""
    if any(step in ("retrieve", "ask") for step in selected_steps):
        query = st.text_input("Query (required for retrieve/ask):", "")

    st.markdown("---")
    # if st.button("🚀 Run Selected Steps"):
    #     st.write("✅ Steps selected:", selected_steps)
    #     st.write("Query:", query)
    #     st.write("🔧 [Validation and execution will follow in next steps]")
    run_disabled = not selected_steps
    if st.button("🚀 Run Selected Steps", disabled=run_disabled):
        unique_steps = list(dict.fromkeys(selected_steps))
        st.write("▶️ Unique steps to run:", unique_steps) 

        # 1️⃣ Validate selection
        is_valid, messages = validate_steps(project, unique_steps, query)
        if not is_valid:
            for msg in messages:
                st.error(msg)  # show errors  
            st.stop()
        # If valid, any messages are warnings
        for msg in messages:
            st.warning(msg)

        # 2️⃣ Instantiate the runner
        runner = PipelineRunner(project, project.config)

        # 3️⃣ Register each step
        for step in selected_steps:
            if step in ("retrieve", "ask"):
                runner.add_step(step, query=query)
            else:
                # st.write("▶️ About to register these steps:", selected_steps)
                runner.add_step(step)
        # Note: add_step(name, **kwargs) queues up your steps :contentReference[oaicite:2]{index=2}

        # 4️⃣ Execute & stream logs
        log_area  = st.empty()
        log_lines = []  # ← buffer to accumulate

        with st.spinner("Running pipeline…"):
            for line in runner.run_steps():
                log_lines.append(line)
                # re-render the full log each iteration
                log_area.text("\n".join(log_lines))

        st.success("🎉 Pipeline complete!")
    if run_disabled:
        st.info("Select at least one pipeline step to enable execution.")