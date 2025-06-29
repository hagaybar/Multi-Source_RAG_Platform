import streamlit as st

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
    st.subheader("Run a CLI Action")
    st.write("(Trigger buttons for each CLI command will be placed here.)")

# Section: Utilities / Tools
elif section == "Utilities / Tools":
    st.header("🧰 Utilities / Tools")
    st.info("Access debugging tools, FAISS index viewer, or embedding inspection.")
    st.subheader("Debugging & Visualization Tools")
    st.write("(Placeholder for utilities like chunk preview, log inspection, etc.)")
