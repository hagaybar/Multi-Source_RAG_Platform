from pathlib import Path
import yaml
import streamlit as st
import os


from scripts.core.project_manager import ProjectManager

def render_project_creation():
    """
    Renders the UI for creating a new RAG-GP project.

    This function displays a Streamlit form with input fields for the project
    name, description, language, image enrichment flag, and embedding model.
    When the form is submitted, it calls the `ProjectManager.create_project`
    method to create the project directory structure and a default `config.yml`
    file.
    """
    st.subheader("Create New Project")
    with st.form("create_project_form"):
        project_name = st.text_input("Project Name")
        project_description = st.text_area("Project Description (Optional)")
        language = st.selectbox("Language", ["en", "he", "multi"])
        image_enrichment = st.checkbox("Enable Image Enrichment")
        embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-large","text-embedding-ada-002", "bge-large-en-v1.5"])

        submitted = st.form_submit_button("Create Project")
        if submitted:
            if not project_name.strip():
                st.error("Project Name cannot be empty.")
            else:
                try:
                    projects_base_dir = Path("data/projects")
                    ProjectManager.create_project(
                        project_name=project_name,
                        project_description=project_description,
                        language=language,
                        image_enrichment=image_enrichment,
                        embedding_model=embedding_model,
                        projects_base_dir=projects_base_dir,
                    )
                    st.success(f"Project '{project_name}' created successfully!")
                    st.rerun()
                except FileExistsError as e:
                    st.error(e)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

def render_config_editor(project_path: Path):
    """
    Renders the UI for editing a project's configuration file.

    This function displays a text area with the contents of the project's
    `config.yml` file. The user can edit the configuration and save it by
    clicking the "Save Config" button.

    Args:
        project_path: The path to the project's root directory.
    """
    st.subheader("Configuration Editor")
    config_path = project_path / "config.yml"

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_text = f.read()

        with st.form("config_editor_form"):
            edited_config = st.text_area(
                "Edit config.yml",
                value=config_text,
                height=400,
                key=f"config_editor_{project_path.name}",
            )
            submitted = st.form_submit_button("Save Config")
            if submitted:
                try:
                    import yaml
                    # Validate YAML syntax
                    yaml.safe_load(edited_config)
                    with config_path.open("w", encoding="utf-8") as f:
                        f.write(edited_config)
                    st.success("Configuration saved successfully!")
                except yaml.YAMLError as e:
                    st.error(f"Invalid YAML format: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    except FileNotFoundError:
        st.error("config.yml not found for this project.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the config: {e}")

def render_raw_data_upload(project_path: Path):
    """
    Renders the UI for uploading raw data files to a project.

    This function displays a file uploader that allows the user to upload
    multiple files of various types. The uploaded files are saved to the
    appropriate subdirectory under the project's `input/raw` directory.

    Args:
        project_path: The path to the project's root directory.
    """
    st.subheader("Upload Raw Data")

    file_types = ["pdf", "docx", "pptx", "xlsx", "txt", "eml", "html"]

    uploaded_files = st.file_uploader(
        "Upload files",
        type=file_types,
        accept_multiple_files=True,
        key=f"file_uploader_{project_path.name}",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = Path(uploaded_file.name).suffix.lower().replace(".", "")

            if file_extension in file_types:
                save_dir = project_path / "input" / "raw" / file_extension
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / uploaded_file.name

                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved {uploaded_file.name} to {save_dir.relative_to(project_path)}")
                except Exception as e:
                    st.error(f"Error saving {uploaded_file.name}: {e}")
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")

def render_raw_file_viewer(project_path: Path):
    """
    Renders the UI for viewing the raw data files in a project.

    This function displays a list of the raw data files in the project's
    `input/raw` directory, grouped by file type. Each file is displayed
    with its name and size.

    Args:
        project_path: The path to the project's root directory.
    """
    st.subheader("Raw File Repository")
    raw_data_path = project_path / "input" / "raw"

    if not raw_data_path.exists():
        st.info("No raw data found for this project.")
        return

    file_types = [d.name for d in raw_data_path.iterdir() if d.is_dir()]

    if not file_types:
        st.info("No raw data found for this project.")
        return

    for file_type in file_types:
        with st.expander(f"{file_type.upper()} Files"):
            type_dir = raw_data_path / file_type
            files = [f for f in type_dir.iterdir() if f.is_file()]
            if not files:
                st.write("No files of this type.")
                continue

            for f in files:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f.name)
                with col2:
                    st.write(f"{f.stat().st_size / 1024:.2f} KB")

from pathlib import Path
import yaml
import streamlit as st

def render_config_editor_v2(project_path: Path):
    """
    Enhanced config editor with structured fields, tooltips, and YAML fallback.
    """

    st.subheader("Configuration Editor")
    config_path = project_path / "config.yml"

    # Load config safely
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        st.error("config.yml not found for this project.")
        return
    except yaml.YAMLError as e:
        st.error(f"Invalid YAML in config: {e}")
        return

    st.markdown("Use this form to review or edit project settings. These control ingestion, embedding, and enrichment behavior.")

    with st.form("config_form"):
        st.markdown("### 🧾 Project Info")
        name = st.text_input("Project Name", value=config_data.get("name", project_path.name), disabled=True)
        description = st.text_area("Description", value=config_data.get("description", ""), help="Optional. Short notes about the project purpose or content.")

        st.markdown("### 🌍 Language & Embedding")
        language = st.selectbox(
            "Language",
            options=["en", "he", "multi"],
            index=["en", "he", "multi"].index(config_data.get("language", "en")),
            help="Language of documents. Affects chunking and translation behavior."
        )

        embedding_model = st.selectbox(
            "Embedding Model",
            options=["text-embedding-3-large", "text-embedding-ada-002", "bge-large-en-v1.5"],
            index=["text-embedding-3-large", "text-embedding-ada-002", "bge-large-en-v1.5"].index(
                config_data.get("embedding_model", "text-embedding-3-large")
            ),
            help="Model used to convert text into embeddings for search. Local models avoid API cost."
        )

        st.markdown("### ⚙️ Features & Flags")
        image_enrichment = st.checkbox(
            "Enable Image Enrichment",
            value=config_data.get("image_enrichment", False),
            help="If enabled, OCR will be extracted from screenshots and included in chunk text."
        )

        web_ingestion = st.checkbox(
            "Enable Web Ingestion",
            value=config_data.get("web_ingestion", False),
            help="Allows the project to ingest URLs and raw HTML files using readability parsing."
        )

        default_chunker = st.text_input(
            "Default Chunker",
            value=config_data.get("default_chunker", "PlainTextSentences"),
            help="Fallback chunking strategy (class name or alias). Used when no format-specific rule is set."
        )

        enable_logging = st.checkbox(
            "Enable Logging",
            value=config_data.get("enable_logging", True),
            help="If checked, detailed logs will be saved during pipeline steps."
        )

        submitted = st.form_submit_button("Save Config")
        if submitted:
            try:
                updated_config = {
                    "name": name,
                    "description": description,
                    "language": language,
                    "embedding_model": embedding_model,
                    "image_enrichment": image_enrichment,
                    "web_ingestion": web_ingestion,
                    "default_chunker": default_chunker,
                    "enable_logging": enable_logging
                }

                with config_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(updated_config, f, sort_keys=False)

                st.success("✅ Configuration saved successfully!")
            except Exception as e:
                st.error(f"Error saving config: {e}")

    # Optional advanced YAML view
    with st.expander("🛠 Advanced: Edit Raw YAML"):
        raw_yaml = yaml.safe_dump(config_data, sort_keys=False)
        edited_yaml = st.text_area("Raw config.yml", value=raw_yaml, height=300, key=f"raw_yaml_{project_path.name}")

        if st.button("Save Raw YAML", key=f"save_raw_{project_path.name}"):
            try:
                parsed_yaml = yaml.safe_load(edited_yaml)
                with config_path.open("w", encoding="utf-8") as f:
                    f.write(edited_yaml)
                st.success("Raw YAML saved successfully!")
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML format: {e}")
