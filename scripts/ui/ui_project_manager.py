from pathlib import Path
import yaml
import streamlit as st
from scripts.core.project_manager import ProjectManager


def render_project_creation():
    """
    Renders the UI for creating a new RAG-GP project with validation and better
    error handling.
    """
    st.subheader("Create New Project")

    with st.form("create_project_form"):
        project_name = st.text_input(
            "Project Name", help="Enter a unique name for your project"
        )
        project_description = st.text_area(
            "Project Description (Optional)",
            help="Brief description of the project's purpose"
        )
        language = st.selectbox(
            "Language",
            ["en", "he", "multi"],
            help="Primary language of your documents"
        )
        image_enrichment = st.checkbox(
            "Enable Image Enrichment",
            help="Extract text from images and screenshots"
        )
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "text-embedding-3-large",
                "text-embedding-ada-002",
                "bge-large-en-v1.5"
            ],
            help="Model used to convert text into embeddings for search",
        )

        submitted = st.form_submit_button("Create Project")

        if submitted:
            # Input validation
            validation_errors = []

            if not project_name.strip():
                validation_errors.append("Project Name cannot be empty")
            elif len(project_name.strip()) < 2:
                validation_errors.append(
                    "Project Name must be at least 2 characters long"
                )
            elif len(project_name.strip()) > 50:
                validation_errors.append(
                    "Project Name must be less than 50 characters"
                )

            # Check for invalid characters
            import re

            if not re.match(r'^[a-zA-Z0-9_\-\s]+$', project_name.strip()):
                validation_errors.append(
                    "Project Name can only contain letters, numbers, spaces, "
                    "hyphens, and underscores"
                )

            # Show validation errors
            if validation_errors:
                for error in validation_errors:
                    st.error(f"❌ {error}")
                return

            # Show progress
            with st.spinner("Creating project..."):
                try:
                    projects_base_dir = Path("data/projects")
                    projects_base_dir.mkdir(parents=True, exist_ok=True)

                    # Create the project
                    project_root = ProjectManager.create_project(
                        project_name=project_name.strip(),
                        project_description=project_description.strip(),
                        language=language,
                        image_enrichment=image_enrichment,
                        embedding_model=embedding_model,
                        projects_base_dir=projects_base_dir,
                    )

                    # Validate the created project
                    config_path = project_root / "config.yml"
                    is_valid, config_errors = (
                        ProjectManager.validate_config_file(config_path)
                    )

                    if not is_valid:
                        # If validation fails, show errors but don't delete the project
                        # (user might be able to fix it manually)
                        st.error("⚠️ Project created but configuration has issues:")
                        for error in config_errors:
                            st.error(f"• {error}")
                        st.info(f"Project location: `{project_root}`")
                        st.info(
                            "You can fix these issues using the Configuration Editor."
                        )
                    else:
                        # Success!
                        st.success(f"✅ Project '{project_name}' created successfully!")

                        # Show project details
                        st.info(f"📁 **Project Location:** `{project_root}`")

                        # Show next steps
                        with st.expander("📋 Next Steps"):
                            st.markdown("""
                            **Your project is ready! Here's what you can do next:**
                            
                            1. **Upload Data**: Use the "Data" tab to upload your 
                               documents
                            2. **Configure Settings**: Fine-tune your project settings 
                               if needed
                            3. **Run Pipeline**: Use "Pipeline Actions" to process your 
                               documents
                            
                            **Project Structure Created:**
                            - `input/raw/` - Upload your documents here
                            - `output/` - Processed data and results
                            - `config.yml` - Project configuration
                            """)

                        # Auto-refresh to show the new project in the selector
                        st.rerun()

                except FileExistsError as e:
                    st.error(f"❌ {e}")
                    st.info(
                        "💡 Try a different project name or delete the existing "
                        "project first."
                    )

                except PermissionError:
                    st.error(
                        "❌ Permission denied. Check that you have write access to "
                        "the projects directory."
                    )

                except OSError as e:
                    st.error(f"❌ File system error: {e}")
                    st.info(
                        "💡 This might be due to insufficient disk space or invalid "
                        "characters in the project name."
                    )

                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")
                    st.info(
                        "💡 Please check the logs or try again with different settings."
                    )

                    # Show technical details in an expander for debugging
                    with st.expander("🔧 Technical Details"):
                        st.code(str(e))
                        import traceback

                        st.code(traceback.format_exc())


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
                    st.success(
                        f"Saved {uploaded_file.name} to "
                        f"{save_dir.relative_to(project_path)}"
                    )
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


def render_config_editor_v2(project_path: Path):
    """
    Enhanced config editor with validation and better error handling.
    """

    st.subheader("Configuration Editor")
    config_path = project_path / "config.yml"

    # Load and validate config on startup
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        st.error("❌ config.yml not found for this project.")
        return
    except yaml.YAMLError as e:
        st.error(f"❌ Invalid YAML in config: {e}")
        return

    # Validate current config and show status
    is_valid, validation_errors = ProjectManager.validate_config(config_data)

    if not is_valid:
        st.warning("⚠️ Current configuration has issues:")
        for error in validation_errors:
            st.warning(f"• {error}")
        st.info("💡 Fix these issues using the form below or the raw YAML editor.")
    else:
        st.success("✅ Configuration is valid")

    st.markdown(
        "Use this form to review or edit project settings. These control "
        "ingestion, embedding, and enrichment behavior."
    )

    with st.form("config_form"):
        st.markdown("### 🧾 Project Info")

        # Access nested project fields correctly
        project_section = config_data.get("project", {})
        name = st.text_input(
            "Project Name",
            value=project_section.get("name", project_path.name),
            disabled=True,
            help="Project name cannot be changed after creation",
        )
        description = st.text_area(
            "Description",
            value=project_section.get("description", ""),
            help="Optional. Short notes about the project purpose or content.",
        )

        st.markdown("### 🌍 Language & Embedding")

        language_options = ["en", "he", "multi"]
        current_language = project_section.get("language", "en")
        language_index = (
            language_options.index(current_language)
            if current_language in language_options
            else 0
        )

        language = st.selectbox(
            "Language",
            options=language_options,
            index=language_index,
            help="Language of documents. Affects chunking and translation behavior.",
        )

        # Access nested embedding fields correctly
        embedding_section = config_data.get("embedding", {})
        embedding_options = [
            "text-embedding-3-large",
            "text-embedding-ada-002",
            "bge-large-en-v1.5",
        ]
        current_model = embedding_section.get("model", "text-embedding-3-large")
        model_index = (
            embedding_options.index(current_model)
            if current_model in embedding_options
            else 0
        )

        embedding_model = st.selectbox(
            "Embedding Model",
            options=embedding_options,
            index=model_index,
            help=(
                "Model used to convert text into embeddings for search. "
                "Local models avoid API cost."
            ),
        )

        st.markdown("### ⚙️ Features & Flags")

        image_enrichment = st.checkbox(
            "Enable Image Enrichment",
            value=embedding_section.get("image_enrichment", False),
            help=(
                "If enabled, OCR will be extracted from screenshots and included "
                "in chunk text."
            ),
        )

        # Advanced settings in an expander
        with st.expander("🔧 Advanced Settings"):
            llm_section = config_data.get("llm", {})

            temperature = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(llm_section.get("temperature", 0.4)),
                step=0.1,
                help=(
                    "Controls randomness in LLM responses. Lower = more focused, "
                    "Higher = more creative"
                ),
            )

            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=4000,
                value=int(llm_section.get("max_tokens", 400)),
                help="Maximum number of tokens in LLM responses",
            )

            batch_size = st.number_input(
                "Embedding Batch Size",
                min_value=1,
                max_value=1000,
                value=int(embedding_section.get("embed_batch_size", 64)),
                help=(
                    "Number of texts to embed in each batch. Higher = faster but "
                    "more memory usage"
                ),
            )

        submitted = st.form_submit_button("💾 Save Configuration")

        if submitted:
            with st.spinner("Validating and saving configuration..."):
                try:
                    # Create updated config maintaining the nested structure
                    updated_config = config_data.copy()  # Start with existing config

                    # Update project section
                    if "project" not in updated_config:
                        updated_config["project"] = {}
                    updated_config["project"]["name"] = name
                    updated_config["project"]["description"] = description
                    updated_config["project"]["language"] = language

                    # Update embedding section
                    if "embedding" not in updated_config:
                        updated_config["embedding"] = {}
                    updated_config["embedding"]["model"] = embedding_model
                    updated_config["embedding"]["image_enrichment"] = image_enrichment
                    updated_config["embedding"]["embed_batch_size"] = batch_size

                    # Update LLM section
                    if "llm" not in updated_config:
                        updated_config["llm"] = {}
                    updated_config["llm"]["temperature"] = temperature
                    updated_config["llm"]["max_tokens"] = max_tokens

                    # Update agents section to match image_enrichment
                    if "agents" not in updated_config:
                        updated_config["agents"] = {}
                    updated_config["agents"]["enable_image_insight"] = image_enrichment

                    # VALIDATE BEFORE SAVING
                    is_valid, errors = ProjectManager.validate_config(updated_config)

                    if not is_valid:
                        st.error("❌ Configuration validation failed:")
                        for error in errors:
                            st.error(f"• {error}")
                        st.info("💡 Please fix the errors above before saving.")
                        return

                    # Save the validated config
                    with config_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(
                            updated_config, f, sort_keys=False, default_flow_style=False
                        )

                    st.success("✅ Configuration saved and validated successfully!")
                    st.rerun()  # Refresh to show updated validation status

                except Exception as e:
                    st.error(f"❌ Error saving configuration: {e}")

                    with st.expander("🔧 Technical Details"):
                        st.code(str(e))

    # Raw YAML editor with validation
    with st.expander("🛠 Advanced: Edit Raw YAML"):
        st.info(
            "⚠️ Direct YAML editing bypasses form validation. Use with caution."
        )

        raw_yaml = yaml.safe_dump(
            config_data, sort_keys=False, default_flow_style=False
        )
        edited_yaml = st.text_area(
            "Raw config.yml",
            value=raw_yaml,
            height=300,
            key=f"raw_yaml_{project_path.name}",
            help="Advanced users can edit the raw YAML directly",
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button(
                "🔍 Validate YAML", key=f"validate_yaml_{project_path.name}"
            ):
                try:
                    parsed_yaml = yaml.safe_load(edited_yaml)
                    is_valid, errors = ProjectManager.validate_config(parsed_yaml)

                    if is_valid:
                        st.success("✅ YAML is valid!")
                    else:
                        st.error("❌ YAML validation failed:")
                        for error in errors:
                            st.error(f"• {error}")

                except yaml.YAMLError as e:
                    st.error(f"❌ Invalid YAML syntax: {e}")

        with col2:
            if st.button(
                "💾 Save Raw YAML", key=f"save_raw_{project_path.name}"
            ):
                try:
                    # Validate YAML syntax first
                    parsed_yaml = yaml.safe_load(edited_yaml)

                    # Then validate config structure
                    is_valid, errors = ProjectManager.validate_config(parsed_yaml)

                    if not is_valid:
                        st.error("❌ Cannot save invalid configuration:")
                        for error in errors:
                            st.error(f"• {error}")
                        return

                    # Save the validated YAML
                    with config_path.open("w", encoding="utf-8") as f:
                        f.write(edited_yaml)

                    st.success("✅ Raw YAML saved successfully!")
                    st.rerun()  # Refresh to show updated values in form

                except yaml.YAMLError as e:
                    st.error(f"❌ Invalid YAML format: {e}")
                except Exception as e:
                    st.error(f"❌ Error saving raw YAML: {e}")
