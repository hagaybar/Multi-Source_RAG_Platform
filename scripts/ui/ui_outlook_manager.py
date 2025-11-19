"""
Streamlit UI components for Outlook integration.

This module provides UI widgets for:
- Creating Outlook-based projects
- Testing Outlook connections
- Previewing emails
- Managing email ingestion
"""

import sys
from pathlib import Path
import streamlit as st
import yaml

from scripts.core.project_manager import ProjectManager
from scripts.utils.logger import LoggerManager

# Check if running on Windows
IS_WINDOWS = sys.platform == "win32"

# Try to import Outlook components
try:
    from scripts.connectors.outlook_connector import OutlookConnector, OutlookConfig, OUTLOOK_AVAILABLE
except ImportError:
    OUTLOOK_AVAILABLE = False
    OutlookConnector = None
    OutlookConfig = None


def render_outlook_requirements_check():
    """
    Display requirements check for Outlook integration.

    Shows warnings if:
    - Not running on Windows
    - pywin32 not installed
    - Outlook not installed/configured
    """
    if not IS_WINDOWS:
        st.error(
            "⚠️ Outlook integration requires Windows OS. "
            "You are currently running on: " + sys.platform
        )
        st.info(
            "💡 Alternative: Use EML/MSG/MBOX file upload instead, "
            "or export emails from your email client."
        )
        return False

    if not OUTLOOK_AVAILABLE:
        st.error(
            "⚠️ pywin32 library not installed. "
            "This is required for Outlook integration."
        )
        st.code("pip install pywin32", language="bash")
        st.info(
            "💡 After installing pywin32, restart the Streamlit app."
        )
        return False

    return True


def render_outlook_connection_test():
    """
    Test Outlook connection and show available accounts/folders.
    """
    st.subheader("🔌 Test Outlook Connection")

    # Check requirements first
    if not render_outlook_requirements_check():
        return

    if st.button("Test Connection", key="test_outlook_connection"):
        with st.spinner("Connecting to Outlook..."):
            try:
                import pythoncom

                # Initialize COM for this thread
                pythoncom.CoInitialize()

                try:
                    # Create a test config
                    test_config = OutlookConfig(
                        account_name="",  # Will list all accounts
                        folder_path="Inbox",
                        days_back=1
                    )

                    connector = OutlookConnector(test_config)
                    outlook = connector.connect_to_outlook()

                    st.success("✅ Successfully connected to Outlook!")

                    # List available accounts
                    st.markdown("### Available Accounts")
                    accounts = []
                    try:
                        for i in range(outlook.Folders.Count):
                            folder = outlook.Folders.Item(i + 1)
                            accounts.append(folder.Name)
                            st.write(f"- {folder.Name}")
                    except Exception as e:
                        st.error(f"Could not list accounts: {e}")

                    if accounts:
                        st.session_state.outlook_accounts = accounts

                    # List folders in first account (if available)
                    if accounts:
                        st.markdown(f"### Folders in '{accounts[0]}'")
                        try:
                            account_folder = outlook.Folders.Item(1)
                            for i in range(account_folder.Folders.Count):
                                folder = account_folder.Folders.Item(i + 1)
                                st.write(f"- {folder.Name}")
                        except Exception as e:
                            st.warning(f"Could not list folders: {e}")

                finally:
                    # Always cleanup COM
                    pythoncom.CoUninitialize()

            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
                st.info(
                    "💡 Ensure Microsoft Outlook is installed and configured "
                    "with at least one email account."
                )
                with st.expander("🔧 Technical Details"):
                    st.code(str(e))


def render_outlook_project_creation():
    """
    Render form for creating an Outlook-based project.
    """
    st.subheader("📧 Create Outlook Project")

    # Check requirements first
    if not render_outlook_requirements_check():
        return

    with st.form("outlook_project_form"):
        st.markdown("### Project Details")

        project_name = st.text_input(
            "Project Name",
            help="Unique name for this Outlook project"
        )

        project_description = st.text_area(
            "Description (Optional)",
            help="What emails are you indexing?"
        )

        st.markdown("### Outlook Settings")

        account_name = st.text_input(
            "Outlook Account Name",
            placeholder="e.g., user@company.com",
            help="The display name of your Outlook account (as shown in Outlook)"
        )

        folder_path = st.text_input(
            "Folder Path",
            value="Inbox",
            help="e.g., 'Inbox', 'Inbox > Subfolder', or 'Sent Items'"
        )

        days_back = st.slider(
            "Days to Look Back",
            min_value=1,
            max_value=365,
            value=30,
            help="Extract emails from the last X days"
        )

        col1, col2 = st.columns(2)
        with col1:
            max_emails = st.number_input(
                "Max Emails (Optional)",
                min_value=0,
                value=0,
                help="0 = no limit"
            )

        with col2:
            include_attachments = st.checkbox(
                "Include Attachments",
                value=False,
                help="Extract text from email attachments (future feature)",
                disabled=True  # Not yet implemented
            )

        st.markdown("### Embedding Settings")

        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-large", "text-embedding-ada-002", "bge-large-en-v1.5"],
            help="Model used to convert text into embeddings for search"
        )

        submitted = st.form_submit_button("📧 Create Outlook Project")

        if submitted:
            # Validation
            validation_errors = []

            if not project_name.strip():
                validation_errors.append("Project Name cannot be empty")
            if not account_name.strip():
                validation_errors.append("Outlook Account Name cannot be empty")
            if not folder_path.strip():
                validation_errors.append("Folder Path cannot be empty")

            if validation_errors:
                for error in validation_errors:
                    st.error(f"❌ {error}")
                return

            # Create project with Outlook configuration
            with st.spinner("Creating Outlook project..."):
                try:
                    projects_base_dir = Path("data/projects")
                    projects_base_dir.mkdir(parents=True, exist_ok=True)

                    # Create project
                    project_root = ProjectManager.create_project(
                        project_name=project_name.strip(),
                        project_description=project_description.strip(),
                        language="en",  # Default for emails
                        image_enrichment=False,  # Not needed for emails
                        embedding_model=embedding_model,
                        projects_base_dir=projects_base_dir,
                    )

                    # Add Outlook-specific configuration
                    config_path = project_root / "config.yml"
                    with config_path.open("r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)

                    # Add sources section with Outlook configuration
                    config["sources"] = {
                        "outlook": {
                            "enabled": True,
                            "account_name": account_name.strip(),
                            "folder_path": folder_path.strip(),
                            "days_back": days_back,
                            "max_emails": max_emails if max_emails > 0 else None,
                            "include_attachments": include_attachments
                        }
                    }

                    # Save updated config
                    with config_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

                    st.success(f"✅ Outlook project '{project_name}' created successfully!")

                    # Show project details
                    st.info(f"📁 **Project Location:** `{project_root}`")

                    # Update session state to select this project
                    st.session_state.selected_project = project_name.strip()

                    # Show next steps
                    with st.expander("📋 Next Steps"):
                        st.markdown("""
                        **Your Outlook project is ready! Here's what you can do next:**

                        1. **Stay on this tab** to test connection and preview emails (after reload)
                        2. **Or go to "Projects" tab** to select this project first
                        3. **Then return here** to see Test Connection and Preview features

                        **Outlook Configuration:**
                        - Account: {account}
                        - Folder: {folder}
                        - Days Back: {days}
                        - Max Emails: {max_emails}
                        """.format(
                            account=account_name,
                            folder=folder_path,
                            days=days_back,
                            max_emails=max_emails if max_emails > 0 else "No limit"
                        ))

                    st.info("🔄 **Refreshing page to load your new project...**")

                    # Auto-refresh to show new project
                    import time
                    time.sleep(2)  # Brief pause to show success message
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error creating project: {e}")
                    with st.expander("🔧 Technical Details"):
                        st.code(str(e))


def render_outlook_email_preview(project_path: Path):
    """
    Preview emails that would be extracted with current settings.

    Args:
        project_path: Path to the project directory
    """
    st.subheader("📬 Email Preview")

    # Check requirements first
    if not render_outlook_requirements_check():
        return

    # Load Outlook config from project
    config_path = project_path / "config.yml"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        outlook_config_dict = config.get("sources", {}).get("outlook", {})

        if not outlook_config_dict.get("enabled"):
            st.warning("Outlook integration is not enabled for this project.")
            return

        # Show current settings
        st.info(f"📂 **Folder:** `{outlook_config_dict.get('folder_path')}`")
        st.info(f"📅 **Days Back:** {outlook_config_dict.get('days_back')}")
        st.info(f"📊 **Max Emails:** {outlook_config_dict.get('max_emails') or 'No limit'}")

        if st.button("🔍 Preview Emails", key="preview_outlook_emails"):
            with st.spinner("Fetching emails from Outlook..."):
                try:
                    # Note: extract_emails() already handles COM initialization internally
                    # Create Outlook config
                    outlook_config = OutlookConfig(
                        account_name=outlook_config_dict["account_name"],
                        folder_path=outlook_config_dict["folder_path"],
                        days_back=outlook_config_dict.get("days_back", 30),
                        max_emails=10  # Preview only first 10
                    )

                    connector = OutlookConnector(outlook_config)
                    emails = connector.extract_emails()

                    if not emails:
                        st.warning("⚠️ No emails found in the specified folder and date range.")
                        return

                    st.success(f"✅ Found {len(emails)} emails (showing first 10)")

                    # Display emails
                    for i, (body, meta) in enumerate(emails[:10]):
                        with st.expander(f"📧 {meta.get('subject', 'No Subject')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**From:** {meta.get('sender_name', 'Unknown')}")
                            with col2:
                                st.markdown(f"**Date:** {meta.get('date', 'Unknown')}")

                            st.markdown("**Preview:**")
                            preview_text = body[:500] + "..." if len(body) > 500 else body
                            st.text(preview_text)

                except Exception as e:
                    st.error(f"❌ Preview failed: {e}")
                    with st.expander("🔧 Technical Details"):
                        st.code(str(e))

    except Exception as e:
        st.error(f"❌ Error loading project configuration: {e}")


def render_outlook_ingestion_controls(project_path: Path):
    """
    Controls for manually triggering Outlook email ingestion.

    Args:
        project_path: Path to the project directory
    """
    st.subheader("⚙️ Outlook Ingestion")

    # Check requirements first
    if not render_outlook_requirements_check():
        return

    # Load Outlook config
    config_path = project_path / "config.yml"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        outlook_config_dict = config.get("sources", {}).get("outlook", {})

        if not outlook_config_dict.get("enabled"):
            st.warning("Outlook integration is not enabled for this project.")
            return

        # Display current config
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Account", outlook_config_dict.get("account_name", "N/A"))
        with col2:
            st.metric("Folder", outlook_config_dict.get("folder_path", "N/A"))
        with col3:
            st.metric("Days Back", outlook_config_dict.get("days_back", 30))

        # Ingestion button
        if st.button("🔄 Extract Emails from Outlook", key="extract_outlook_emails"):
            with st.spinner("Extracting emails from Outlook..."):
                try:
                    from scripts.ingestion.manager import IngestionManager

                    # Create Outlook config
                    outlook_config = OutlookConfig(
                        account_name=outlook_config_dict["account_name"],
                        folder_path=outlook_config_dict["folder_path"],
                        days_back=outlook_config_dict.get("days_back", 30),
                        max_emails=outlook_config_dict.get("max_emails")
                    )

                    # Create log directory if it doesn't exist
                    log_dir = project_path / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)

                    # Create log file path for this run
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file = log_dir / f"outlook_ingestion_{timestamp}.log"

                    # Run ingestion with proper logging
                    ingestion_mgr = IngestionManager(log_file=log_file)
                    raw_docs = ingestion_mgr.ingest_outlook(outlook_config)

                    if raw_docs:
                        # Save emails to disk as .outlook_eml files (JSON format)
                        import json
                        raw_dir = project_path / "input" / "raw" / "outlook_eml"
                        raw_dir.mkdir(parents=True, exist_ok=True)

                        # Save each email as a separate .outlook_eml file
                        for i, doc in enumerate(raw_docs, start=1):
                            email_file = raw_dir / f"email_{i:03d}.outlook_eml"
                            with email_file.open("w", encoding="utf-8") as f:
                                json.dump({
                                    "content": doc.content,
                                    "metadata": doc.metadata
                                }, f, indent=2, ensure_ascii=False)

                        st.success(f"✅ Extracted and saved {len(raw_docs)} emails to disk!")
                        st.info(f"📁 **Saved to:** `{raw_dir.relative_to(project_path)}`")

                        # Show statistics
                        with st.expander("📊 Email Statistics"):
                            st.write(f"**Total Emails:** {len(raw_docs)}")

                            # Count by sender
                            senders = {}
                            for doc in raw_docs:
                                sender = doc.metadata.get("sender_name", "Unknown")
                                senders[sender] = senders.get(sender, 0) + 1

                            st.write("**Top Senders:**")
                            for sender, count in sorted(senders.items(), key=lambda x: x[1], reverse=True)[:5]:
                                st.write(f"- {sender}: {count} emails")

                        st.info(
                            "📋 **Next Steps:** Go to 'Pipeline Actions' tab and run:\n"
                            "1. **ingest** - Load emails from disk\n"
                            "2. **chunk** - Break emails into chunks\n"
                            "3. **embed** - Create embeddings and index"
                        )
                    else:
                        st.warning("⚠️ No emails were extracted. Check your settings and try again.")

                except Exception as e:
                    st.error(f"❌ Extraction failed: {e}")
                    with st.expander("🔧 Technical Details"):
                        st.code(str(e))
                        import traceback
                        st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Error loading project configuration: {e}")


def load_outlook_config(project_path: Path) -> dict:
    """
    Load Outlook configuration from project config file.

    Args:
        project_path: Path to the project directory

    Returns:
        Dict with Outlook configuration, or empty dict if not found
    """
    config_path = project_path / "config.yml"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("sources", {}).get("outlook", {})
    except Exception:
        return {}


def is_outlook_project(project_path: Path) -> bool:
    """
    Check if a project has Outlook integration enabled.

    Args:
        project_path: Path to the project directory

    Returns:
        True if Outlook is enabled, False otherwise
    """
    outlook_config = load_outlook_config(project_path)
    return outlook_config.get("enabled", False)
