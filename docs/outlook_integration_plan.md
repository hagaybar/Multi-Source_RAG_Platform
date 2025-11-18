# Outlook Local Client Integration Plan

**Project**: Multi-Source RAG Platform
**Feature**: Outlook Local Client as Data Source
**Date**: 2025-01-18
**Status**: Planning Phase

---

## Executive Summary

This document outlines the feasibility analysis and implementation plan for integrating Microsoft Outlook local client as a data source in the Multi-Source RAG Platform. The integration will enable users to connect to their local Outlook installation, select specific folders, and ingest emails from the last X days into the RAG pipeline for semantic search and retrieval.

### Key Findings

✅ **HIGHLY FEASIBLE** - The platform already has substantial infrastructure in place:
- Email file format support (EML, MSG, MBOX)
- Email-specific chunking strategies
- Fully functional Streamlit UI
- Modular ingestion pipeline
- Configuration-driven architecture
- **🎯 EXISTING IMPLEMENTATION**: A working `EmailFetcher` class from a previous project (`docs/reference_docs/email_fetcher.py`)

**MAJOR UPDATE**: After discovering the existing `EmailFetcher` implementation, this integration is **even more feasible** than initially assessed. The core Outlook connector is already built and production-tested. Implementation now focuses on **adaptation and integration** rather than building from scratch.

The main work remaining is:
1. ✅ Adapting `EmailFetcher` to work with the ingestion pipeline (convert to `RawDoc` format)
2. ✅ Creating Streamlit UI components for Outlook projects
3. ✅ Updating configuration schemas
4. ✅ Writing integration tests

---

## 1. Current Infrastructure Assessment

### 1.1 Existing Email Support

The platform already supports three email formats with complete infrastructure:

| Format | Status | Location | Features |
|--------|--------|----------|----------|
| **EML** | ✅ Complete | `scripts/ingestion/email_loader.py:18-41` | Multipart MIME, metadata extraction |
| **MSG** | ✅ Complete | `scripts/ingestion/email_loader.py:44-80` | Outlook format, uses `extract-msg` |
| **MBOX** | ✅ Complete | `scripts/ingestion/email_loader.py:83-128` | Bulk mailbox, multiple messages |
| **PST** | ⚠️ Prepared | `scripts/ingestion/email_loader.py:131-173` | Code ready, dependency disabled |

### 1.2 Email Processing Pipeline

**Email Cleaning** (`scripts/utils/email_utils.py:4-48`):
- Removes quoted replies (lines starting with `>`)
- Removes reply blocks (`On ... wrote:` patterns)
- Removes signatures (after `-- ` delimiter)
- Configurable options

**Email Chunking** (`configs/chunk_rules.yaml:5-31`):
- Strategy: `by_email_block` (spaCy sentence tokenization)
- Min tokens: 20
- Max tokens: 300
- Overlap: 5 tokens

**Implementation** (`scripts/chunking/chunker_v3.py:146-150`):
```python
elif rule.strategy in ("by_email_block", "eml"):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(cleaned_text)
    items = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
```

### 1.3 Streamlit UI Infrastructure

**Location**: `scripts/ui/`

**Current Features**:
- Project creation and management (`ui_project_manager.py`)
- File upload interface with support for: `["pdf", "docx", "pptx", "xlsx", "txt", "eml", "msg", "mbox", "html"]`
- Configuration editor with validation
- Pipeline execution interface
- Raw file viewer

**Key Components**:
1. `ui_v3.py` - Main UI with navigation tabs
2. `ui_project_manager.py` - Project setup and file management
3. `ui_custom_pipeline.py` - Pipeline execution
4. `validation_helpers.py` - Form validation utilities

### 1.4 Project Architecture

**Project Structure**:
```
project_root/
├── config.yml                 # Project configuration
├── input/
│   ├── raw/                  # Source files (auto-discovered by type)
│   │   ├── pdf/
│   │   ├── eml/
│   │   └── msg/
│   └── cache/images/         # Extracted images
├── output/
│   ├── faiss/               # Vector indexes (per doc_type)
│   ├── metadata/            # Chunk metadata (JSONL)
│   └── chunks/              # Processed chunks
└── logs/                    # Project logs
```

**Pipeline Flow**:
```
Outlook Connector → EML Extraction → Ingestion Manager → RawDoc list
    ↓
Email Cleaning → Chunker v3 → Chunk list
    ↓
Unified Embedder → Vector Embeddings
    ↓
FAISS Indexing → Searchable via API
    ↓
Query → Retrieval Manager → LLM Response
```

---

## 2. Outlook Integration: Technical Analysis

### 2.1 Existing EmailFetcher Implementation ⭐

**Location**: `docs/reference_docs/email_fetcher.py`

**Status**: ✅ Complete and production-tested

**Key Features**:
- **Connection Management**: Proper COM initialization with `pythoncom.CoInitializeEx(0)`
- **Folder Navigation**: Supports nested folders with `>` separator (e.g., "Inbox > Work > Projects")
- **Date Filtering**: DASL query for ReceivedTime: `[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'`
- **Email Extraction**: Subject, Sender, ReceivedTime, Body (raw and cleaned)
- **Error Handling**: Comprehensive try-catch blocks with proper cleanup
- **Logging**: Integrated with `LoggerManager`
- **Output Formats**: Returns DataFrame or saves to TSV

**Connection Pattern (from existing code)**:
```python
import win32com.client as win32
import pythoncom

# Initialize COM (required for threading)
pythoncom.CoInitializeEx(0)

try:
    # Connect to Outlook
    outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")

    # Navigate to account
    account_folder = None
    for i in range(outlook.Folders.Count):
        folder = outlook.Folders.Item(i + 1)
        if folder.Name == account_name:
            account_folder = folder
            break

    # Navigate to target folder
    folder = account_folder.Folders["Inbox"]
    for name in folder_path.split(">"):
        folder = folder.Folders[name]

    # Filter by date
    cutoff = datetime.now() - timedelta(days=days)
    filter_str = f"[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'"
    filtered_items = folder.Items.Restrict(filter_str)

    # Extract emails
    for item in filtered_items:
        if hasattr(item, "Class") and item.Class == 43:  # olMailItem
            # Process email...

finally:
    pythoncom.CoUninitialize()
```

### 2.2 Connection Method: pywin32

**Library**: `pywin32` (Windows-only) - **Already used in EmailFetcher**

**Installation**:
```bash
poetry add pywin32
```

**Folder Constants** (for `GetDefaultFolder`):
```python
olFolderInbox = 6
olFolderSentMail = 5
olFolderDeletedItems = 3
olFolderDrafts = 16
olFolderJunk = 23
# ... and more
```

### 2.2 Email Retrieval Strategy

**Date Filtering**:
```python
from datetime import datetime, timedelta

# Calculate date range
days_back = 30  # Configurable
end_date = datetime.now()
start_date = end_date - timedelta(days=days_back)

# Filter emails using DASL query
filter_str = f"[ReceivedTime] >= '{start_date.strftime('%m/%d/%Y %H:%M %p')}'"
filtered_items = folder.Items.Restrict(filter_str)
```

**Message Extraction**:
```python
for item in filtered_items:
    # Extract fields
    subject = item.Subject
    sender = item.SenderEmailAddress
    received_time = item.ReceivedTime
    body = item.Body  # Plain text
    html_body = item.HTMLBody  # HTML version

    # Save as EML or process directly
    # Option 1: Save as EML file
    item.SaveAs(f"path/to/{item.EntryID}.eml", 3)  # 3 = olMSG or olEML

    # Option 2: Process directly (preferred)
    metadata = {
        "subject": subject,
        "sender": sender,
        "date": received_time.strftime("%Y-%m-%d %H:%M:%S"),
        "message_id": item.EntryID,
        "doc_type": "outlook_eml",
        "source_type": "outlook_local",
    }
```

### 2.3 Platform Considerations

**Requirements**:
- ✅ Windows OS (WSL2 compatible with Windows Outlook installed on host)
- ✅ Outlook installed and configured
- ✅ Local execution (not remote)
- ⚠️ Outlook must be running or will be launched by COM

**Limitations**:
- Windows-only (macOS would require different approach using AppleScript)
- Requires Outlook desktop client (not compatible with Outlook Web)
- User must have local Outlook profile configured

**Security**:
- Outlook may prompt user for permission (COM access warning)
- This is a security feature and expected behavior
- Can be suppressed with registry settings (organization policy)

---

## 3. Implementation Plan

### Phase 1: Adapt EmailFetcher to Ingestion Pipeline (Priority: HIGH)

**Status**: ✅ EmailFetcher already exists at `docs/reference_docs/email_fetcher.py`

**Files to Create/Modify**:
1. ✅ **Adapt existing**: `docs/reference_docs/email_fetcher.py` → `scripts/connectors/outlook_connector.py`
2. 🆕 **Create**: `scripts/connectors/__init__.py` - Module initialization
3. 🆕 **Create**: `tests/connectors/test_outlook_connector.py` - Unit tests

**Adaptation Strategy**:

The existing `EmailFetcher` class is nearly perfect but needs minor modifications:

**Current Output**: DataFrame with columns `[Subject, Sender, Received, Raw Body, Cleaned Body]`

**Required Output**: `List[Tuple[str, dict]]` where each tuple is `(body_text, metadata)`

**Modifications Needed**:

```python
# Current method (EmailFetcher):
def fetch_emails_from_folder(self, return_dataframe=False, save=True) -> pd.DataFrame | str:
    # ... returns DataFrame or TSV path

# New method (OutlookConnector):
def extract_emails(self) -> List[Tuple[str, dict]]:
    """
    Extract emails and return in format compatible with ingestion pipeline.
    Returns list of (body_text, metadata) tuples.
    """
    pythoncom.CoInitializeEx(0)
    try:
        outlook = self.connect_to_outlook()
        account_folder = self._get_account_folder(outlook)
        target_folder = self._get_target_folder(account_folder)

        cutoff = datetime.now() - timedelta(days=self.days)
        filter_str = f"[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'"
        filtered_items = target_folder.Items.Restrict(filter_str)

        email_tuples = []
        for item in filtered_items:
            if hasattr(item, "Class") and item.Class == 43:  # olMailItem
                # Extract body (use cleaned version from email_utils)
                raw_body = item.Body if hasattr(item, "Body") else ""

                # Use existing email_utils.clean_email_text()
                from scripts.utils.email_utils import clean_email_text
                cleaned_body = clean_email_text(
                    raw_body,
                    remove_quoted_lines=True,
                    remove_reply_blocks=True,
                    remove_signature=True
                )

                # Build metadata compatible with ingestion pipeline
                metadata = {
                    "source_filepath": f"outlook://{self.account_name}/{self.folder_path}",
                    "content_type": "email",
                    "doc_type": "outlook_eml",
                    "subject": item.Subject if hasattr(item, "Subject") else "",
                    "sender": item.SenderEmailAddress if hasattr(item, "SenderEmailAddress") else "",
                    "sender_name": item.SenderName if hasattr(item, "SenderName") else "",
                    "date": item.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S") if hasattr(item, "ReceivedTime") else "",
                    "message_id": item.EntryID if hasattr(item, "EntryID") else "",
                }

                email_tuples.append((cleaned_body, metadata))

                # Respect max_emails limit
                if self.max_emails and len(email_tuples) >= self.max_emails:
                    break

        self.logger.info(f"Extracted {len(email_tuples)} emails from {self.folder_path}")
        return email_tuples

    finally:
        pythoncom.CoUninitialize()
```

**Key Differences from Original EmailFetcher**:

1. ✅ **Return format**: Changed from DataFrame to `List[Tuple[str, dict]]`
2. ✅ **Email cleaning**: Use existing `scripts/utils/email_utils.clean_email_text()` instead of TSV sanitization
3. ✅ **Metadata format**: Match ingestion pipeline expectations (doc_type, source_filepath, etc.)
4. ✅ **Configuration**: Use dataclass for cleaner config management
5. ✅ **Keep core logic**: All Outlook connection, navigation, filtering logic remains unchanged

**OutlookConnector Class (Adapted from EmailFetcher)**:

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import win32com.client as win32
import pythoncom
from scripts.utils.logger import LoggerManager
from scripts.utils.email_utils import clean_email_text

@dataclass
class OutlookConfig:
    """Configuration for Outlook connection."""
    account_name: str              # Outlook account name
    folder_path: str               # e.g., "Inbox" or "Inbox > Subfolder"
    days_back: int = 30            # Number of days to look back
    max_emails: Optional[int] = None  # Limit number of emails extracted
    include_attachments: bool = False # Future: extract attachments

class OutlookConnector:
    """
    Adapter around EmailFetcher for RAG pipeline integration.
    Based on existing implementation from docs/reference_docs/email_fetcher.py
    """

    def __init__(self, config: OutlookConfig):
        self.config = config
        self.logger = LoggerManager.get_logger("OutlookConnector")
        self.account_name = config.account_name
        self.folder_path = config.folder_path
        self.days = config.days_back
        self.max_emails = config.max_emails

    # Methods from EmailFetcher (keep as-is):
    # - connect_to_outlook()
    # - _get_account_folder()
    # - _get_target_folder()

    # New method for pipeline integration:
    def extract_emails(self) -> List[Tuple[str, dict]]:
        # Implementation shown above
        pass
```

**Key Methods** (from EmailFetcher, keeping unchanged):

1. ✅ **`connect_to_outlook()`**: Establish COM connection (line 70-92)
2. ✅ **`_get_account_folder()`**: Navigate to account (line 208-228)
3. ✅ **`_get_target_folder()`**: Navigate to folder (line 230-251)
4. 🔄 **`extract_emails()`**: NEW method adapted from `fetch_emails_from_folder()` (line 94-172)

**Error Handling**:
- Connection failures (Outlook not installed, not running)
- Folder not found
- Permission denied (COM security)
- Corrupted messages
- Network issues (for Exchange online)

### Phase 2: Integration with Ingestion Pipeline (Priority: HIGH)

**Modification**: `scripts/ingestion/manager.py`

Add support for **dynamic sources** (not just files):

```python
class IngestionManager:
    def __init__(self):
        self.loader_registry = LOADER_REGISTRY
        self.connector_registry = {}  # NEW: For dynamic sources

    def register_connector(self, source_type: str, connector_class):
        """Register a dynamic data source connector."""
        self.connector_registry[source_type] = connector_class

    def ingest_outlook(self, outlook_config: OutlookConfig) -> List[RawDoc]:
        """Ingest emails from Outlook connector."""
        connector = OutlookConnector(outlook_config)
        connector.connect()

        raw_emails = connector.extract_emails()
        connector.close()

        # Convert to RawDoc objects
        raw_docs = []
        for body, metadata in raw_emails:
            raw_docs.append(RawDoc(content=body, metadata=metadata))

        return raw_docs
```

**New Configuration**: `config.yml` schema extension

```yaml
project:
  name: "my_outlook_project"
  description: "Emails from my Outlook inbox"
  language: "en"

sources:
  outlook:
    enabled: true
    folder_path: "Inbox/Work Projects"
    days_back: 60
    max_emails: 1000
    account_name: null  # null = default account
    include_attachments: false
    refresh_interval: "daily"  # or "manual"

embedding:
  model: "text-embedding-3-large"
  # ... rest of config
```

### Phase 3: Streamlit UI for Outlook Integration (Priority: HIGH)

**File**: `scripts/ui/ui_outlook_manager.py` (NEW)

**UI Components**:

#### 3.1 Outlook Project Creation

```python
def render_outlook_project_creation():
    """
    Streamlit form for creating an Outlook-based project.
    """
    st.subheader("📧 Create Outlook Project")

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

        # Test connection button
        col1, col2 = st.columns([3, 1])
        with col1:
            test_connection_info = st.empty()
        with col2:
            if st.form_submit_button("🔌 Test Connection"):
                # Test Outlook connection
                pass

        folder_path = st.text_input(
            "Folder Path",
            value="Inbox",
            help="e.g., 'Inbox', 'Inbox/Subfolder', or 'Sent Items'"
        )

        days_back = st.slider(
            "Days to Look Back",
            min_value=1,
            max_value=365,
            value=30,
            help="Extract emails from the last X days"
        )

        max_emails = st.number_input(
            "Max Emails (Optional)",
            min_value=0,
            value=0,
            help="0 = no limit"
        )

        account_name = st.text_input(
            "Account Name (Optional)",
            value="",
            help="Leave empty to use default Outlook account"
        )

        include_attachments = st.checkbox(
            "Include Attachments",
            value=False,
            help="Extract text from email attachments (PDFs, DOCX, etc.)"
        )

        st.markdown("### Embedding Settings")

        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-large", "text-embedding-ada-002", "bge-large-en-v1.5"]
        )

        submitted = st.form_submit_button("📧 Create Outlook Project")

        if submitted:
            # Validation
            # Create project with Outlook source configuration
            # Show success/error messages
            pass
```

#### 3.2 Outlook Connection Test Widget

```python
def render_outlook_connection_test():
    """
    Test Outlook connection and show available folders.
    """
    st.subheader("🔌 Test Outlook Connection")

    if st.button("Test Connection"):
        with st.spinner("Connecting to Outlook..."):
            try:
                connector = OutlookConnector(OutlookConfig(folder_path="Inbox"))
                connector.connect()

                # Get account info
                accounts = list_outlook_accounts(connector.namespace)
                folders = list_outlook_folders(connector.namespace)

                st.success("✅ Successfully connected to Outlook!")

                st.markdown("### Available Accounts")
                for account in accounts:
                    st.write(f"- {account}")

                st.markdown("### Available Folders (Top Level)")
                for folder in folders:
                    st.write(f"- {folder}")

                connector.close()

            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
                st.info("💡 Ensure Outlook is installed and configured on this machine.")
```

#### 3.3 Outlook Email Preview

```python
def render_outlook_email_preview(project_path: Path):
    """
    Preview emails that would be extracted with current settings.
    """
    st.subheader("📬 Email Preview")

    # Load Outlook config from project
    config = load_outlook_config(project_path)

    if st.button("🔍 Preview Emails"):
        with st.spinner("Fetching emails from Outlook..."):
            try:
                connector = OutlookConnector(config)
                connector.connect()

                # Get first 10 emails (preview only)
                preview_config = OutlookConfig(
                    folder_path=config.folder_path,
                    days_back=config.days_back,
                    max_emails=10
                )
                emails = connector.extract_emails()
                connector.close()

                st.success(f"✅ Found {len(emails)} emails (showing first 10)")

                for i, (body, meta) in enumerate(emails[:10]):
                    with st.expander(f"📧 {meta.get('subject', 'No Subject')}"):
                        st.markdown(f"**From:** {meta.get('sender', 'Unknown')}")
                        st.markdown(f"**Date:** {meta.get('date', 'Unknown')}")
                        st.markdown(f"**Preview:**")
                        st.text(body[:500] + "..." if len(body) > 500 else body)

            except Exception as e:
                st.error(f"❌ Preview failed: {e}")
```

#### 3.4 Outlook Ingestion Control

```python
def render_outlook_ingestion_controls(project_path: Path):
    """
    Controls for manually triggering Outlook email ingestion.
    """
    st.subheader("⚙️ Outlook Ingestion")

    config = load_outlook_config(project_path)

    st.info(f"📂 Folder: `{config.folder_path}`")
    st.info(f"📅 Days Back: {config.days_back}")
    st.info(f"📊 Max Emails: {config.max_emails or 'No limit'}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Emails"):
            with st.spinner("Extracting emails from Outlook..."):
                try:
                    # Run ingestion
                    ingestion_mgr = IngestionManager()
                    raw_docs = ingestion_mgr.ingest_outlook(config)

                    st.success(f"✅ Extracted {len(raw_docs)} emails")
                    st.session_state.outlook_raw_docs = raw_docs

                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    with col2:
        if st.button("📊 Show Statistics"):
            # Show email statistics (count by sender, date distribution, etc.)
            pass

    with col3:
        if st.button("🧹 Clear Cache"):
            # Clear cached emails
            pass
```

**UI Integration**: Update `scripts/ui/ui_v3.py`

Add new tab in navigation:

```python
TAB_OPTIONS = [
    "Projects",
    "Data",
    "Outlook Integration",  # NEW
    "Pipeline Actions",
    "Utilities / Tools"
]

# ...

elif section == "Outlook Integration":
    st.header("📧 Outlook Integration")

    # Check if current project is an Outlook project
    if st.session_state.selected_project:
        project_path = Path("data/projects") / st.session_state.selected_project
        config = load_project_config(project_path)

        if config.get("sources", {}).get("outlook", {}).get("enabled"):
            # Show Outlook-specific UI
            render_outlook_connection_test()
            st.markdown("---")
            render_outlook_email_preview(project_path)
            st.markdown("---")
            render_outlook_ingestion_controls(project_path)
        else:
            st.info("This project does not have Outlook integration enabled.")
            if st.button("Enable Outlook for this project"):
                # Add Outlook config to existing project
                pass
    else:
        st.warning("Please select a project first.")
        render_outlook_project_creation()
```

### Phase 4: Configuration & Scheduling (Priority: MEDIUM)

**Automatic Email Refresh**:

Option 1: **Manual Refresh** (Phase 3)
- User clicks "Refresh Emails" button in UI
- Simple, no background processes

Option 2: **Scheduled Refresh** (Phase 4)
- Use `schedule` library or cron jobs
- Configurable interval (daily, hourly, etc.)

**Implementation** (`scripts/scheduling/outlook_scheduler.py`):

```python
import schedule
import time
from pathlib import Path

class OutlookScheduler:
    """
    Automatically refresh Outlook emails on a schedule.
    """

    def __init__(self, project_path: Path, interval: str = "daily"):
        self.project_path = project_path
        self.interval = interval

    def run_ingestion(self):
        """Execute Outlook ingestion."""
        config = load_outlook_config(self.project_path)
        ingestion_mgr = IngestionManager()
        raw_docs = ingestion_mgr.ingest_outlook(config)

        # Optionally trigger full pipeline
        # pipeline_runner = PipelineRunner(self.project_path)
        # pipeline_runner.run_all()

        print(f"✅ Ingested {len(raw_docs)} emails at {datetime.now()}")

    def start(self):
        """Start the scheduler."""
        if self.interval == "daily":
            schedule.every().day.at("08:00").do(self.run_ingestion)
        elif self.interval == "hourly":
            schedule.every().hour.do(self.run_ingestion)

        while True:
            schedule.run_pending()
            time.sleep(60)
```

**CLI Command** (`scripts/cli.py`):

```bash
# Manual refresh
python -m scripts.cli outlook-refresh /path/to/project

# Start scheduler (runs in background)
python -m scripts.cli outlook-schedule /path/to/project --interval daily
```

### Phase 5: Advanced Features (Priority: LOW)

**5.1 Attachment Extraction**:
- Extract text from PDF, DOCX, XLSX attachments
- Save attachments to `input/cache/attachments/`
- Index attachment content separately

**5.2 Email Thread Reconstruction**:
- Group emails by conversation ID
- Maintain thread context in chunking

**5.3 Multi-Account Support**:
- Connect to multiple Outlook accounts
- Aggregate emails from different sources

**5.4 Smart Filtering**:
- Filter by sender
- Filter by subject keywords
- Exclude spam/junk folders
- Include/exclude specific labels

---

## 4. Testing Strategy

### 4.1 Unit Tests

**File**: `tests/connectors/test_outlook_connector.py`

```python
import pytest
from scripts.connectors.outlook_connector import OutlookConnector, OutlookConfig

@pytest.mark.skipif(not is_windows(), reason="Outlook only available on Windows")
def test_outlook_connection():
    """Test basic Outlook connection."""
    config = OutlookConfig(folder_path="Inbox", days_back=7)
    connector = OutlookConnector(config)

    assert connector.connect() is True
    assert connector.namespace is not None

    connector.close()

def test_outlook_folder_navigation():
    """Test folder navigation."""
    # Implementation...

def test_email_extraction():
    """Test email extraction with date filtering."""
    # Implementation...

def test_email_metadata_parsing():
    """Test metadata extraction from emails."""
    # Implementation...
```

### 4.2 Integration Tests

**File**: `tests/integration/test_outlook_pipeline.py`

```python
def test_outlook_to_faiss_pipeline():
    """
    End-to-end test: Outlook → Ingestion → Chunking → Embedding → FAISS
    """
    # Setup test project
    project_path = create_test_project_outlook()

    # Extract emails
    config = OutlookConfig(folder_path="Inbox", days_back=7, max_emails=10)
    ingestion_mgr = IngestionManager()
    raw_docs = ingestion_mgr.ingest_outlook(config)

    assert len(raw_docs) > 0

    # Run full pipeline
    runner = PipelineRunner(project_path)
    runner.run_all()

    # Verify FAISS index created
    faiss_path = project_path / "output" / "faiss" / "outlook_eml.faiss"
    assert faiss_path.exists()

    # Test retrieval
    retrieval_mgr = RetrievalManager(project_path)
    results = retrieval_mgr.search("test query", top_k=5)

    assert len(results) > 0
```

### 4.3 Manual Testing Checklist

- [ ] Outlook connection successful
- [ ] Folder navigation works for nested folders
- [ ] Date filtering returns correct email range
- [ ] Email metadata extracted correctly
- [ ] Email body text clean (no HTML artifacts)
- [ ] Email cleaning removes signatures and quotes
- [ ] Chunking produces reasonable chunks
- [ ] Embeddings generated successfully
- [ ] FAISS index searchable
- [ ] UI displays connection status correctly
- [ ] UI preview shows accurate email count
- [ ] Error messages are user-friendly
- [ ] Performance acceptable for 1000+ emails

---

## 5. Dependencies

### 5.1 New Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies
    "pywin32 (>=308,<309)",  # Outlook COM interface
    "schedule (>=1.2.0,<2.0.0)",  # Optional: for scheduled refresh
]
```

### 5.2 Platform-Specific Installation

**Windows**:
```bash
poetry add pywin32
poetry install
```

**WSL2** (if accessing Windows Outlook):
- Requires Windows Outlook installed on host
- May need special COM configuration

**macOS / Linux**:
- Not supported (alternative: IMAP/Exchange protocols)
- Could implement IMAP fallback for cross-platform support

---

## 6. File Structure

### New Files to Create

```
scripts/
├── connectors/                          # NEW
│   ├── __init__.py
│   ├── outlook_connector.py             # Main Outlook connector
│   └── base_connector.py                # Abstract base class for connectors
│
├── scheduling/                          # NEW (Phase 4)
│   ├── __init__.py
│   └── outlook_scheduler.py
│
├── ui/
│   └── ui_outlook_manager.py            # NEW - Outlook UI components
│
└── ingestion/
    └── manager.py                       # MODIFY - Add connector support

tests/
├── connectors/                          # NEW
│   ├── __init__.py
│   └── test_outlook_connector.py
│
└── integration/
    └── test_outlook_pipeline.py         # NEW

docs/
└── outlook_integration_plan.md          # THIS FILE

configs/
└── chunk_rules.yaml                     # MODIFY - Add outlook_eml rules (if different)
```

### Files to Modify

1. **`scripts/ui/ui_v3.py`**: Add "Outlook Integration" tab
2. **`scripts/ingestion/manager.py`**: Add `ingest_outlook()` method
3. **`scripts/core/project_manager.py`**: Support Outlook project type
4. **`pyproject.toml`**: Add `pywin32` dependency
5. **`configs/chunk_rules.yaml`**: Add `outlook_eml` chunking rules (if needed)

---

## 7. Configuration Schema

### Project Config Example: `data/projects/my_outlook_project/config.yml`

```yaml
project:
  name: "my_outlook_project"
  description: "Work emails from Outlook Inbox"
  language: "en"

sources:
  outlook:
    enabled: true
    folder_path: "Inbox/Work Projects"
    days_back: 60
    max_emails: 1000
    account_name: null  # null = default account
    include_attachments: false
    refresh_interval: "daily"  # "manual", "hourly", "daily", "weekly"

    # Advanced filters (Phase 5)
    filters:
      senders:
        include: ["important@company.com"]
        exclude: ["spam@example.com"]
      subject_keywords:
        include: ["[PROJECT]", "urgent"]
        exclude: ["newsletter", "unsubscribe"]
      exclude_folders: ["Junk Email", "Deleted Items"]

embedding:
  model: "text-embedding-3-large"
  embed_batch_size: 64
  image_enrichment: false
  skip_duplicates: true

llm:
  model: "gpt-4"
  temperature: 0.4
  max_tokens: 400

agents:
  enable_image_insight: false

chunking:
  doc_type: "outlook_eml"  # Use email chunking rules
```

---

## 8. User Workflow

### 8.1 First-Time Setup

1. **Create Outlook Project** (UI):
   - Navigate to "Projects" tab
   - Click "Create New Project" → "Outlook Project"
   - Fill in project details:
     - Project name: "My Work Emails"
     - Folder path: "Inbox"
     - Days back: 30
     - Embedding model: text-embedding-3-large
   - Click "Create Outlook Project"

2. **Test Connection** (UI):
   - Navigate to "Outlook Integration" tab
   - Click "🔌 Test Connection"
   - Verify Outlook accounts and folders are detected

3. **Preview Emails** (UI):
   - Click "🔍 Preview Emails"
   - Review the first 10 emails that would be extracted
   - Verify date range and folder are correct

4. **Extract Emails** (UI):
   - Click "🔄 Refresh Emails"
   - Wait for extraction to complete
   - View statistics (emails extracted, date range, etc.)

5. **Run Pipeline** (UI):
   - Navigate to "Pipeline Actions" tab
   - Click "▶️ Run Full Pipeline"
   - Monitor progress (ingestion → chunking → embedding → indexing)

6. **Test Search** (UI):
   - Navigate to "Utilities / Tools" tab
   - Enter a search query
   - Verify relevant emails are retrieved

### 8.2 Daily Usage

**Option A: Manual Refresh**
1. Open Streamlit UI
2. Navigate to "Outlook Integration"
3. Click "🔄 Refresh Emails"
4. Run pipeline to process new emails

**Option B: Scheduled Refresh** (Phase 4)
1. Run scheduler in background:
   ```bash
   python -m scripts.cli outlook-schedule /path/to/project --interval daily
   ```
2. Emails automatically refresh at scheduled time
3. Check logs for status

**Option C: API Integration**
```bash
# Trigger refresh via CLI
python -m scripts.cli outlook-refresh /path/to/project

# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"project": "my_outlook_project", "query": "meeting notes from last week"}'
```

---

## 9. Error Handling & Edge Cases

### 9.1 Connection Errors

| Error | Cause | Solution |
|-------|-------|----------|
| **COM Connection Failed** | Outlook not installed | Show error message with installation link |
| **Access Denied** | COM security settings | Guide user to enable programmatic access |
| **Outlook Not Running** | Outlook needs to be launched | Auto-launch or prompt user |
| **Multiple Outlook Profiles** | User has multiple profiles | Prompt user to select profile |

### 9.2 Data Errors

| Error | Cause | Solution |
|-------|-------|----------|
| **Folder Not Found** | Invalid folder path | Show available folders in error message |
| **No Emails in Date Range** | Date range too narrow | Suggest expanding date range |
| **Corrupted Email** | Email cannot be parsed | Skip email, log warning, continue |
| **Oversized Email** | Email > 10MB | Truncate or skip, log warning |
| **Attachment Extraction Failed** | Unsupported attachment type | Skip attachment, process email body |

### 9.3 Performance Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Slow Extraction** | Too many emails | Add pagination, progress bar |
| **High Memory Usage** | Loading all emails at once | Process in batches of 100 |
| **Outlook Freezing** | Too many concurrent requests | Add throttling (delay between requests) |

---

## 10. Performance Considerations

### 10.1 Email Extraction

**Estimated Performance** (based on typical Outlook performance):
- **Connection time**: 1-3 seconds
- **Email retrieval**: ~10-50 emails/second (depends on email size)
- **Date filtering**: Fast (uses indexed ReceivedTime field)
- **Folder navigation**: 0.1-0.5 seconds per level

**Optimization Strategies**:
1. **Batch Processing**: Process emails in chunks of 100
2. **Progress Tracking**: Show progress bar in UI
3. **Caching**: Cache folder structure to avoid repeated lookups
4. **Incremental Updates**: Only extract emails newer than last sync
5. **Parallel Processing**: Use threading for attachment extraction

### 10.2 Pipeline Performance

**Typical Timeline** (1000 emails):
1. Outlook extraction: 20-60 seconds
2. Email cleaning: 10-30 seconds
3. Chunking: 30-60 seconds
4. Embedding (OpenAI): 5-10 minutes
5. FAISS indexing: 5-15 seconds
**Total**: ~6-12 minutes

**Optimization**:
- Use local embeddings (sentence-transformers) for faster processing: ~2-3 minutes total
- Enable `skip_duplicates` to avoid re-embedding unchanged emails

---

## 11. Security & Privacy

### 11.1 Data Security

**Local Storage**:
- All emails stored locally in project directory
- No data sent to cloud (except embeddings if using OpenAI)
- User has full control over data retention

**Access Control**:
- Outlook COM requires user to be logged into Windows
- No credential storage needed (uses Windows authentication)
- Optional: Add project-level password protection

### 11.2 Privacy Considerations

**Sensitive Data**:
- Emails may contain PII, confidential information
- Recommend users to:
  - Use specific folders (not entire mailbox)
  - Filter out sensitive senders/subjects
  - Review emails before indexing

**Compliance**:
- GDPR: User controls data, can delete project anytime
- Data residency: All data stays on user's machine
- Audit trail: Log all email extractions with timestamps

---

## 12. Deployment & Rollout

### 12.1 Development Environment

**Phase 1-2 Development**:
```bash
# Install dependencies
poetry add pywin32

# Run tests
pytest tests/connectors/test_outlook_connector.py -v

# Test manually
python scripts/connectors/outlook_connector.py
```

### 12.2 Testing Environment

**Integration Testing**:
1. Create test Outlook profile with sample emails
2. Run full pipeline test
3. Verify FAISS index and search results

### 12.3 Production Deployment

**User Installation**:
1. Update dependencies: `poetry install`
2. Configure project via Streamlit UI
3. Test connection before first use

**Documentation to Provide**:
- User guide with screenshots
- Troubleshooting guide for common errors
- Video tutorial (optional)

---

## 13. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Windows-only limitation** | Medium | High | Document clearly, consider IMAP fallback |
| **Outlook COM instability** | High | Low | Robust error handling, retry logic |
| **User permission issues** | Medium | Medium | Clear error messages, setup guide |
| **Performance with large mailboxes** | Medium | Medium | Pagination, progress bars, limits |
| **Email format inconsistencies** | Low | Medium | Extensive testing, fallback parsing |
| **Outlook version compatibility** | Medium | Low | Test with Outlook 2016, 2019, 365 |

---

## 14. Success Metrics

### 14.1 Technical Metrics

- ✅ Connection success rate > 95%
- ✅ Email extraction speed > 20 emails/second
- ✅ Pipeline completion time < 15 minutes for 1000 emails
- ✅ Search relevance (user feedback) > 80% positive
- ✅ Error rate < 5%

### 14.2 User Experience Metrics

- ✅ Setup time < 5 minutes (first-time users)
- ✅ Daily usage time < 2 minutes (refresh + query)
- ✅ User satisfaction score > 4/5

---

## 15. Timeline Estimate

**⚡ SIGNIFICANTLY REDUCED** due to existing EmailFetcher implementation

### Phase 1: Adapt EmailFetcher (2-3 days) ⬇️ Reduced from 5-7 days
- Day 1: Adapt `EmailFetcher` to `OutlookConnector` (change return format to tuples)
- Day 2: Integrate with `email_utils.clean_email_text()` for email cleaning
- Day 3: Unit tests and validation

### Phase 2: Pipeline Integration (2-3 days) ✅ Same
- Day 1: Modify `IngestionManager` to support `ingest_outlook()`
- Day 2: Configuration schema updates
- Day 3: Integration tests

### Phase 3: Streamlit UI (5-7 days) ✅ Same
- Day 1-2: Project creation UI with Outlook settings
- Day 3: Connection test widget (reuse EmailFetcher connection logic)
- Day 4: Email preview UI
- Day 5: Ingestion controls
- Day 6-7: Testing and refinement

### Phase 4: Scheduling (2-3 days) - OPTIONAL
- Day 1-2: Scheduler implementation
- Day 3: CLI commands and testing

### Phase 5: Advanced Features (3-5 days) - OPTIONAL
- Day 1-2: Attachment extraction
- Day 3: Thread reconstruction
- Day 4-5: Smart filtering

**Total Estimate** (with existing EmailFetcher):
- **Minimum Viable Product (Phase 1-3)**: **9-13 days** ⬇️ (was 12-17 days)
- **Full Feature Set (Phase 1-5)**: **14-21 days** ⬇️ (was 17-25 days)

**Time Saved**: ~3-4 days by leveraging existing EmailFetcher implementation

---

## 16. Next Steps

### Immediate Actions

1. **Approval Decision**:
   - Review this plan with stakeholders
   - Confirm scope (which phases to implement)
   - Allocate development time

2. **Environment Setup**:
   - Install `pywin32` on development machine
   - Create test Outlook profile with sample emails
   - Set up test project structure

3. **Proof of Concept**:
   - Create minimal `outlook_connector.py` (50 lines)
   - Test basic connection and email extraction
   - Validate approach before full implementation

4. **Development Start**:
   - Create feature branch: `feature/outlook-integration`
   - Begin Phase 1 implementation
   - Set up CI/CD for Windows testing (if applicable)

### Questions to Resolve

1. **Platform Priority**:
   - Focus on Windows only, or plan for cross-platform (IMAP)?

2. **UI vs CLI**:
   - Prioritize Streamlit UI or CLI commands first?

3. **Scheduling**:
   - Manual refresh only, or automated scheduling needed?

4. **Attachment Handling**:
   - Essential feature or Phase 5?

5. **Multi-Account Support**:
   - Single account sufficient, or multi-account needed?

---

## 17. Conclusion

### Feasibility Assessment: ✅ HIGHLY FEASIBLE

**Strengths**:
- Existing email infrastructure (EML, MSG, MBOX support)
- Mature Streamlit UI framework
- Modular architecture with clear extension points
- Well-established pywin32 library for Outlook integration

**Challenges**:
- Windows-only limitation (acceptable for enterprise users)
- COM connection stability (mitigated with error handling)
- Performance with large mailboxes (mitigated with pagination)

**Recommendation**:
**PROCEED** with implementation. Start with Phase 1-3 (MVP) to deliver core functionality quickly, then evaluate need for Phase 4-5 based on user feedback.

### Value Proposition

This feature enables users to:
1. **Centralize Knowledge**: Make email content searchable alongside documents
2. **Improve Productivity**: Find information faster across all sources
3. **Automate Workflows**: Automatically index new emails daily
4. **Enhance Context**: RAG can reference email conversations in responses

### Final Notes

This plan is comprehensive and actionable. The platform's existing architecture makes this integration straightforward. The biggest unknown is Outlook COM stability in production, but this can be mitigated with robust error handling and user guidance.

**Ready to begin implementation when approved.**

---

**Document Version**: 1.0
**Last Updated**: 2025-01-18
**Author**: RAG Platform Development Team
**Status**: Awaiting Approval
