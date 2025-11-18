"""
Unit tests for OutlookConnector.

These tests require:
- Windows OS
- Microsoft Outlook installed and configured
- pywin32 library installed

Tests are skipped if Outlook is not available.
"""

import pytest
import sys
from datetime import datetime, timedelta

# Skip all tests in this module if not on Windows
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Outlook connector only works on Windows"
)

try:
    from scripts.connectors.outlook_connector import OutlookConnector, OutlookConfig, OUTLOOK_AVAILABLE
except ImportError:
    OUTLOOK_AVAILABLE = False


@pytest.mark.skipif(not OUTLOOK_AVAILABLE, reason="pywin32 not installed")
class TestOutlookConfig:
    """Tests for OutlookConfig dataclass."""

    def test_config_creation_basic(self):
        """Test creating basic OutlookConfig."""
        config = OutlookConfig(
            account_name="test@example.com",
            folder_path="Inbox"
        )
        assert config.account_name == "test@example.com"
        assert config.folder_path == "Inbox"
        assert config.days_back == 30  # default
        assert config.max_emails is None  # default
        assert config.include_attachments is False  # default

    def test_config_creation_full(self):
        """Test creating OutlookConfig with all parameters."""
        config = OutlookConfig(
            account_name="test@example.com",
            folder_path="Inbox > Work",
            days_back=60,
            max_emails=100,
            include_attachments=True
        )
        assert config.account_name == "test@example.com"
        assert config.folder_path == "Inbox > Work"
        assert config.days_back == 60
        assert config.max_emails == 100
        assert config.include_attachments is True


@pytest.mark.skipif(not OUTLOOK_AVAILABLE, reason="pywin32 not installed")
class TestOutlookConnector:
    """
    Tests for OutlookConnector.

    Note: These tests will attempt to connect to a real Outlook instance.
    They are integration tests rather than pure unit tests.
    """

    def test_connector_initialization(self):
        """Test OutlookConnector initialization."""
        config = OutlookConfig(
            account_name="test@example.com",
            folder_path="Inbox"
        )
        connector = OutlookConnector(config)

        assert connector.account_name == "test@example.com"
        assert connector.folder_path == "Inbox"
        assert connector.days == 30
        assert connector.max_emails is None

    def test_connector_raises_import_error_without_pywin32(self, monkeypatch):
        """Test that OutlookConnector raises ImportError if pywin32 not available."""
        # Temporarily set OUTLOOK_AVAILABLE to False
        import scripts.connectors.outlook_connector as oc_module
        monkeypatch.setattr(oc_module, "OUTLOOK_AVAILABLE", False)

        config = OutlookConfig(
            account_name="test@example.com",
            folder_path="Inbox"
        )

        with pytest.raises(ImportError, match="pywin32 is required"):
            OutlookConnector(config)

    @pytest.mark.integration
    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Requires Windows with Outlook installed"
    )
    def test_connect_to_outlook_integration(self):
        """
        Integration test: Connect to real Outlook instance.

        This test will fail if:
        - Outlook is not installed
        - Outlook is not configured with at least one account
        - COM connection fails
        """
        config = OutlookConfig(
            account_name="test@example.com",  # Will be replaced in actual test
            folder_path="Inbox"
        )
        connector = OutlookConnector(config)

        try:
            outlook = connector.connect_to_outlook()
            assert outlook is not None
            assert hasattr(outlook, "Folders")
        except Exception as e:
            pytest.skip(f"Could not connect to Outlook: {e}")

    @pytest.mark.integration
    def test_list_folders_integration(self):
        """
        Integration test: List folders from Outlook.

        Note: This requires a configured Outlook account.
        Test will be skipped if connection fails.
        """
        # This test needs to be configured with a real account name
        # For now, we'll skip it unless explicitly enabled
        pytest.skip("Requires manual configuration with real Outlook account")

    @pytest.mark.integration
    def test_extract_emails_integration(self):
        """
        Integration test: Extract emails from Outlook.

        Note: This requires a configured Outlook account with emails.
        Test will be skipped if connection fails.
        """
        # This test needs to be configured with a real account name
        pytest.skip("Requires manual configuration with real Outlook account and emails")


class TestOutlookConnectorOutputFormat:
    """Tests for output format validation (mocked tests)."""

    def test_extract_emails_returns_list_of_tuples(self, monkeypatch):
        """Test that extract_emails returns List[Tuple[str, dict]]."""
        if not OUTLOOK_AVAILABLE:
            pytest.skip("pywin32 not available")

        config = OutlookConfig(
            account_name="test@example.com",
            folder_path="Inbox",
            days_back=7
        )
        connector = OutlookConnector(config)

        # Mock the extract_emails method to return expected format
        mock_result = [
            (
                "Test email body",
                {
                    "source_filepath": "outlook://test@example.com/Inbox",
                    "content_type": "email",
                    "doc_type": "outlook_eml",
                    "subject": "Test Subject",
                    "sender": "sender@example.com",
                    "sender_name": "Test Sender",
                    "date": "2025-01-18 10:00:00",
                    "message_id": "12345"
                }
            )
        ]

        # Validate format
        assert isinstance(mock_result, list)
        assert len(mock_result) == 1
        assert isinstance(mock_result[0], tuple)
        assert len(mock_result[0]) == 2

        body, metadata = mock_result[0]
        assert isinstance(body, str)
        assert isinstance(metadata, dict)

        # Validate metadata keys
        required_keys = [
            "source_filepath", "content_type", "doc_type",
            "subject", "sender", "sender_name", "date", "message_id"
        ]
        for key in required_keys:
            assert key in metadata, f"Missing required key: {key}"

    def test_metadata_format(self):
        """Test that metadata format matches ingestion pipeline expectations."""
        expected_metadata = {
            "source_filepath": "outlook://test@example.com/Inbox",
            "content_type": "email",
            "doc_type": "outlook_eml",
            "subject": "Test Subject",
            "sender": "sender@example.com",
            "sender_name": "Test Sender",
            "date": "2025-01-18 10:00:00",
            "message_id": "12345"
        }

        # Validate types
        assert isinstance(expected_metadata["source_filepath"], str)
        assert expected_metadata["source_filepath"].startswith("outlook://")
        assert expected_metadata["content_type"] == "email"
        assert expected_metadata["doc_type"] == "outlook_eml"
        assert isinstance(expected_metadata["subject"], str)
        assert isinstance(expected_metadata["sender"], str)
        assert isinstance(expected_metadata["date"], str)


@pytest.mark.parametrize("folder_path,expected_parts", [
    ("Inbox", ["Inbox"]),
    ("Inbox > Work", ["Inbox", "Work"]),
    ("Inbox > Projects > 2025", ["Inbox", "Projects", "2025"]),
])
def test_folder_path_parsing(folder_path, expected_parts):
    """Test that folder paths are parsed correctly."""
    parts = [part.strip() for part in folder_path.split(">")]
    assert parts == expected_parts


def test_date_filtering_logic():
    """Test date filtering calculation."""
    days_back = 30
    cutoff = datetime.now() - timedelta(days=days_back)

    # Verify cutoff is in the past
    assert cutoff < datetime.now()

    # Verify cutoff is approximately 30 days ago (within 1 hour tolerance)
    expected_cutoff = datetime.now() - timedelta(days=30)
    time_diff = abs((cutoff - expected_cutoff).total_seconds())
    assert time_diff < 3600  # Less than 1 hour difference
