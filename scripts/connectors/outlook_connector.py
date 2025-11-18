"""
Outlook Connector for Multi-Source RAG Platform

This module provides the OutlookConnector class for connecting to a local Microsoft
Outlook client and extracting emails for ingestion into the RAG pipeline.

Based on the EmailFetcher implementation from docs/reference_docs/email_fetcher.py,
adapted to work with the ingestion pipeline's expected output format.

Key adaptations:
- Changed return format from DataFrame to List[Tuple[str, dict]]
- Integrated with email_utils.clean_email_text for email cleaning
- Updated metadata format to match ingestion pipeline expectations
- Added OutlookConfig dataclass for configuration management
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

try:
    import win32com.client as win32  # type: ignore
    import pythoncom  # type: ignore
    OUTLOOK_AVAILABLE = True
except ImportError:
    OUTLOOK_AVAILABLE = False
    win32 = None  # type: ignore
    pythoncom = None  # type: ignore

from scripts.utils.logger import LoggerManager
from scripts.utils.email_utils import clean_email_text


@dataclass
class OutlookConfig:
    """
    Configuration for Outlook connection.

    Attributes:
        account_name: Name of the Outlook account (e.g., "user@company.com")
        folder_path: Path to folder using '>' separator (e.g., "Inbox" or "Inbox > Work")
        days_back: Number of days to look back for emails (default: 30)
        max_emails: Maximum number of emails to extract (None = no limit)
        include_attachments: Whether to extract attachments (not yet implemented)
    """
    account_name: str
    folder_path: str
    days_back: int = 30
    max_emails: Optional[int] = None
    include_attachments: bool = False


class OutlookConnector:
    """
    Connects to local Outlook client and extracts emails for RAG pipeline ingestion.

    This class is adapted from the EmailFetcher implementation to work with the
    ingestion pipeline. It connects to Microsoft Outlook via COM, navigates to
    specified folders, and extracts emails within a date range.

    Example:
        >>> config = OutlookConfig(
        ...     account_name="user@company.com",
        ...     folder_path="Inbox > Work Projects",
        ...     days_back=30
        ... )
        >>> connector = OutlookConnector(config)
        >>> emails = connector.extract_emails()
        >>> # Returns: [(body_text, metadata), ...]

    Requirements:
        - Windows OS
        - Microsoft Outlook installed and configured
        - pywin32 library installed
    """

    def __init__(self, config: OutlookConfig):
        """
        Initialize the OutlookConnector.

        Args:
            config: OutlookConfig dataclass with connection parameters

        Raises:
            ImportError: If pywin32 is not installed
        """
        if not OUTLOOK_AVAILABLE:
            raise ImportError(
                "pywin32 is required for Outlook integration. "
                "Install it with: pip install pywin32"
            )

        self.config = config
        self.logger = LoggerManager.get_logger("OutlookConnector")
        self.account_name = config.account_name
        self.folder_path = config.folder_path
        self.days = config.days_back
        self.max_emails = config.max_emails

    def connect_to_outlook(self):
        """
        Establish connection to Microsoft Outlook application.

        Uses win32com.client to dispatch an Outlook Application COM object
        and get the MAPI namespace.

        Returns:
            win32com.client.Dispatch: Outlook MAPI namespace object

        Raises:
            Exception: If connection to Outlook fails
        """
        try:
            outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
            self.logger.info("Successfully connected to Outlook", extra={
                "action": "outlook_connect",
                "account_name": self.account_name
            })
            return outlook
        except Exception as e:
            self.logger.error(f"Outlook connection failed: {e}", extra={
                "action": "outlook_connect_failed",
                "account_name": self.account_name,
                "error": str(e)
            })
            raise

    def _get_account_folder(self, outlook):
        """
        Retrieve the specified Outlook account folder object.

        Iterates through top-level folders in the Outlook namespace to find
        the one matching self.account_name.

        Args:
            outlook: Outlook MAPI namespace object

        Returns:
            win32com.client.Dispatch: Account folder object or None

        Raises:
            ValueError: If account not found
        """
        try:
            for i in range(outlook.Folders.Count):
                folder = outlook.Folders.Item(i + 1)
                if folder.Name == self.account_name:
                    self.logger.debug(f"Found account folder: {self.account_name}", extra={
                        "action": "account_folder_found",
                        "account_name": self.account_name
                    })
                    return folder

            raise ValueError(f"Account '{self.account_name}' not found")
        except Exception as e:
            self.logger.error(f"Failed to get account folder: {e}", extra={
                "action": "get_account_folder_failed",
                "account_name": self.account_name,
                "error": str(e)
            })
            raise

    def _get_target_folder(self, account_folder):
        """
        Navigate to target email folder within the account.

        The folder_path is expected to be a string like "Inbox" or
        "Inbox > Subfolder > AnotherSubfolder" using '>' as separator.

        Args:
            account_folder: Outlook account folder object

        Returns:
            win32com.client.Dispatch: Target email folder object

        Raises:
            Exception: If any part of folder_path is not found
        """
        try:
            # Start at Inbox
            folder = account_folder.Folders["Inbox"]

            # Navigate through folder path
            folder_parts = [part.strip() for part in self.folder_path.split(">")]

            # If folder_path is just "Inbox", we're already there
            if len(folder_parts) == 1 and folder_parts[0].lower() == "inbox":
                self.logger.debug(f"Using Inbox folder", extra={
                    "action": "target_folder_found",
                    "folder_path": self.folder_path
                })
                return folder

            # Navigate to subfolders (skip first if it's "Inbox")
            start_index = 1 if folder_parts[0].lower() == "inbox" else 0
            for name in folder_parts[start_index:]:
                folder = folder.Folders[name]

            self.logger.debug(f"Found target folder: {self.folder_path}", extra={
                "action": "target_folder_found",
                "folder_path": self.folder_path
            })
            return folder
        except Exception as e:
            self.logger.error(f"Failed to navigate to folder '{self.folder_path}': {e}", extra={
                "action": "get_target_folder_failed",
                "folder_path": self.folder_path,
                "error": str(e)
            })
            raise

    def extract_emails(self) -> List[Tuple[str, dict]]:
        """
        Extract emails from Outlook and return in ingestion pipeline format.

        This is the main method adapted from EmailFetcher.fetch_emails_from_folder().
        Key differences:
        - Returns List[Tuple[str, dict]] instead of DataFrame
        - Uses clean_email_text() for email cleaning
        - Metadata format matches ingestion pipeline expectations

        Returns:
            List of tuples (body_text, metadata) where:
            - body_text: Cleaned email body (str)
            - metadata: Dict with keys: source_filepath, content_type, doc_type,
                       subject, sender, sender_name, date, message_id

        Raises:
            Exception: If connection or extraction fails
        """
        pythoncom.CoInitializeEx(0)  # Initialize COM for this thread

        try:
            # Connect to Outlook
            outlook = self.connect_to_outlook()
            if outlook is None:
                self.logger.error("Failed to connect to Outlook, cannot extract emails")
                return []

            # Navigate to account folder
            account_folder = self._get_account_folder(outlook)
            if account_folder is None:
                self.logger.error(f"Account folder for '{self.account_name}' not found")
                return []

            # Navigate to target folder
            target_folder = self._get_target_folder(account_folder)
            if target_folder is None:
                self.logger.error(f"Target folder '{self.folder_path}' not found")
                return []

            # Calculate date cutoff
            cutoff = datetime.now() - timedelta(days=self.days)

            # Check if folder has items
            if target_folder.Items is None:
                self.logger.warning(f"Folder '{target_folder.Name}' has no items collection")
                return []

            # Filter emails by date using DASL query
            try:
                filter_str = f"[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'"
                filtered_items = target_folder.Items.Restrict(filter_str)

                if filtered_items is None:
                    self.logger.info(f"No items found after filtering by date: {filter_str}")
                    filtered_items = []
            except Exception as e:
                self.logger.error(f"Error restricting items in folder '{target_folder.Name}': {e}")
                filtered_items = []

            # Extract emails
            email_tuples = []
            for item in filtered_items:
                # Only process mail items (Class 43 = olMailItem)
                if hasattr(item, "Class") and item.Class == 43:
                    try:
                        # Extract raw body
                        raw_body = item.Body if hasattr(item, "Body") else ""

                        # Clean email using email_utils
                        cleaned_body = clean_email_text(
                            raw_body,
                            remove_quoted_lines=True,
                            remove_reply_blocks=True,
                            remove_signature=True,
                            signature_delimiter="-- "
                        )

                        # Build metadata compatible with ingestion pipeline
                        metadata = {
                            "source_filepath": f"outlook://{self.account_name}/{self.folder_path}",
                            "content_type": "email",
                            "doc_type": "outlook_eml",
                            "subject": item.Subject if hasattr(item, "Subject") else "",
                            "sender": item.SenderEmailAddress if hasattr(item, "SenderEmailAddress") else "",
                            "sender_name": item.SenderName if hasattr(item, "SenderName") else "",
                            "date": item.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S") if hasattr(item, "ReceivedTime") and item.ReceivedTime else "",
                            "message_id": item.EntryID if hasattr(item, "EntryID") else "",
                        }

                        email_tuples.append((cleaned_body, metadata))

                        # Respect max_emails limit
                        if self.max_emails and len(email_tuples) >= self.max_emails:
                            self.logger.info(f"Reached max_emails limit of {self.max_emails}")
                            break

                    except Exception as e:
                        self.logger.warning(f"Failed to process email: {e}", extra={
                            "action": "email_processing_error",
                            "error": str(e)
                        })
                        continue

            self.logger.info(f"Extracted {len(email_tuples)} emails from {self.folder_path}", extra={
                "action": "emails_extracted",
                "count": len(email_tuples),
                "folder_path": self.folder_path,
                "days_back": self.days
            })

            return email_tuples

        except Exception as e:
            self.logger.error(f"Error in extract_emails: {e}", extra={
                "action": "extract_emails_failed",
                "error": str(e)
            })
            return []

        finally:
            pythoncom.CoUninitialize()  # Uninitialize COM for this thread

    def list_folders(self, account_folder=None) -> List[str]:
        """
        List available folders in the Outlook account.

        Useful for debugging and UI folder selection.

        Args:
            account_folder: Specific account folder to list (optional)

        Returns:
            List of folder names
        """
        pythoncom.CoInitializeEx(0)

        try:
            outlook = self.connect_to_outlook()
            if account_folder is None:
                account_folder = self._get_account_folder(outlook)

            folders = []
            for i in range(account_folder.Folders.Count):
                folder = account_folder.Folders.Item(i + 1)
                folders.append(folder.Name)

            return folders

        except Exception as e:
            self.logger.error(f"Error listing folders: {e}")
            return []

        finally:
            pythoncom.CoUninitialize()
