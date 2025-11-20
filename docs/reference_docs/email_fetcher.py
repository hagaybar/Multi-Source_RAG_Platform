"""
This module defines the `EmailFetcher` class, used for retrieving emails
from Microsoft Outlook. It connects to a specified account, navigates to a
target folder, fetches emails within a defined number of past days,
and can save them as a TSV or return a DataFrame. Includes basic email body
cleaning functionality.
"""
import os
import win32com.client as win32
import pythoncom # Added for COM library management
import pandas as pd
from datetime import datetime, timedelta
from scripts.utils.logger import LoggerManager


class EmailFetcher:
    """
    Connects to Microsoft Outlook to fetch emails from a specified account and folder.

    This class uses the `win32com.client` library to interact with the local
    Outlook application. It is initialized with a configuration dictionary
    (`config`) which supplies necessary parameters such as:
    - `outlook.account_name`: The Outlook account to use.
    - `outlook.folder_path`: The specific folder path within the account
      (e.g., "Inbox" or "Inbox > MySubfolder").
    - `outlook.days_to_fetch`: The number of past days from which to retrieve emails.
    - `paths.email_dir`: Directory to save fetched email data.
    - `paths.output_file`: Filename for the saved TSV.

    The core functionality is in `fetch_emails_from_folder`, which handles
    connecting to Outlook, navigating to the target folder, filtering emails
    based on the `days_to_fetch` criterion, and extracting relevant data
    (Subject, Sender, Received Time, Body) into a Pandas DataFrame.
    A basic `clean_email_body` method is included as a placeholder for
    text preprocessing. The resulting DataFrame can be saved as a TSV file or
    returned by the method. Logging is performed using a `LoggerManager` instance.
    """
    def __init__(self, config: dict):
        """
        Initializes the EmailFetcher instance.

        Args:
            config (dict): A configuration dictionary containing settings for
                           the Outlook account (config["outlook"]), paths for
                           saving data (config.get("paths", {})), and logging.
                           Extracts account_name, folder_path, days_to_fetch,
                           output_dir, output_file, and log_dir. Initializes a
                           logger and creates output directories if they don't exist.
        """
        self.config = config
        self.logger = LoggerManager.get_logger("EmailFetcher")
        self.account_name = config["outlook"]["account_name"]
        self.folder_path = config["outlook"]["folder_path"]
        self.days = config["outlook"].get("days_to_fetch", 1)

        self.output_dir = config.get("paths", {}).get("email_dir", "data/cleaned")
        self.output_file = config.get("paths", {}).get("output_file", "emails.tsv")
        self.log_dir = config.get("paths", {}).get("log_dir", "logs/email_fetching")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.output_full_path = os.path.join(self.output_dir, self.output_file)

    # def log_action(self, message, logfile_name="email_processing.log"):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     logfile_path = os.path.join(self.log_dir, f"{timestamp}_{logfile_name}")
    #     with open(logfile_path, "a", encoding="utf-8") as log:
    #         log.write(f"{datetime.now()} - {message}\n")

    def connect_to_outlook(self):
        """
        Establishes a connection to the Microsoft Outlook application.

        Uses `win32com.client` to dispatch an Outlook Application COM object
        and get the MAPI namespace.

        Args:
            None.

        Returns:
            win32com.client.Dispatch: An Outlook MAPI namespace object if successful.

        Raises:
            Exception: If the connection to Outlook fails for any reason.
        """
        try:
            outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
            self.logger.info("✅ Connected to Outlook.")
            return outlook
        except Exception as e:
            self.logger.info(f"❌ Outlook connection failed: {e}")
            raise

    def fetch_emails_from_folder(self, return_dataframe: bool = False, save: bool = True):
        pythoncom.CoInitializeEx(0)  # Initialize COM for this thread
        try:
            # Existing logic starts here
            outlook = self.connect_to_outlook()
            if outlook is None: # connect_to_outlook raises an exception on failure, so this check might be redundant
                                # but good for safety if connect_to_outlook changes.
                self.logger.error("Failed to connect to Outlook, cannot fetch emails.")
                # Depending on desired behavior, return empty DataFrame or None
                return pd.DataFrame() if return_dataframe else None

            account_folder = self._get_account_folder(outlook)
            if account_folder is None:
                self.logger.error(f"Account folder for '{self.account_name}' not found.")
                return pd.DataFrame() if return_dataframe else None
            
            target_folder = self._get_target_folder(account_folder)
            if target_folder is None:
                self.logger.error(f"Target folder '{self.folder_path}' not found.")
                return pd.DataFrame() if return_dataframe else None

            cutoff = datetime.now() - timedelta(days=self.days)
            # Ensure target_folder.Items is not None before calling Restrict
            if target_folder.Items is None:
                self.logger.warning(f"Folder '{target_folder.Name}' has no items collection.")
                return pd.DataFrame() if return_dataframe else None

            try: # Inner try for Restrict, as it can fail on some folder types or if Items is empty
                restricted_items_str = f"[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'"
                filtered_items = target_folder.Items.Restrict(restricted_items_str)
                if filtered_items is None: # Restrict can return None
                    self.logger.info(f"No items found after restricting by date: {restricted_items_str}")
                    filtered_items = [] # Ensure it's an iterable
            except Exception as e:
                self.logger.error(f"Error restricting items in folder '{target_folder.Name}': {e}")
                filtered_items = []


            email_data = []
            for item in filtered_items: # filtered_items could be None if Restrict fails gracefully or folder is empty
                if hasattr(item, "Class") and item.Class == 43: # olMailItem
                    raw_subject = item.Subject if hasattr(item, "Subject") else ""
                    # Use item.Body for both Raw Body and as input to clean_email_body
                    raw_body_content = item.Body if hasattr(item, "Body") else ""

                    email_data.append({
                        "Subject": self._sanitize_text_for_tsv(raw_subject),
                        "Sender": item.SenderName if hasattr(item, "SenderName") else "", # SenderName is unlikely to have tabs/newlines
                        "Received": item.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S") if hasattr(item, "ReceivedTime") and item.ReceivedTime else "",
                        "Raw Body": raw_body_content, # Keep raw body as is, assuming it's for inspection, not direct TSV parsing field issue
                        "Cleaned Body": self.clean_email_body(raw_body_content) # clean_email_body now uses _sanitize_text_for_tsv
                    })

            df = pd.DataFrame(email_data)
            self.logger.info(f"Fetched {len(df)} emails from {self.folder_path}")

            out_path = None # Initialize out_path
            if save and not df.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use self.output_full_path which is defined in __init__ if it's meant to be the template
                # Or construct as originally:
                out_path = os.path.join(self.output_dir, f"{timestamp}_{self.output_file}")
                df.to_csv(out_path, sep="	", index=False, encoding="utf-8") # Note: sep="\t" was "	" (tab character)
                self.logger.info(f"Saved to: {out_path}")
                if not return_dataframe:
                    return out_path
            
            return df if return_dataframe else (out_path if save and not df.empty and out_path else None)

        except Exception as e:
            self.logger.error(f"An error occurred in fetch_emails_from_folder: {e}")
            # Re-raise the exception if you want it to propagate or handle it here
            # For now, let's ensure it returns appropriately for different call patterns
            if return_dataframe:
                return pd.DataFrame() # Return empty DataFrame on error
            else:
                return None # Return None if not returning DataFrame
        finally:
            pythoncom.CoUninitialize() # Uninitialize COM for this thread

    def _get_account_folder(self, outlook):
        """
        Retrieves the specified Outlook account folder object.
        extracts data, and optionally saves to TSV or returns a DataFrame.

        The process involves:
        1. Connecting to Outlook.
        2. Navigating to the specified account and target folder.
        3. Filtering emails based on `self.days` (number of past days).
        4. Extracting Subject, Sender, Received Time, Raw Body, and Cleaned Body.
        5. Optionally saving the data to a timestamped TSV file.
        6. Optionally returning the data as a Pandas DataFrame.

        Args:
            return_dataframe (bool, optional): If True, returns the fetched emails
                as a Pandas DataFrame. Defaults to False.
            save (bool, optional): If True and emails are fetched, saves the emails
                to a timestamped TSV file in the configured output directory.
                Defaults to True.

        Returns:
            pd.DataFrame | str | None:
                - If `return_dataframe` is True, returns a Pandas DataFrame of the email data.
                - If `save` is True and `return_dataframe` is False, and emails were saved,
                  returns the path (str) to the saved TSV file.
                - Otherwise, returns `None`.
                - Returns `None` or an empty DataFrame if no emails are fetched or an error occurs.
        """
        # The main logic is now inside the try block in the replacement above.
        # This entire block of original code is replaced by the new try/except/finally structure.
        # connect_to_outlook, _get_account_folder, _get_target_folder are called within the new structure.
        pass # Placeholder, as the original content of this method is moved and wrapped.


    def _get_account_folder(self, outlook):
        """
        Retrieves the specified Outlook account folder object.

        Iterates through the top-level folders in the Outlook namespace to find
        the one matching `self.account_name`.

        Args:
            outlook (win32com.client.Dispatch): The Outlook MAPI namespace object.

        Returns:
            win32com.client.Dispatch: The account folder object.

        Raises:
            ValueError: If the account specified in `self.account_name` is not found.
        """
        for i in range(outlook.Folders.Count):
            folder = outlook.Folders.Item(i + 1)
            if folder.Name == self.account_name:
                return folder
        raise ValueError(f"Account '{self.account_name}' not found")

    def _get_target_folder(self, account_folder):
        """
        Navigates to and returns the target email folder object within the
        given account folder, based on `self.folder_path`.

        The `self.folder_path` is expected to be a string like "Inbox" or
        "Inbox > Subfolder > AnotherSubfolder".

        Args:
            account_folder (win32com.client.Dispatch): The Outlook account folder object.

        Returns:
            win32com.client.Dispatch: The target email folder object.

        Raises:
            Exception: If any part of the `self.folder_path` is not found (propagated
                       from accessing `folder.Folders[name]`).
        """
        folder = account_folder.Folders["Inbox"]
        for name in self.folder_path.split(">"):
            folder = folder.Folders[name]
        return folder

    def clean_email_body(self, body: str) -> str:
        """
        Placeholder method for cleaning the text content of an email body.

        Currently, this method returns the email body as is. It is intended
        to be a point for future enhancements where specific cleaning logic
        (e.g., removing signatures, reply chains, HTML tags if not plain text)
        can be implemented.

        Args:
            body (str): The raw text content of the email body.

        Returns:
            str: The cleaned email body text (currently, the original body).
        """
        # This method might have more domain-specific cleaning in the future (e.g., signature removal).
        # For now, it primarily ensures TSV compatibility.
        return self._sanitize_text_for_tsv(body)

    def _sanitize_text_for_tsv(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Replace tab, newline, and carriage return with a single space
        # Corrected: \r for carriage return
        sanitized_text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        # Replace multiple consecutive spaces with a single space
        return ' '.join(sanitized_text.split())
