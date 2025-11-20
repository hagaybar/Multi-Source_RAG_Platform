"""
WSL Client for Windows Outlook Helper

This module provides a WSL-compatible wrapper that calls the Windows
helper script via subprocess to extract emails from Outlook.

The WSL client uses the same OutlookConfig as the native connector,
making it a drop-in replacement for WSL environments.
"""

import subprocess
import json
import time
from typing import List, Tuple
from pathlib import Path

from scripts.connectors.outlook_helper_utils import (
    OutlookHelperValidator,
    ValidationResult,
    is_outlook_helper_ready
)
from scripts.connectors.outlook_connector import OutlookConfig
from scripts.utils.logger import LoggerManager


class OutlookWSLClient:
    """
    WSL client that calls Windows helper to extract Outlook emails.

    This class provides the same interface as OutlookConnector but works
    from WSL by delegating to the Windows helper script via subprocess.

    Example:
        >>> config = OutlookConfig(
        ...     account_name="user@company.com",
        ...     folder_path="Inbox > Work",
        ...     days_back=30
        ... )
        >>> client = OutlookWSLClient(config)
        >>> emails = client.extract_emails()
        >>> # Returns: [(body_text, metadata), ...]

    Requirements:
        - WSL environment
        - Windows Python with pywin32 installed
        - Helper script deployed to Windows
    """

    def __init__(self, config: OutlookConfig):
        """
        Initialize WSL client.

        Args:
            config: OutlookConfig with extraction parameters
        """
        self.config = config
        self.logger = LoggerManager.get_logger("OutlookWSLClient")

        # Load and validate helper configuration
        self.validator = OutlookHelperValidator()
        self.helper_config = self.validator.config

    def validate(self) -> ValidationResult:
        """
        Validate helper configuration before extraction.

        Returns:
            ValidationResult with status and details
        """
        return self.validator.validate_all()

    def extract_emails(self) -> List[Tuple[str, dict]]:
        """
        Extract emails from Outlook via Windows helper.

        Returns:
            List of (body_text, metadata) tuples

        Raises:
            RuntimeError: If helper validation fails or extraction fails
            subprocess.SubprocessError: If helper execution fails
        """
        # Validate helper before extraction
        self.logger.info("Validating Outlook helper configuration")
        validation = self.validate()

        if not validation.passed:
            error_msg = "Outlook helper validation failed:\n" + "\n".join(validation.errors)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.info("Helper validation passed, starting extraction")

        # Build command
        python_path = self.helper_config["windows"]["python_path"]
        helper_script = self.helper_config["windows"]["helper_script"]

        # Convert python path to WSL so WSL can execute it
        # But keep helper_script as Windows path so Windows Python can read it
        wsl_python_path = self.validator.windows_to_wsl_path(python_path)

        cmd = [
            wsl_python_path,
            helper_script,  # Use Windows path, not WSL path
            "--account", self.config.account_name,
            "--folder", self.config.folder_path,
            "--days", str(self.config.days_back)
        ]

        if self.config.max_emails:
            cmd.extend(["--max-emails", str(self.config.max_emails)])

        self.logger.debug(f"Executing helper: {' '.join(cmd)}")

        # Execute with retry logic
        max_retries = self.helper_config["execution"]["max_retries"]
        timeout = self.helper_config["execution"]["timeout"]
        backoff = self.helper_config["execution"]["retry_backoff"]

        last_error = None
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout
                )

                # Log stderr (structured logs from helper)
                if result.stderr:
                    self._process_helper_logs(result.stderr)

                # Check exit code
                if result.returncode != 0:
                    error_msg = f"Helper exited with code {result.returncode}"
                    self.logger.error(error_msg)
                    raise subprocess.CalledProcessError(
                        result.returncode,
                        cmd,
                        output=result.stdout,
                        stderr=result.stderr
                    )

                # Parse JSON output
                self.logger.info("Parsing helper output")
                try:
                    emails_data = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse helper output: {e}")
                    self.logger.error(f"Helper stdout (first 500 chars): {result.stdout[:500]}")
                    raise RuntimeError("Helper returned invalid JSON") from e

                # Convert to expected format
                emails = []
                for item in emails_data:
                    content = item.get("content", "")
                    metadata = item.get("metadata", {})
                    emails.append((content, metadata))

                self.logger.info(f"Successfully extracted {len(emails)} emails")
                return emails

            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = backoff ** attempt
                    self.logger.warning(
                        f"Helper timeout (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Helper timeout after {max_retries} attempts")
                    raise RuntimeError(f"Helper execution timeout after {timeout}s") from e

            except subprocess.CalledProcessError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = backoff ** attempt
                    self.logger.warning(
                        f"Helper failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Helper failed after {max_retries} attempts")
                    self.logger.error(f"stderr: {e.stderr}")
                    raise RuntimeError(f"Helper execution failed: {e.stderr}") from e

            except json.JSONDecodeError as e:
                # JSON errors are not transient, don't retry
                self.logger.error(f"JSON parsing failed: {e}")
                raise

            except Exception as e:
                # Unexpected errors
                self.logger.error(f"Unexpected error during extraction: {e}")
                raise

        # Should never reach here, but just in case
        raise RuntimeError("Helper extraction failed") from last_error

    def _process_helper_logs(self, stderr: str):
        """
        Process structured logs from helper stderr.

        Args:
            stderr: Helper stderr output (JSON lines)
        """
        for line in stderr.strip().split('\n'):
            if not line:
                continue

            try:
                log_entry = json.loads(line)
                level = log_entry.get("level", "INFO")
                message = log_entry.get("message", line)

                if level == "ERROR":
                    self.logger.error(f"Helper: {message}")
                elif level == "WARNING":
                    self.logger.warning(f"Helper: {message}")
                else:
                    self.logger.info(f"Helper: {message}")
            except json.JSONDecodeError:
                # Not JSON, log as-is
                self.logger.debug(f"Helper: {line}")


def get_outlook_connector(config: OutlookConfig):
    """
    Factory function to get appropriate Outlook connector.

    Returns OutlookWSLClient if running in WSL, otherwise OutlookConnector.

    Args:
        config: OutlookConfig with extraction parameters

    Returns:
        OutlookWSLClient or OutlookConnector instance
    """
    if OutlookHelperValidator.is_wsl():
        # Check if helper is ready
        if is_outlook_helper_ready():
            return OutlookWSLClient(config)
        else:
            raise RuntimeError(
                "Outlook helper not configured. "
                "Please run setup wizard or use: "
                "python scripts/tools/outlook_helper_check.py --auto-fix"
            )
    else:
        # Native Windows environment
        from scripts.connectors.outlook_connector import OutlookConnector
        return OutlookConnector(config)
