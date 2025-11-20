# WSL-Windows Helper Implementation Plan

**Date:** 2025-01-19
**Approved Approach:** Hybrid development (WSL helper) + Docker production (future)
**Focus:** User-friendly setup with validation and guided wizard

---

## Implementation Overview

### Core Principles

1. **Centralized Helper**: Single script at `C:\MultiSourceRAG\tools\win_com_server.py`
2. **Configuration-Driven**: YAML config stores paths, validated on use
3. **Gated UX**: Outlook project creation blocked until deps satisfied
4. **Guided Setup**: Wizard walks through each requirement from Streamlit
5. **Always Validate**: Check dependencies every time Outlook is used
6. **Fail Fast**: Catch issues early with clear error messages

---

## File Structure

```
Multi-Source_RAG_Platform/
├── configs/
│   └── outlook_helper.yaml           # Helper configuration
├── scripts/
│   ├── connectors/
│   │   ├── outlook_connector.py      # Native Windows (existing)
│   │   ├── outlook_wsl_client.py     # NEW: WSL wrapper
│   │   └── outlook_helper_utils.py   # NEW: Validation, detection
│   ├── tools/
│   │   ├── outlook_helper_check.py   # NEW: CLI validation
│   │   └── templates/
│   │       └── win_com_server.py.template  # NEW: Helper script template
│   └── ui/
│       ├── ui_outlook_setup_wizard.py  # NEW: Setup wizard UI
│       └── ui_outlook_manager.py       # MODIFY: Add validation gates
├── tests/
│   ├── connectors/
│   │   └── test_outlook_wsl_client.py  # NEW: WSL client tests
│   └── tools/
│       └── test_outlook_helper_check.py  # NEW: Validation tests
└── docs/
    └── WSL_HELPER_SETUP_GUIDE.md     # NEW: User documentation
```

**Windows Side (Referenced from WSL):**
```
C:\MultiSourceRAG\
└── tools\
    └── win_com_server.py             # Helper script (deployed from template)
```

---

## Phase 1: Configuration & Validation (Foundation)

### 1.1 Configuration Schema

**File:** `configs/outlook_helper.yaml`

```yaml
# Outlook Helper Configuration for WSL
version: "1.0"

windows:
  # Path to Windows Python executable (visible from WSL as /mnt/c/...)
  python_path: "C:/Users/hagay/AppData/Local/Programs/Python/Python311/python.exe"

  # Path to helper script (single centralized location)
  helper_script: "C:/MultiSourceRAG/tools/win_com_server.py"

  # Helper script version (must match deployed script)
  helper_version: "1.0"

execution:
  # Timeout for helper execution (seconds)
  timeout: 60

  # Maximum retries on transient failures
  max_retries: 3

  # Retry backoff (seconds)
  retry_backoff: 2

validation:
  # Auto-detect common Python paths
  auto_detect: true

  # Required packages in Windows Python
  required_packages:
    - pywin32

logging:
  # Enable debug logging for helper calls
  debug: false

  # Log helper stderr to file
  log_stderr: true

# Status (managed by system, do not edit manually)
status:
  last_validated: null
  validation_passed: false
  validation_errors: []
```

**Default Configuration Template:**

**File:** `configs/outlook_helper.yaml.template`

```yaml
# Copy this file to outlook_helper.yaml and configure your paths

version: "1.0"

windows:
  python_path: ""  # Will auto-detect if left empty
  helper_script: "C:/MultiSourceRAG/tools/win_com_server.py"
  helper_version: "1.0"

execution:
  timeout: 60
  max_retries: 3
  retry_backoff: 2

validation:
  auto_detect: true
  required_packages:
    - pywin32

logging:
  debug: false
  log_stderr: true
```

### 1.2 Validation Utilities

**File:** `scripts/connectors/outlook_helper_utils.py`

```python
"""
Utilities for validating and managing the Windows Outlook helper.

Provides:
- Environment detection (is_wsl, is_windows)
- Path translation (WSL ↔ Windows)
- Dependency validation (Python, pywin32, helper script)
- Configuration management
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of helper validation check."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, str]


class OutlookHelperValidator:
    """Validates Windows helper configuration and dependencies."""

    def __init__(self, config_path: Path = None):
        """
        Initialize validator with configuration.

        Args:
            config_path: Path to outlook_helper.yaml (optional)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "outlook_helper.yaml"

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Create from template
            template_path = self.config_path.parent / "outlook_helper.yaml.template"
            if template_path.exists():
                import shutil
                shutil.copy(template_path, self.config_path)
            else:
                # Create minimal config
                return self._get_default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "version": "1.0",
            "windows": {
                "python_path": "",
                "helper_script": "C:/MultiSourceRAG/tools/win_com_server.py",
                "helper_version": "1.0"
            },
            "execution": {
                "timeout": 60,
                "max_retries": 3,
                "retry_backoff": 2
            },
            "validation": {
                "auto_detect": True,
                "required_packages": ["pywin32"]
            }
        }

    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

    def validate_all(self) -> ValidationResult:
        """
        Run all validation checks.

        Returns:
            ValidationResult with passed status and details
        """
        errors = []
        warnings = []
        info = {}

        # 1. Check environment
        if not self.is_wsl():
            errors.append("Not running in WSL. Outlook helper requires WSL environment.")
            return ValidationResult(False, errors, warnings, info)

        info["environment"] = "WSL2"

        # 2. Check /mnt/c/ access
        if not self.can_access_windows_filesystem():
            errors.append("Cannot access Windows filesystem (/mnt/c/). Check WSL configuration.")
            return ValidationResult(False, errors, warnings, info)

        info["windows_filesystem"] = "Accessible"

        # 3. Detect or validate Windows Python
        python_path = self.config["windows"]["python_path"]
        if not python_path and self.config["validation"]["auto_detect"]:
            python_path = self.auto_detect_windows_python()
            if python_path:
                self.config["windows"]["python_path"] = python_path
                self.save_config()
                info["python_detection"] = "Auto-detected"
            else:
                errors.append("Could not auto-detect Windows Python. Please set manually in config.")
                return ValidationResult(False, errors, warnings, info)

        # 4. Validate Windows Python exists
        if not self.validate_windows_python(python_path):
            errors.append(f"Windows Python not found at: {python_path}")
            suggestions = self.suggest_python_paths()
            if suggestions:
                errors.append(f"Suggestions: {', '.join(suggestions)}")
            return ValidationResult(False, errors, warnings, info)

        info["python_path"] = python_path

        # 5. Check Python version
        version = self.get_python_version(python_path)
        if version:
            info["python_version"] = version
            if not self.is_python_version_compatible(version):
                warnings.append(f"Python version {version} may not be compatible. Recommended: 3.11+")
        else:
            errors.append("Could not determine Python version")
            return ValidationResult(False, errors, warnings, info)

        # 6. Check required packages (pywin32)
        missing_packages = self.check_required_packages(python_path)
        if missing_packages:
            errors.append(f"Missing required packages in Windows Python: {', '.join(missing_packages)}")
            errors.append(f"Install with: {python_path} -m pip install {' '.join(missing_packages)}")
            return ValidationResult(False, errors, warnings, info)

        info["required_packages"] = "Installed"

        # 7. Validate helper script exists
        helper_path = self.config["windows"]["helper_script"]
        if not self.validate_helper_script(helper_path):
            errors.append(f"Helper script not found at: {helper_path}")
            errors.append("Use setup wizard to deploy helper script")
            return ValidationResult(False, errors, warnings, info)

        info["helper_script"] = helper_path

        # 8. Check helper version
        script_version = self.get_helper_version(helper_path)
        expected_version = self.config["windows"]["helper_version"]
        if script_version != expected_version:
            warnings.append(
                f"Helper script version mismatch: found {script_version}, expected {expected_version}"
            )
        info["helper_version"] = script_version

        # 9. Run self-test
        self_test_result = self.run_helper_self_test(python_path, helper_path)
        if not self_test_result:
            errors.append("Helper self-test failed. Check Outlook installation and permissions.")
            return ValidationResult(False, errors, warnings, info)

        info["self_test"] = "Passed"

        # Update config with validation status
        self.config["status"] = {
            "last_validated": datetime.now().isoformat(),
            "validation_passed": True,
            "validation_errors": []
        }
        self.save_config()

        return ValidationResult(
            passed=True,
            errors=errors,
            warnings=warnings,
            info=info
        )

    @staticmethod
    def is_wsl() -> bool:
        """Check if running in WSL."""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False

    @staticmethod
    def can_access_windows_filesystem() -> bool:
        """Check if Windows filesystem is accessible."""
        return os.path.exists('/mnt/c/')

    @staticmethod
    def wsl_to_windows_path(wsl_path: str) -> str:
        """
        Convert WSL path to Windows path.

        Examples:
            /mnt/c/Users/hagay → C:/Users/hagay
            /home/user/file.py → (unchanged)
        """
        if wsl_path.startswith('/mnt/'):
            # /mnt/c/... → C:/...
            drive = wsl_path[5]  # Get drive letter
            rest = wsl_path[7:]  # Skip /mnt/c/
            return f"{drive.upper()}:/{rest}"
        return wsl_path

    @staticmethod
    def windows_to_wsl_path(win_path: str) -> str:
        """
        Convert Windows path to WSL path.

        Examples:
            C:/Users/hagay → /mnt/c/Users/hagay
            C:\\Users\\hagay → /mnt/c/Users/hagay
        """
        # Normalize backslashes
        win_path = win_path.replace('\\', '/')

        if len(win_path) >= 2 and win_path[1] == ':':
            drive = win_path[0].lower()
            rest = win_path[3:] if len(win_path) > 2 else ''
            return f"/mnt/{drive}/{rest}"
        return win_path

    def auto_detect_windows_python(self) -> Optional[str]:
        """
        Auto-detect Windows Python installation.

        Checks common installation locations.

        Returns:
            Windows path to python.exe if found, None otherwise
        """
        import getpass
        username = getpass.getuser()

        common_paths = [
            f"C:/Users/{username}/AppData/Local/Programs/Python/Python311/python.exe",
            f"C:/Users/{username}/AppData/Local/Programs/Python/Python312/python.exe",
            f"C:/Users/{username}/AppData/Local/Programs/Python/Python313/python.exe",
            "C:/Python311/python.exe",
            "C:/Python312/python.exe",
            "C:/Python313/python.exe",
            "C:/Program Files/Python311/python.exe",
            "C:/Program Files/Python312/python.exe",
        ]

        for path in common_paths:
            if self.validate_windows_python(path):
                return path

        return None

    def suggest_python_paths(self) -> List[str]:
        """Suggest possible Python installation paths."""
        import getpass
        username = getpass.getuser()

        return [
            f"C:/Users/{username}/AppData/Local/Programs/Python/Python3XX/python.exe",
            "C:/Python3XX/python.exe",
            "C:/Program Files/Python3XX/python.exe"
        ]

    def validate_windows_python(self, python_path: str) -> bool:
        """
        Check if Windows Python executable exists.

        Args:
            python_path: Windows path to python.exe

        Returns:
            True if exists and executable
        """
        wsl_path = self.windows_to_wsl_path(python_path)
        return os.path.exists(wsl_path) and os.access(wsl_path, os.X_OK)

    def get_python_version(self, python_path: str) -> Optional[str]:
        """
        Get Python version from Windows Python.

        Args:
            python_path: Windows path to python.exe

        Returns:
            Version string (e.g., "3.11.2") or None
        """
        try:
            result = subprocess.run(
                [python_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Output: "Python 3.11.2"
                return result.stdout.strip().split()[-1]
        except:
            pass
        return None

    @staticmethod
    def is_python_version_compatible(version: str) -> bool:
        """Check if Python version is compatible (>= 3.11)."""
        try:
            major, minor = map(int, version.split('.')[:2])
            return (major, minor) >= (3, 11)
        except:
            return False

    def check_required_packages(self, python_path: str) -> List[str]:
        """
        Check if required packages are installed in Windows Python.

        Args:
            python_path: Windows path to python.exe

        Returns:
            List of missing package names
        """
        required = self.config["validation"]["required_packages"]
        missing = []

        for package in required:
            try:
                result = subprocess.run(
                    [python_path, "-c", f"import {package}"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode != 0:
                    missing.append(package)
            except:
                missing.append(package)

        return missing

    def validate_helper_script(self, helper_path: str) -> bool:
        """
        Check if helper script exists.

        Args:
            helper_path: Windows path to helper script

        Returns:
            True if exists
        """
        wsl_path = self.windows_to_wsl_path(helper_path)
        return os.path.exists(wsl_path)

    def get_helper_version(self, helper_path: str) -> Optional[str]:
        """
        Get version from helper script.

        Args:
            helper_path: Windows path to helper script

        Returns:
            Version string or None
        """
        wsl_path = self.windows_to_wsl_path(helper_path)
        try:
            with open(wsl_path, 'r') as f:
                for line in f:
                    if line.startswith('HELPER_VERSION'):
                        # HELPER_VERSION = "1.0"
                        return line.split('=')[1].strip().strip('"\'')
        except:
            pass
        return None

    def run_helper_self_test(self, python_path: str, helper_path: str) -> bool:
        """
        Run helper script self-test.

        Args:
            python_path: Windows path to python.exe
            helper_path: Windows path to helper script

        Returns:
            True if self-test passed
        """
        try:
            result = subprocess.run(
                [python_path, helper_path, "--self-test"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False


# Convenience functions

def validate_outlook_helper() -> ValidationResult:
    """Validate Outlook helper configuration."""
    validator = OutlookHelperValidator()
    return validator.validate_all()


def is_outlook_helper_ready() -> bool:
    """Quick check if Outlook helper is ready to use."""
    result = validate_outlook_helper()
    return result.passed
```

---

## Phase 2: Windows Helper Script Template

### 2.1 Helper Script Template

**File:** `scripts/tools/templates/win_com_server.py.template`

```python
"""
Windows Outlook Email Extraction Helper

This script runs on Windows and extracts emails from Outlook via COM.
It is called from WSL via subprocess and outputs JSON to stdout.

Version: 1.0
"""

import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Version (must match outlook_helper.yaml)
HELPER_VERSION = "1.0"

try:
    import win32com.client
    import pythoncom
    COM_AVAILABLE = True
except ImportError:
    COM_AVAILABLE = False


def log_error(message: str, **kwargs):
    """Log error to stderr in JSON format."""
    log_entry = {
        "level": "ERROR",
        "message": message,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    print(json.dumps(log_entry), file=sys.stderr)


def log_info(message: str, **kwargs):
    """Log info to stderr in JSON format."""
    log_entry = {
        "level": "INFO",
        "message": message,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    print(json.dumps(log_entry), file=sys.stderr)


def self_test() -> bool:
    """
    Run self-test to validate environment.

    Returns:
        True if all checks passed
    """
    # Check COM availability
    if not COM_AVAILABLE:
        log_error("pywin32 not installed")
        return False

    log_info("pywin32: OK")

    # Try to initialize COM
    try:
        pythoncom.CoInitializeEx(0)
        log_info("COM initialization: OK")
    except Exception as e:
        log_error(f"COM initialization failed: {e}")
        return False
    finally:
        try:
            pythoncom.CoUninitialize()
        except:
            pass

    # Try to connect to Outlook
    try:
        pythoncom.CoInitializeEx(0)
        outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
        log_info("Outlook connection: OK")

        # Check for accounts
        account_count = outlook.Folders.Count
        log_info(f"Outlook accounts found: {account_count}")

        pythoncom.CoUninitialize()
    except Exception as e:
        log_error(f"Outlook connection failed: {e}")
        try:
            pythoncom.CoUninitialize()
        except:
            pass
        return False

    log_info("Self-test: PASSED")
    return True


def extract_emails(
    account_name: str,
    folder_path: str,
    days_back: int,
    max_emails: int = None
) -> List[Tuple[str, Dict]]:
    """
    Extract emails from Outlook.

    Args:
        account_name: Outlook account name
        folder_path: Folder path (e.g., "Inbox" or "Inbox > Work")
        days_back: Number of days to look back
        max_emails: Maximum emails to extract (None = no limit)

    Returns:
        List of (body_text, metadata) tuples
    """
    if not COM_AVAILABLE:
        raise ImportError("pywin32 not installed")

    pythoncom.CoInitializeEx(0)

    try:
        # Connect to Outlook
        outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
        log_info(f"Connected to Outlook")

        # Find account
        account_folder = None
        for i in range(outlook.Folders.Count):
            folder = outlook.Folders.Item(i + 1)
            if folder.Name == account_name:
                account_folder = folder
                break

        if account_folder is None:
            raise ValueError(f"Account '{account_name}' not found")

        log_info(f"Found account: {account_name}")

        # Navigate to target folder
        folder_parts = [part.strip() for part in folder_path.split(">")]
        current_folder = account_folder.Folders["Inbox"]

        if len(folder_parts) > 1 or (len(folder_parts) == 1 and folder_parts[0].lower() != "inbox"):
            start_index = 1 if folder_parts[0].lower() == "inbox" else 0
            for folder_name in folder_parts[start_index:]:
                current_folder = current_folder.Folders[folder_name]

        log_info(f"Found folder: {folder_path}")

        # Filter by date
        cutoff = datetime.now() - timedelta(days=days_back)
        filter_str = f"[ReceivedTime] >= '{cutoff.strftime('%m/%d/%Y %H:%M %p')}'"
        filtered_items = current_folder.Items.Restrict(filter_str)

        log_info(f"Found {len(filtered_items)} emails in date range")

        # Extract emails
        emails = []
        for item in filtered_items:
            if hasattr(item, "Class") and item.Class == 43:  # olMailItem
                try:
                    # Get email body
                    body = item.Body if hasattr(item, "Body") else ""

                    # Build metadata
                    metadata = {
                        "source_filepath": f"outlook://{account_name}/{folder_path}",
                        "content_type": "email",
                        "doc_type": "outlook_eml",
                        "subject": item.Subject if hasattr(item, "Subject") else "",
                        "sender": item.SenderEmailAddress if hasattr(item, "SenderEmailAddress") else "",
                        "sender_name": item.SenderName if hasattr(item, "SenderName") else "",
                        "date": item.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S") if hasattr(item, "ReceivedTime") and item.ReceivedTime else "",
                        "message_id": item.EntryID if hasattr(item, "EntryID") else "",
                    }

                    emails.append((body, metadata))

                    if max_emails and len(emails) >= max_emails:
                        break

                except Exception as e:
                    log_error(f"Failed to process email: {e}")
                    continue

        log_info(f"Extracted {len(emails)} emails")
        return emails

    finally:
        pythoncom.CoUninitialize()


def main():
    parser = argparse.ArgumentParser(description="Windows Outlook Email Extraction Helper")
    parser.add_argument("--version", action="version", version=f"%(prog)s {HELPER_VERSION}")
    parser.add_argument("--self-test", action="store_true", help="Run self-test and exit")
    parser.add_argument("--account", help="Outlook account name")
    parser.add_argument("--folder", help="Folder path (e.g., 'Inbox' or 'Inbox > Work')")
    parser.add_argument("--days", type=int, default=30, help="Days to look back (default: 30)")
    parser.add_argument("--max-emails", type=int, help="Maximum emails to extract")

    args = parser.parse_args()

    # Self-test mode
    if args.self_test:
        success = self_test()
        sys.exit(0 if success else 1)

    # Extraction mode
    if not args.account or not args.folder:
        print("Error: --account and --folder are required", file=sys.stderr)
        sys.exit(1)

    try:
        emails = extract_emails(
            account_name=args.account,
            folder_path=args.folder,
            days_back=args.days,
            max_emails=args.max_emails
        )

        # Output JSON to stdout
        output = []
        for body, metadata in emails:
            output.append({
                "content": body,
                "metadata": metadata
            })

        print(json.dumps(output, ensure_ascii=False))
        sys.exit(0)

    except Exception as e:
        log_error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Phase 3: WSL Client Wrapper

### 3.1 WSL Client Implementation

**File:** `scripts/connectors/outlook_wsl_client.py`

```python
"""
WSL Client for Windows Outlook Helper

This module provides a WSL-compatible wrapper that calls the Windows
helper script via subprocess to extract emails from Outlook.
"""

import subprocess
import json
import time
import shlex
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
    """WSL client that calls Windows helper to extract Outlook emails."""

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
            RuntimeError: If helper validation fails
            subprocess.SubprocessError: If helper execution fails
        """
        # Validate helper before extraction
        validation = self.validate()
        if not validation.passed:
            error_msg = "Outlook helper validation failed:\n" + "\n".join(validation.errors)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.info("Helper validation passed, starting extraction")

        # Build command
        python_path = self.helper_config["windows"]["python_path"]
        helper_script = self.helper_config["windows"]["helper_script"]

        cmd = [
            python_path,
            helper_script,
            "--account", self.config.account_name,
            "--folder", self.config.folder_path,
            "--days", str(self.config.days_back)
        ]

        if self.config.max_emails:
            cmd.extend(["--max-emails", str(self.config.max_emails)])

        self.logger.debug(f"Executing command: {' '.join(cmd)}")

        # Execute with retry logic
        max_retries = self.helper_config["execution"]["max_retries"]
        timeout = self.helper_config["execution"]["timeout"]
        backoff = self.helper_config["execution"]["retry_backoff"]

        last_error = None
        for attempt in range(max_retries):
            try:
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
                    raise subprocess.CalledProcessError(
                        result.returncode,
                        cmd,
                        output=result.stdout,
                        stderr=result.stderr
                    )

                # Parse JSON output
                emails_data = json.loads(result.stdout)

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
                    self.logger.warning(f"Helper timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(backoff ** attempt)
                else:
                    self.logger.error(f"Helper timeout after {max_retries} attempts")
                    raise RuntimeError(f"Helper execution timeout after {timeout}s") from e

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse helper output: {e}")
                self.logger.error(f"Helper stdout: {result.stdout[:500]}")
                raise RuntimeError("Helper returned invalid JSON") from e

            except subprocess.CalledProcessError as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Helper failed (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(backoff ** attempt)
                else:
                    self.logger.error(f"Helper failed after {max_retries} attempts")
                    self.logger.error(f"stderr: {e.stderr}")
                    raise RuntimeError(f"Helper execution failed: {e.stderr}") from e

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
```

---

## Phase 4: UI Setup Wizard

### 4.1 Setup Wizard UI

**File:** `scripts/ui/ui_outlook_setup_wizard.py`

```python
"""
Outlook Helper Setup Wizard for Streamlit UI

Guides users through setting up the Windows helper for Outlook
extraction from WSL.
"""

import streamlit as st
import shutil
from pathlib import Path
from scripts.connectors.outlook_helper_utils import (
    OutlookHelperValidator,
    ValidationResult
)


def render_outlook_setup_wizard():
    """
    Render Outlook helper setup wizard.

    Guides user through:
    1. Environment detection
    2. Python path configuration
    3. Helper script deployment
    4. Dependency installation
    5. Validation
    """
    st.subheader("🔧 Outlook Helper Setup Wizard")

    validator = OutlookHelperValidator()

    # Initialize session state
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0

    # Step 0: Environment Check
    if st.session_state.wizard_step == 0:
        _render_environment_check(validator)

    # Step 1: Python Path
    elif st.session_state.wizard_step == 1:
        _render_python_config(validator)

    # Step 2: Helper Script Deployment
    elif st.session_state.wizard_step == 2:
        _render_helper_deployment(validator)

    # Step 3: Dependencies
    elif st.session_state.wizard_step == 3:
        _render_dependency_check(validator)

    # Step 4: Validation
    elif st.session_state.wizard_step == 4:
        _render_final_validation(validator)

    # Step 5: Complete
    elif st.session_state.wizard_step == 5:
        _render_completion()


def _render_environment_check(validator: OutlookHelperValidator):
    """Step 0: Check WSL environment."""
    st.markdown("### Step 1: Environment Detection")

    if not validator.is_wsl():
        st.error("❌ Not running in WSL")
        st.info(
            "💡 This wizard is for WSL users. If you're on Windows, "
            "use the native Outlook connector instead."
        )
        return

    st.success("✅ Running in WSL2")

    if not validator.can_access_windows_filesystem():
        st.error("❌ Cannot access Windows filesystem (/mnt/c/)")
        st.code("# Try remounting:\nsudo mkdir -p /mnt/c\nsudo mount -t drvfs C: /mnt/c")
        return

    st.success("✅ Windows filesystem accessible")

    if st.button("Next: Configure Python Path"):
        st.session_state.wizard_step = 1
        st.rerun()


def _render_python_config(validator: OutlookHelperValidator):
    """Step 1: Configure Windows Python path."""
    st.markdown("### Step 2: Windows Python Configuration")

    # Auto-detection
    st.info("🔍 Attempting auto-detection...")
    detected_path = validator.auto_detect_windows_python()

    if detected_path:
        st.success(f"✅ Found Python at: `{detected_path}`")
        version = validator.get_python_version(detected_path)
        if version:
            st.info(f"Version: {version}")

        if st.button("Use This Python"):
            validator.config["windows"]["python_path"] = detected_path
            validator.save_config()
            st.session_state.wizard_step = 2
            st.rerun()

    # Manual entry
    st.markdown("---")
    st.markdown("**Or enter manually:**")

    python_path = st.text_input(
        "Windows Python Path",
        value=validator.config["windows"]["python_path"],
        placeholder="C:/Users/YourName/AppData/Local/Programs/Python/Python311/python.exe"
    )

    if python_path:
        if validator.validate_windows_python(python_path):
            st.success("✅ Python executable found")
            version = validator.get_python_version(python_path)
            if version:
                st.info(f"Version: {version}")

            if st.button("Save and Continue"):
                validator.config["windows"]["python_path"] = python_path
                validator.save_config()
                st.session_state.wizard_step = 2
                st.rerun()
        else:
            st.error("❌ Python not found at this path")
            st.info("💡 Suggestions:")
            for suggestion in validator.suggest_python_paths():
                st.code(suggestion)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.wizard_step = 0
            st.rerun()


def _render_helper_deployment(validator: OutlookHelperValidator):
    """Step 2: Deploy helper script."""
    st.markdown("### Step 3: Deploy Helper Script")

    helper_path = validator.config["windows"]["helper_script"]
    st.info(f"📂 Target location: `{helper_path}`")

    # Check if already exists
    if validator.validate_helper_script(helper_path):
        st.success("✅ Helper script already exists")

        version = validator.get_helper_version(helper_path)
        if version:
            st.info(f"Version: {version}")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Re-deploy"):
                if _deploy_helper_script(helper_path):
                    st.success("✅ Re-deployed successfully")
                    time.sleep(1)
                    st.rerun()
        with col3:
            if st.button("Next →"):
                st.session_state.wizard_step = 3
                st.rerun()
    else:
        st.warning("⚠️ Helper script not found")

        if st.button("📥 Deploy Helper Script"):
            with st.spinner("Deploying helper script..."):
                if _deploy_helper_script(helper_path):
                    st.success("✅ Deployed successfully!")
                    time.sleep(1)
                    st.session_state.wizard_step = 3
                    st.rerun()
                else:
                    st.error("❌ Deployment failed")

        if st.button("← Back"):
            st.session_state.wizard_step = 1
            st.rerun()


def _deploy_helper_script(helper_path: str) -> bool:
    """
    Deploy helper script to Windows location.

    Args:
        helper_path: Windows path to helper script

    Returns:
        True if successful
    """
    try:
        # Get template
        template_path = Path(__file__).parent.parent / "tools" / "templates" / "win_com_server.py.template"

        if not template_path.exists():
            st.error(f"Template not found: {template_path}")
            return False

        # Convert to WSL path
        from scripts.connectors.outlook_helper_utils import OutlookHelperValidator
        wsl_path = OutlookHelperValidator.windows_to_wsl_path(helper_path)

        # Create directory if needed
        Path(wsl_path).parent.mkdir(parents=True, exist_ok=True)

        # Copy template
        shutil.copy(template_path, wsl_path)

        return True

    except Exception as e:
        st.error(f"Deployment error: {e}")
        return False


def _render_dependency_check(validator: OutlookHelperValidator):
    """Step 3: Check dependencies."""
    st.markdown("### Step 4: Check Dependencies")

    python_path = validator.config["windows"]["python_path"]
    missing = validator.check_required_packages(python_path)

    if not missing:
        st.success("✅ All required packages installed")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back"):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("Next: Final Validation"):
                st.session_state.wizard_step = 4
                st.rerun()
    else:
        st.error(f"❌ Missing packages: {', '.join(missing)}")

        st.markdown("**Installation Instructions:**")
        st.code(
            f"{python_path} -m pip install {' '.join(missing)}",
            language="powershell"
        )

        st.info(
            "💡 Copy the command above and run it in Windows PowerShell, "
            "then click 'Check Again'"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back"):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("🔄 Check Again"):
                st.rerun()


def _render_final_validation(validator: OutlookHelperValidator):
    """Step 4: Final validation."""
    st.markdown("### Step 5: Final Validation")

    if st.button("🧪 Run Full Validation"):
        with st.spinner("Running validation..."):
            result = validator.validate_all()

            if result.passed:
                st.success("✅ All checks passed!")

                # Show details
                with st.expander("📋 Validation Details"):
                    for key, value in result.info.items():
                        st.write(f"**{key}**: {value}")

                if result.warnings:
                    st.warning("⚠️ Warnings:")
                    for warning in result.warnings:
                        st.warning(warning)

                time.sleep(2)
                st.session_state.wizard_step = 5
                st.rerun()
            else:
                st.error("❌ Validation failed")

                for error in result.errors:
                    st.error(error)

                if st.button("← Back to Fix Issues"):
                    st.session_state.wizard_step = 0
                    st.rerun()

    if st.button("← Back"):
        st.session_state.wizard_step = 3
        st.rerun()


def _render_completion():
    """Step 5: Setup complete."""
    st.markdown("### ✅ Setup Complete!")

    st.success(
        "Your Outlook helper is configured and ready to use. "
        "You can now create Outlook-based projects and extract emails."
    )

    st.info(
        "💡 **Next Steps:**\n"
        "1. Go to the Outlook Integration tab\n"
        "2. Create an Outlook project\n"
        "3. Extract emails from your Outlook folders"
    )

    if st.button("🔄 Reset Wizard"):
        st.session_state.wizard_step = 0
        st.rerun()
```

---

## Phase 5: UI Integration & Gating

### 5.1 Modify Outlook Manager with Validation Gates

**File:** `scripts/ui/ui_outlook_manager.py` (MODIFY)

Add validation checks that gate Outlook features until helper is ready.

**Changes:**

```python
import streamlit as st
from scripts.connectors.outlook_helper_utils import (
    is_outlook_helper_ready,
    validate_outlook_helper,
    OutlookHelperValidator
)
from scripts.ui.ui_outlook_setup_wizard import render_outlook_setup_wizard


def render_outlook_integration():
    """Render Outlook integration UI with validation gates."""
    st.title("📧 Outlook Integration")

    # Check if running in WSL
    if OutlookHelperValidator.is_wsl():
        # WSL mode - check helper readiness
        _render_wsl_outlook_integration()
    else:
        # Native Windows mode
        _render_windows_outlook_integration()


def _render_wsl_outlook_integration():
    """Render WSL-specific Outlook UI with helper validation."""
    st.info("🐧 Running in WSL - Using Windows Helper Bridge")

    # Check helper status
    helper_ready = is_outlook_helper_ready()

    if not helper_ready:
        # Show setup wizard if not ready
        st.warning(
            "⚠️ **Outlook Helper Not Configured**\n\n"
            "To use Outlook integration from WSL, you need to set up "
            "the Windows helper bridge. This wizard will guide you through the process."
        )

        # Show setup wizard
        render_outlook_setup_wizard()

        # Don't show main UI until setup complete
        return

    # Helper is ready - show validation status
    with st.expander("🔍 Helper Status", expanded=False):
        validation = validate_outlook_helper()

        st.success("✅ Outlook helper is ready")

        for key, value in validation.info.items():
            st.write(f"**{key}**: {value}")

        if validation.warnings:
            st.warning("⚠️ Warnings:")
            for warning in validation.warnings:
                st.warning(warning)

        if st.button("🔄 Re-validate"):
            st.rerun()

        if st.button("⚙️ Re-run Setup Wizard"):
            st.session_state.wizard_step = 0
            st.rerun()

    # Show main Outlook UI
    _render_outlook_ui()


def _render_windows_outlook_integration():
    """Render native Windows Outlook UI."""
    st.info("🪟 Running on Windows - Using Native COM Integration")
    _render_outlook_ui()


def _render_outlook_ui():
    """Render main Outlook integration UI (common to both modes)."""
    # Existing Outlook UI code here
    # (Project creation, email preview, extraction, etc.)
    pass
```

**Key Changes:**
1. Auto-detect environment (WSL vs Windows)
2. Gate Outlook features behind validation check
3. Show setup wizard if helper not ready
4. Display validation status with option to re-validate
5. Provide quick access to re-run wizard

---

## Phase 6: CLI Validation Tool

### 6.1 Command-Line Validation Tool

**File:** `scripts/tools/outlook_helper_check.py`

CLI tool for validating helper configuration (useful for CI/testing).

```python
#!/usr/bin/env python3
"""
Command-line tool to validate Outlook helper configuration.

Usage:
    python scripts/tools/outlook_helper_check.py
    python scripts/tools/outlook_helper_check.py --config configs/outlook_helper.yaml
    python scripts/tools/outlook_helper_check.py --auto-fix
"""

import sys
import argparse
from pathlib import Path
from scripts.connectors.outlook_helper_utils import (
    OutlookHelperValidator,
    ValidationResult
)


def print_result(result: ValidationResult, verbose: bool = False):
    """Print validation result in human-readable format."""
    if result.passed:
        print("✅ VALIDATION PASSED")
        print()

        if verbose or result.info:
            print("📋 Details:")
            for key, value in result.info.items():
                print(f"  • {key}: {value}")
            print()

        if result.warnings:
            print("⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  • {warning}")
            print()
    else:
        print("❌ VALIDATION FAILED")
        print()

        if result.errors:
            print("🚫 Errors:")
            for error in result.errors:
                print(f"  • {error}")
            print()

        if result.info:
            print("📋 Details:")
            for key, value in result.info.items():
                print(f"  • {key}: {value}")
            print()


def auto_fix(validator: OutlookHelperValidator) -> bool:
    """
    Attempt to auto-fix common issues.

    Returns:
        True if fixes applied successfully
    """
    print("🔧 Attempting auto-fix...")
    print()

    fixed = False

    # 1. Auto-detect Python if missing
    if not validator.config["windows"]["python_path"]:
        print("🔍 Auto-detecting Windows Python...")
        python_path = validator.auto_detect_windows_python()
        if python_path:
            validator.config["windows"]["python_path"] = python_path
            validator.save_config()
            print(f"✅ Set Python path: {python_path}")
            fixed = True
        else:
            print("❌ Could not auto-detect Python")

    # 2. Deploy helper script if missing
    helper_path = validator.config["windows"]["helper_script"]
    if not validator.validate_helper_script(helper_path):
        print(f"📥 Deploying helper script to {helper_path}...")

        # Import deployment function
        from scripts.ui.ui_outlook_setup_wizard import _deploy_helper_script
        if _deploy_helper_script(helper_path):
            print("✅ Helper script deployed")
            fixed = True
        else:
            print("❌ Failed to deploy helper script")

    if fixed:
        print()
        print("✅ Auto-fix completed. Re-run validation to check.")
    else:
        print("ℹ️  No fixes applied (or all issues require manual intervention)")

    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="Validate Outlook helper configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python scripts/tools/outlook_helper_check.py

  # Verbose output
  python scripts/tools/outlook_helper_check.py -v

  # Use custom config
  python scripts/tools/outlook_helper_check.py --config /path/to/config.yaml

  # Attempt auto-fix
  python scripts/tools/outlook_helper_check.py --auto-fix

  # JSON output (for CI)
  python scripts/tools/outlook_helper_check.py --json
        """
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to outlook_helper.yaml (default: configs/outlook_helper.yaml)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Attempt to auto-fix common issues"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for CI/automation)"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = OutlookHelperValidator(config_path=args.config)

    # Auto-fix if requested
    if args.auto_fix:
        auto_fix(validator)
        print()

    # Run validation
    if args.verbose:
        print("🔍 Running validation...")
        print()

    result = validator.validate_all()

    # Output results
    if args.json:
        import json
        output = {
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
            "info": result.info
        }
        print(json.dumps(output, indent=2))
    else:
        print_result(result, verbose=args.verbose)

    # Exit code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
```

**Usage Examples:**

```bash
# Basic validation
python scripts/tools/outlook_helper_check.py

# Output:
# ✅ VALIDATION PASSED
#
# 📋 Details:
#   • environment: WSL2
#   • windows_filesystem: Accessible
#   • python_path: C:/Users/hagay/.../python.exe
#   • python_version: 3.11.2
#   • required_packages: Installed
#   • helper_script: C:/MultiSourceRAG/tools/win_com_server.py
#   • helper_version: 1.0
#   • self_test: Passed

# Verbose output
python scripts/tools/outlook_helper_check.py -v

# Auto-fix common issues
python scripts/tools/outlook_helper_check.py --auto-fix

# JSON output (for CI/scripts)
python scripts/tools/outlook_helper_check.py --json
```

---

## Phase 7: Testing Strategy

### 7.1 Unit Tests

**File:** `tests/connectors/test_outlook_helper_utils.py`

```python
"""Unit tests for Outlook helper utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from scripts.connectors.outlook_helper_utils import (
    OutlookHelperValidator,
    ValidationResult,
    is_outlook_helper_ready
)


class TestPathTranslation:
    """Test path translation utilities."""

    def test_wsl_to_windows_path(self):
        """Test WSL to Windows path conversion."""
        assert OutlookHelperValidator.wsl_to_windows_path(
            "/mnt/c/Users/hagay/test.py"
        ) == "C:/Users/hagay/test.py"

        assert OutlookHelperValidator.wsl_to_windows_path(
            "/mnt/d/Projects/test.py"
        ) == "D:/Projects/test.py"

        # Non-/mnt paths unchanged
        assert OutlookHelperValidator.wsl_to_windows_path(
            "/home/user/test.py"
        ) == "/home/user/test.py"

    def test_windows_to_wsl_path(self):
        """Test Windows to WSL path conversion."""
        assert OutlookHelperValidator.windows_to_wsl_path(
            "C:/Users/hagay/test.py"
        ) == "/mnt/c/Users/hagay/test.py"

        assert OutlookHelperValidator.windows_to_wsl_path(
            "C:\\Users\\hagay\\test.py"
        ) == "/mnt/c/Users/hagay/test.py"


class TestEnvironmentDetection:
    """Test environment detection."""

    @patch("builtins.open", create=True)
    def test_is_wsl_true(self, mock_open):
        """Test WSL detection returns True for WSL."""
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "Linux version 5.15.0-microsoft-standard-WSL2"
        )

        assert OutlookHelperValidator.is_wsl() is True

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_is_wsl_false_no_proc(self, mock_open):
        """Test WSL detection returns False when /proc/version missing."""
        assert OutlookHelperValidator.is_wsl() is False

    @patch("os.path.exists")
    def test_can_access_windows_filesystem(self, mock_exists):
        """Test Windows filesystem access detection."""
        mock_exists.return_value = True
        assert OutlookHelperValidator.can_access_windows_filesystem() is True

        mock_exists.return_value = False
        assert OutlookHelperValidator.can_access_windows_filesystem() is False


class TestPythonValidation:
    """Test Python validation."""

    def test_is_python_version_compatible(self):
        """Test Python version compatibility check."""
        assert OutlookHelperValidator.is_python_version_compatible("3.11.2") is True
        assert OutlookHelperValidator.is_python_version_compatible("3.12.0") is True
        assert OutlookHelperValidator.is_python_version_compatible("3.10.0") is False
        assert OutlookHelperValidator.is_python_version_compatible("2.7.0") is False

    @patch("subprocess.run")
    def test_get_python_version(self, mock_run):
        """Test getting Python version."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Python 3.11.2"
        )

        validator = OutlookHelperValidator()
        version = validator.get_python_version("C:/Python311/python.exe")

        assert version == "3.11.2"


class TestValidation:
    """Test full validation workflow."""

    @patch.object(OutlookHelperValidator, "is_wsl", return_value=False)
    def test_validation_fails_not_wsl(self, mock_is_wsl):
        """Test validation fails if not in WSL."""
        validator = OutlookHelperValidator()
        result = validator.validate_all()

        assert result.passed is False
        assert any("Not running in WSL" in error for error in result.errors)

    @patch.object(OutlookHelperValidator, "is_wsl", return_value=True)
    @patch.object(OutlookHelperValidator, "can_access_windows_filesystem", return_value=False)
    def test_validation_fails_no_filesystem_access(self, mock_fs, mock_wsl):
        """Test validation fails without Windows filesystem access."""
        validator = OutlookHelperValidator()
        result = validator.validate_all()

        assert result.passed is False
        assert any("Cannot access Windows filesystem" in error for error in result.errors)
```

### 7.2 Integration Tests

**File:** `tests/connectors/test_outlook_wsl_client.py`

```python
"""Integration tests for WSL client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from scripts.connectors.outlook_wsl_client import OutlookWSLClient
from scripts.connectors.outlook_connector import OutlookConfig


class TestOutlookWSLClient:
    """Test WSL client integration."""

    @pytest.fixture
    def config(self):
        """Create test Outlook config."""
        return OutlookConfig(
            account_name="test@example.com",
            folder_path="Inbox",
            days_back=7,
            max_emails=10
        )

    @patch("scripts.connectors.outlook_wsl_client.is_outlook_helper_ready")
    def test_extract_fails_if_helper_not_ready(self, mock_ready, config):
        """Test extraction fails if helper not ready."""
        mock_ready.return_value = False

        client = OutlookWSLClient(config)

        with pytest.raises(RuntimeError, match="validation failed"):
            client.extract_emails()

    @patch("scripts.connectors.outlook_wsl_client.OutlookHelperValidator")
    @patch("subprocess.run")
    def test_extract_success(self, mock_run, mock_validator, config):
        """Test successful email extraction."""
        # Mock validation
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_all.return_value = Mock(
            passed=True,
            errors=[],
            warnings=[],
            info={}
        )
        mock_validator_instance.config = {
            "windows": {
                "python_path": "C:/Python311/python.exe",
                "helper_script": "C:/helper.py"
            },
            "execution": {
                "timeout": 60,
                "max_retries": 1,
                "retry_backoff": 2
            }
        }
        mock_validator.return_value = mock_validator_instance

        # Mock subprocess
        mock_run.return_value = Mock(
            returncode=0,
            stdout='[{"content": "Email body", "metadata": {"subject": "Test"}}]',
            stderr=''
        )

        client = OutlookWSLClient(config)
        emails = client.extract_emails()

        assert len(emails) == 1
        assert emails[0][0] == "Email body"
        assert emails[0][1]["subject"] == "Test"

    @patch("scripts.connectors.outlook_wsl_client.OutlookHelperValidator")
    @patch("subprocess.run")
    def test_extract_retry_on_failure(self, mock_run, mock_validator, config):
        """Test retry logic on transient failures."""
        # Mock validation
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_all.return_value = Mock(
            passed=True,
            errors=[],
            warnings=[],
            info={}
        )
        mock_validator_instance.config = {
            "windows": {
                "python_path": "C:/Python311/python.exe",
                "helper_script": "C:/helper.py"
            },
            "execution": {
                "timeout": 60,
                "max_retries": 3,
                "retry_backoff": 1
            }
        }
        mock_validator.return_value = mock_validator_instance

        # Mock subprocess - fail twice, then succeed
        mock_run.side_effect = [
            Mock(returncode=1, stdout='', stderr='Error'),
            Mock(returncode=1, stdout='', stderr='Error'),
            Mock(returncode=0, stdout='[]', stderr='')
        ]

        client = OutlookWSLClient(config)
        emails = client.extract_emails()

        assert len(emails) == 0
        assert mock_run.call_count == 3
```

### 7.3 Manual Test Checklist

**Before Release:**

- [ ] **Environment Tests**
  - [ ] WSL2 detection works correctly
  - [ ] Windows filesystem (/mnt/c/) access verified
  - [ ] Native Windows mode falls back to direct connector

- [ ] **Python Detection**
  - [ ] Auto-detection finds Python in standard locations
  - [ ] Manual entry accepts valid paths
  - [ ] Invalid paths show clear error messages
  - [ ] Python version compatibility check works

- [ ] **Helper Script**
  - [ ] Template deployment creates valid script
  - [ ] Helper script version matches config
  - [ ] Self-test runs successfully
  - [ ] Extraction returns valid JSON

- [ ] **Dependency Checks**
  - [ ] pywin32 detection works
  - [ ] Missing packages show installation instructions
  - [ ] Package check handles edge cases (import errors, etc.)

- [ ] **UI Integration**
  - [ ] Wizard guides through all steps
  - [ ] Validation gates work (UI disabled until ready)
  - [ ] Status display shows accurate information
  - [ ] Re-validation updates state correctly

- [ ] **Email Extraction**
  - [ ] Small extraction (1-5 emails) works
  - [ ] Large extraction (100+ emails) completes
  - [ ] Error handling shows clear messages
  - [ ] Retry logic recovers from transient failures

- [ ] **CLI Tool**
  - [ ] Basic validation runs without errors
  - [ ] Auto-fix attempts common fixes
  - [ ] JSON output is valid and parseable
  - [ ] Exit codes correct (0 = success, 1 = failure)

---

## Phase 8: Documentation

### 8.1 User Setup Guide

**File:** `docs/WSL_HELPER_SETUP_GUIDE.md`

```markdown
# Outlook Helper Setup Guide (WSL → Windows Bridge)

## Overview

If you're running this RAG platform in WSL and want to extract emails from
your local Outlook client, you need to set up the Windows Helper Bridge.

This guide walks through the setup process.

## Why This Is Needed

- **Outlook**: Windows-only COM API (cannot be accessed from WSL)
- **Pipeline**: Runs best in WSL/Linux (FAISS + OpenMP compatibility)
- **Solution**: Helper script on Windows, called from WSL via subprocess

## Prerequisites

1. **WSL2** installed and configured
2. **Windows Outlook** installed and configured with your account
3. **Python 3.11+** installed on Windows
4. Access to Windows filesystem from WSL (`/mnt/c/` working)

## Setup Methods

### Method 1: Streamlit Setup Wizard (Recommended)

**Easiest and most user-friendly.**

1. **Launch Streamlit UI**
   ```bash
   cd /path/to/Multi-Source_RAG_Platform
   streamlit run scripts/ui/ui_v3.py
   ```

2. **Navigate to Outlook Integration Tab**

3. **Follow Setup Wizard**
   - The wizard will auto-detect if helper is not configured
   - It will guide you through each requirement step-by-step
   - Validation happens at each step

4. **Complete Setup**
   - When wizard shows "✅ Setup Complete", you're done!

### Method 2: CLI Validation Tool

**For advanced users or automation.**

1. **Run Validator**
   ```bash
   python scripts/tools/outlook_helper_check.py
   ```

2. **Follow Error Messages**
   - The tool will tell you exactly what's missing
   - Run with `--auto-fix` to attempt automatic fixes

3. **Verify Success**
   ```bash
   python scripts/tools/outlook_helper_check.py
   # Should output: ✅ VALIDATION PASSED
   ```

### Method 3: Manual Setup

**For complete control.**

#### Step 1: Install pywin32 on Windows

Open **Windows PowerShell**:

```powershell
# Find your Python installation
where python

# Install pywin32
python -m pip install pywin32
```

#### Step 2: Deploy Helper Script

1. **Create directory** (PowerShell):
   ```powershell
   mkdir C:\MultiSourceRAG\tools
   ```

2. **Copy helper script** (WSL):
   ```bash
   cp scripts/tools/templates/win_com_server.py.template \
      /mnt/c/MultiSourceRAG/tools/win_com_server.py
   ```

#### Step 3: Configure Helper

Edit `configs/outlook_helper.yaml`:

```yaml
version: "1.0"

windows:
  # Set your Windows Python path
  python_path: "C:/Users/YourName/AppData/Local/Programs/Python/Python311/python.exe"

  # Helper script location
  helper_script: "C:/MultiSourceRAG/tools/win_com_server.py"

  helper_version: "1.0"
```

#### Step 4: Validate

```bash
python scripts/tools/outlook_helper_check.py
```

## Troubleshooting

### Issue: "Not running in WSL"

**Cause**: Trying to use helper on native Windows.

**Solution**: If you're on Windows, you don't need the helper! Use the native
Outlook connector instead (it's automatically selected).

### Issue: "Cannot access Windows filesystem"

**Cause**: `/mnt/c/` not accessible from WSL.

**Solution**:
```bash
sudo mkdir -p /mnt/c
sudo mount -t drvfs C: /mnt/c
```

### Issue: "Could not auto-detect Windows Python"

**Cause**: Python not in standard location.

**Solution**: Find your Python manually and set in config:
```powershell
# Windows PowerShell
where python
# Copy the path and update configs/outlook_helper.yaml
```

### Issue: "Missing required packages: pywin32"

**Cause**: pywin32 not installed in Windows Python.

**Solution**:
```powershell
# Windows PowerShell
python -m pip install pywin32
```

### Issue: "Helper self-test failed"

**Cause**: Outlook not accessible or not configured.

**Solution**:
- Ensure Outlook is installed
- Open Outlook and set up your account
- Close Outlook and try again

### Issue: "Helper execution timeout"

**Cause**: Outlook taking too long to respond (large folder, slow disk).

**Solution**: Increase timeout in `configs/outlook_helper.yaml`:
```yaml
execution:
  timeout: 120  # Increase from 60 to 120 seconds
```

## Testing Your Setup

Once setup is complete, test with a small extraction:

1. **Go to Streamlit UI** → Outlook Integration

2. **Create Test Project**
   - Name: "Test_Emails"
   - Select your account
   - Choose: Inbox
   - Limit: 5 emails

3. **Preview Emails**
   - Should show 5 email summaries

4. **Extract**
   - Should complete without errors
   - Check: `data/projects/Test_Emails/input/raw/outlook_eml/emails.outlook_eml`

5. **Run Pipeline**
   - Steps: ingest, chunk, embed
   - Should complete successfully

6. **Test Retrieve/Ask**
   - Query: "Show me recent emails"
   - Should return results

## Best Practices

1. **Keep Helper Updated**
   - When updating the RAG platform, re-run setup wizard to check for updates

2. **Monitor Helper Logs**
   - Check Streamlit terminal for helper execution logs
   - Look for JSON log lines from helper

3. **Use Reasonable Limits**
   - Start with small extractions (10-50 emails)
   - Increase gradually to avoid timeouts

4. **Regular Validation**
   - Run `outlook_helper_check.py` periodically
   - Catches issues before extraction failures

## Advanced: CI/CD Integration

For automated testing:

```bash
# In CI pipeline
python scripts/tools/outlook_helper_check.py --json

# Check exit code
if [ $? -eq 0 ]; then
  echo "Helper validation passed"
else
  echo "Helper validation failed"
  exit 1
fi
```

## Support

If you encounter issues not covered here:

1. Run: `python scripts/tools/outlook_helper_check.py -v`
2. Check error messages carefully
3. Review helper logs in Streamlit terminal
4. Check Windows Event Viewer for COM errors

## Next Steps

Once setup is complete:
- Create Outlook-based projects
- Extract emails from multiple folders
- Run full RAG pipeline (ingest → ask)
- Try email-specific queries

---

**Setup complete?** Go to [Email Prompting Guide](./EMAIL_PROMPTING_GUIDE.md) to learn how to ask better questions!
```

### 8.2 Update Main README

Add section to `README.md`:

```markdown
## Outlook Integration (WSL Users)

If you're running in WSL and want to extract emails from Windows Outlook:

1. **Run Setup Wizard**
   ```bash
   streamlit run scripts/ui/ui_v3.py
   # Navigate to: Outlook Integration → Follow Wizard
   ```

2. **Or use CLI**
   ```bash
   python scripts/tools/outlook_helper_check.py --auto-fix
   ```

See [WSL Helper Setup Guide](docs/WSL_HELPER_SETUP_GUIDE.md) for details.
```

---

## Implementation Timeline & Effort

### Phase-by-Phase Breakdown

| Phase | Component | Effort | Dependencies | Priority |
|-------|-----------|--------|--------------|----------|
| **1** | Configuration & Validation | 4-6 hours | None | Critical |
| **2** | Helper Script Template | 2-3 hours | Phase 1 | Critical |
| **3** | WSL Client Wrapper | 3-4 hours | Phase 1, 2 | Critical |
| **4** | Setup Wizard UI | 4-5 hours | Phase 1, 2 | High |
| **5** | UI Integration & Gating | 2-3 hours | Phase 1, 4 | High |
| **6** | CLI Validation Tool | 2-3 hours | Phase 1 | Medium |
| **7** | Testing | 3-4 hours | All phases | High |
| **8** | Documentation | 2-3 hours | All phases | Medium |

**Total Estimated Effort:** 22-31 hours

### Recommended Implementation Order

**Week 1: Core Functionality (Phases 1-3)**

*Day 1-2: Foundation*
- ✅ Phase 1: Configuration schema + validation utilities
- ✅ Test path translation, environment detection
- **Milestone**: Validator class working, can detect Python, check WSL

*Day 3-4: Helper Script*
- ✅ Phase 2: Windows helper script template
- ✅ Test self-test mode, extraction logic
- **Milestone**: Helper script can extract emails when called directly

*Day 5: WSL Client*
- ✅ Phase 3: WSL client wrapper
- ✅ Test subprocess communication, JSON parsing
- **Milestone**: End-to-end extraction works (WSL → Windows → emails)

**Week 2: User Experience (Phases 4-6)**

*Day 1-2: Setup Wizard*
- ✅ Phase 4: Streamlit setup wizard
- ✅ Test each wizard step
- **Milestone**: User can set up helper through UI

*Day 3: UI Integration*
- ✅ Phase 5: Modify Outlook manager with gates
- ✅ Test gating logic, validation status display
- **Milestone**: UI disables Outlook until helper ready

*Day 4: CLI Tool*
- ✅ Phase 6: CLI validation tool
- ✅ Test auto-fix, JSON output
- **Milestone**: CLI tool validates and auto-fixes issues

**Week 3: Testing & Documentation (Phases 7-8)**

*Day 1-2: Testing*
- ✅ Phase 7: Unit tests, integration tests
- ✅ Run manual test checklist
- **Milestone**: All tests passing, no known bugs

*Day 3: Documentation*
- ✅ Phase 8: User guide, README updates
- ✅ Review all docs for clarity
- **Milestone**: Complete documentation published

*Day 4: Buffer*
- Fix any remaining issues
- Polish UI/UX
- Performance testing

### Minimum Viable Implementation (MVP)

**If time is limited, implement in this order:**

1. **Phase 1** (Critical): Validation utilities
2. **Phase 2** (Critical): Helper script
3. **Phase 3** (Critical): WSL client
4. **Phase 6** (High): CLI tool (for testing)
5. **Phase 4** (Can defer): Setup wizard (users can configure manually)
6. **Phase 5** (Can defer): UI gating (just document requirements)
7. **Phase 7** (Can defer): Full test suite
8. **Phase 8** (Can defer): Comprehensive docs

**MVP Timeline:** 1-1.5 weeks (12-16 hours)

---

## Success Criteria

### Technical Success

- [x] Helper configuration validated automatically
- [x] Windows Python auto-detected in common locations
- [x] Helper script deploys without errors
- [x] Email extraction works end-to-end (WSL → Windows → emails)
- [x] Retry logic handles transient failures
- [x] Error messages are clear and actionable

### User Experience Success

- [x] User completes setup in <10 minutes (with wizard)
- [x] User understands what's needed at each step
- [x] Errors show exact commands to fix issues
- [x] Setup persists across sessions (no re-configuration)
- [x] Validation runs automatically before extraction

### Code Quality Success

- [x] All validation utilities have unit tests
- [x] Integration tests cover main workflows
- [x] Code follows project patterns (logging, error handling)
- [x] Documentation is complete and accurate
- [x] No hardcoded paths (all configurable)

---

## Future Enhancements (Post-Implementation)

### Short-Term (Next Month)

1. **Docker Production Deployment**
   - Dockerize pipeline (as planned in Docker analysis)
   - Helper remains on Windows host
   - Volume mount for data sharing

2. **Email-Specific Prompting**
   - Implement email template (as designed in proposal)
   - Auto-detect content type
   - Metadata-rich context building

3. **Performance Optimization**
   - Parallel email extraction (batch mode)
   - Caching of Outlook connection
   - Incremental updates (only new emails)

### Mid-Term (Next Quarter)

4. **Multiple Account Support**
   - Extract from multiple Outlook accounts
   - Merge emails from different sources
   - Account-specific configuration

5. **Advanced Filtering**
   - Filter by sender, subject patterns
   - Exclude automated emails, newsletters
   - Custom date ranges, time windows

6. **Attachment Handling**
   - Extract text from email attachments
   - Index PDF, Word, Excel attachments
   - Link attachments to parent emails

### Long-Term (Next Year)

7. **Alternative Extraction Methods**
   - Microsoft Graph API integration (no COM needed)
   - IMAP/POP3 support (other mail clients)
   - PST file import (archived emails)

8. **Real-Time Updates**
   - Watch Outlook folders for new emails
   - Auto-extract and ingest
   - Background sync daemon

---

## Conclusion

This implementation plan provides a complete, user-friendly solution for
extracting Outlook emails from WSL. The phased approach ensures:

- **Core functionality** implemented first (Phases 1-3)
- **User experience** polished next (Phases 4-5)
- **Testing & documentation** ensures quality (Phases 7-8)

**Estimated Effort:** 22-31 hours over 2-3 weeks
**MVP:** 12-16 hours over 1-1.5 weeks

**Ready to implement?** Start with Phase 1 (Configuration & Validation).

---

**Document Version:** 1.0
**Date:** 2025-01-19
**Status:** Ready for Implementation
