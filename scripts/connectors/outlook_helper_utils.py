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
            },
            "logging": {
                "debug": False,
                "log_stderr": True
            },
            "status": {
                "last_validated": None,
                "validation_passed": False,
                "validation_errors": []
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
            # Convert Windows path to WSL path for execution
            wsl_python_path = self.windows_to_wsl_path(python_path)

            result = subprocess.run(
                [wsl_python_path, "--version"],
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

        # Convert Windows path to WSL path for execution
        wsl_python_path = self.windows_to_wsl_path(python_path)

        # Map package names to actual import names
        # pywin32 is installed as 'pywin32' but imported as 'win32com', 'win32api', etc.
        import_name_map = {
            "pywin32": "win32com.client"
        }

        for package in required:
            # Get the actual import name (may differ from package name)
            import_name = import_name_map.get(package, package)

            try:
                result = subprocess.run(
                    [wsl_python_path, "-c", f"import {import_name}"],
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
        success, _ = self.run_helper_self_test_detailed(python_path, helper_path)
        return success

    def run_helper_self_test_detailed(self, python_path: str, helper_path: str) -> tuple[bool, str]:
        """
        Run helper script self-test with detailed error output.

        Args:
            python_path: Windows path to python.exe
            helper_path: Windows path to helper script

        Returns:
            Tuple of (success: bool, error_output: str)
        """
        try:
            # Convert python path to WSL so WSL can execute it
            # But keep helper_path as Windows path so Windows Python can read it
            wsl_python_path = self.windows_to_wsl_path(python_path)

            result = subprocess.run(
                [wsl_python_path, helper_path, "--self-test"],  # Use Windows path for helper
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, result.stderr  # Success, return any info logs
            else:
                return False, result.stderr  # Failure, return error logs
        except subprocess.TimeoutExpired:
            return False, "Self-test timed out after 10 seconds"
        except Exception as e:
            return False, f"Failed to run self-test: {str(e)}"


# Convenience functions

def validate_outlook_helper() -> ValidationResult:
    """Validate Outlook helper configuration."""
    validator = OutlookHelperValidator()
    return validator.validate_all()


def is_outlook_helper_ready() -> bool:
    """Quick check if Outlook helper is ready to use."""
    result = validate_outlook_helper()
    return result.passed
