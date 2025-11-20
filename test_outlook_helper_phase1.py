#!/usr/bin/env python3
"""
Test script for Phase 1: Configuration & Validation

Tests:
- Environment detection (is_wsl)
- Path translation (WSL ↔ Windows)
- Configuration loading
- Basic validation flow
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.connectors.outlook_helper_utils import (
    OutlookHelperValidator,
    validate_outlook_helper,
    is_outlook_helper_ready
)


def test_environment_detection():
    """Test WSL and filesystem detection."""
    print("=" * 60)
    print("TEST: Environment Detection")
    print("=" * 60)

    is_wsl = OutlookHelperValidator.is_wsl()
    print(f"✓ is_wsl(): {is_wsl}")

    can_access = OutlookHelperValidator.can_access_windows_filesystem()
    print(f"✓ can_access_windows_filesystem(): {can_access}")

    print()


def test_path_translation():
    """Test path translation utilities."""
    print("=" * 60)
    print("TEST: Path Translation")
    print("=" * 60)

    # Test WSL to Windows
    test_cases_wsl_to_win = [
        ("/mnt/c/Users/hagay/test.py", "C:/Users/hagay/test.py"),
        ("/mnt/d/Projects/test.py", "D:/Projects/test.py"),
        ("/home/user/test.py", "/home/user/test.py"),  # Unchanged
    ]

    print("WSL → Windows:")
    for wsl_path, expected in test_cases_wsl_to_win:
        result = OutlookHelperValidator.wsl_to_windows_path(wsl_path)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {wsl_path} → {result}")
        if result != expected:
            print(f"     Expected: {expected}")

    print()

    # Test Windows to WSL
    test_cases_win_to_wsl = [
        ("C:/Users/hagay/test.py", "/mnt/c/Users/hagay/test.py"),
        ("C:\\Users\\hagay\\test.py", "/mnt/c/Users/hagay/test.py"),
        ("D:/Projects/test.py", "/mnt/d/Projects/test.py"),
    ]

    print("Windows → WSL:")
    for win_path, expected in test_cases_win_to_wsl:
        result = OutlookHelperValidator.windows_to_wsl_path(win_path)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {win_path} → {result}")
        if result != expected:
            print(f"     Expected: {expected}")

    print()


def test_config_loading():
    """Test configuration loading."""
    print("=" * 60)
    print("TEST: Configuration Loading")
    print("=" * 60)

    validator = OutlookHelperValidator()

    print(f"✓ Config path: {validator.config_path}")
    print(f"✓ Config loaded: {bool(validator.config)}")
    print(f"✓ Version: {validator.config.get('version')}")
    print(f"✓ Helper script: {validator.config['windows']['helper_script']}")
    print(f"✓ Auto-detect: {validator.config['validation']['auto_detect']}")

    print()


def test_python_version_check():
    """Test Python version compatibility check."""
    print("=" * 60)
    print("TEST: Python Version Compatibility")
    print("=" * 60)

    test_versions = [
        ("3.11.2", True),
        ("3.12.0", True),
        ("3.10.5", False),
        ("2.7.18", False),
    ]

    for version, expected in test_versions:
        result = OutlookHelperValidator.is_python_version_compatible(version)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {version}: {result} (expected: {expected})")

    print()


def test_full_validation():
    """Test full validation workflow."""
    print("=" * 60)
    print("TEST: Full Validation")
    print("=" * 60)

    print("Running full validation...")
    result = validate_outlook_helper()

    print(f"\nValidation Result: {'✓ PASSED' if result.passed else '✗ FAILED'}")
    print()

    if result.info:
        print("📋 Info:")
        for key, value in result.info.items():
            print(f"  • {key}: {value}")
        print()

    if result.warnings:
        print("⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")
        print()

    if result.errors:
        print("🚫 Errors:")
        for error in result.errors:
            print(f"  • {error}")
        print()

    print(f"Helper ready: {is_outlook_helper_ready()}")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 TESTING: Configuration & Validation")
    print("=" * 60 + "\n")

    try:
        test_environment_detection()
        test_path_translation()
        test_config_loading()
        test_python_version_check()
        test_full_validation()

        print("=" * 60)
        print("✓ All tests completed")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
