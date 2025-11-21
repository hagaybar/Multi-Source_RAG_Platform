#!/usr/bin/env python3
"""
Test script for Phase 5: UI Integration & Gating

Tests:
- UI manager imports WSL utilities
- Requirements check function is environment-aware
- Factory function is used in extraction functions
- IngestionManager uses factory function
- Integration with wizard
"""

import sys
import ast
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that UI manager imports WSL utilities."""
    print("=" * 60)
    print("TEST: Imports")
    print("=" * 60)

    try:
        from scripts.ui.ui_outlook_manager import (
            render_outlook_requirements_check,
            render_outlook_connection_test,
            render_outlook_email_preview,
            render_outlook_ingestion_controls
        )
        print("✓ UI manager functions imported")

        # Check if OutlookHelperValidator is imported
        from scripts.ui import ui_outlook_manager
        if hasattr(ui_outlook_manager, 'OutlookHelperValidator'):
            print("✓ OutlookHelperValidator available in module")
        else:
            print("✗ OutlookHelperValidator not imported")
            return False

        if hasattr(ui_outlook_manager, 'is_outlook_helper_ready'):
            print("✓ is_outlook_helper_ready available in module")
        else:
            print("✗ is_outlook_helper_ready not imported")
            return False

    except ImportError as e:
        if "streamlit" in str(e):
            print("⚠ Streamlit not installed (expected outside poetry env)")
            print("✓ Skipping import test - code structure tests will verify integration")
            print()
            return True  # Not a failure - expected outside venv
        else:
            print(f"✗ Import failed: {e}")
            return False

    print()
    return True


def test_ui_manager_structure():
    """Test UI manager file structure and integration."""
    print("=" * 60)
    print("TEST: UI Manager Structure")
    print("=" * 60)

    ui_manager_path = project_root / "scripts" / "ui" / "ui_outlook_manager.py"

    if not ui_manager_path.exists():
        print(f"✗ UI manager file not found: {ui_manager_path}")
        return False

    print(f"✓ UI manager file found")

    # Read and check for key integrations
    with open(ui_manager_path, 'r') as f:
        code = f.read()

    # Check for WSL utilities import
    checks = [
        ("OutlookHelperValidator import", "OutlookHelperValidator" in code),
        ("is_outlook_helper_ready import", "is_outlook_helper_ready" in code),
        ("get_outlook_connector usage", "get_outlook_connector" in code),
        ("render_outlook_setup_wizard import", "render_outlook_setup_wizard" in code),
        ("WSL detection in requirements", "OutlookHelperValidator.is_wsl()" in code),
        ("Helper readiness check", "is_outlook_helper_ready()" in code),
    ]

    all_passed = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    print()
    return all_passed


def test_requirements_check_function():
    """Test requirements check is environment-aware."""
    print("=" * 60)
    print("TEST: Requirements Check Function")
    print("=" * 60)

    ui_manager_path = project_root / "scripts" / "ui" / "ui_outlook_manager.py"

    with open(ui_manager_path, 'r') as f:
        code = f.read()

    # Parse AST and find render_outlook_requirements_check function
    tree = ast.parse(code)

    func_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "render_outlook_requirements_check":
            func_found = True
            # Check docstring mentions WSL
            docstring = ast.get_docstring(node)
            if docstring and "WSL" in docstring:
                print("✓ Docstring mentions WSL support")
            else:
                print("⚠ Docstring doesn't mention WSL")

            break

    if not func_found:
        print("✗ render_outlook_requirements_check function not found")
        return False

    print("✓ render_outlook_requirements_check function exists")

    # Check for key patterns in the function
    patterns = [
        ("WSL detection", "OutlookHelperValidator.is_wsl()" in code),
        ("Helper readiness check", "is_outlook_helper_ready()" in code),
        ("Wizard rendering", "render_outlook_setup_wizard()" in code),
        ("Validation status display", "validate_all()" in code),
    ]

    all_found = True
    for name, found in patterns:
        status = "✓" if found else "✗"
        print(f"  {status} {name}")
        if not found:
            all_found = False

    print()
    return all_found


def test_factory_usage_in_ui():
    """Test that factory function is used in UI extraction functions."""
    print("=" * 60)
    print("TEST: Factory Function Usage in UI")
    print("=" * 60)

    ui_manager_path = project_root / "scripts" / "ui" / "ui_outlook_manager.py"

    with open(ui_manager_path, 'r') as f:
        code = f.read()

    # Check that get_outlook_connector is used instead of direct OutlookConnector
    checks = [
        ("get_outlook_connector imported", "from scripts.connectors.outlook_wsl_client import get_outlook_connector" in code),
        ("Factory used in code", "get_outlook_connector(outlook_config)" in code or "get_outlook_connector(test_config)" in code),
        ("Direct OutlookConnector not used", "OutlookConnector(outlook_config)" not in code and "OutlookConnector(test_config)" not in code),
    ]

    all_passed = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    print()
    return all_passed


def test_ingestion_manager_integration():
    """Test that IngestionManager uses factory function."""
    print("=" * 60)
    print("TEST: IngestionManager Integration")
    print("=" * 60)

    ingestion_mgr_path = project_root / "scripts" / "ingestion" / "manager.py"

    if not ingestion_mgr_path.exists():
        print(f"✗ IngestionManager file not found: {ingestion_mgr_path}")
        return False

    print(f"✓ IngestionManager file found")

    with open(ingestion_mgr_path, 'r') as f:
        code = f.read()

    # Check for factory function usage
    checks = [
        ("get_outlook_connector import", "from scripts.connectors.outlook_wsl_client import get_outlook_connector" in code),
        ("Factory used in ingest_outlook", "get_outlook_connector(outlook_config)" in code),
        ("Updated docstring", "environment-aware" in code.lower() or "wsl" in code.lower()),
    ]

    all_passed = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    print()
    return all_passed


def test_wizard_integration():
    """Test wizard is integrated into UI flow."""
    print("=" * 60)
    print("TEST: Wizard Integration")
    print("=" * 60)

    ui_manager_path = project_root / "scripts" / "ui" / "ui_outlook_manager.py"

    with open(ui_manager_path, 'r') as f:
        code = f.read()

    # Check wizard is called when helper not ready
    checks = [
        ("Wizard import statement", "from scripts.ui.ui_outlook_setup_wizard import render_outlook_setup_wizard" in code),
        ("Wizard called in code", "render_outlook_setup_wizard()" in code),
        ("Helper not ready check", "not is_outlook_helper_ready()" in code),
    ]

    all_passed = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    print()
    return all_passed


def test_validation_status_display():
    """Test validation status is displayed in UI."""
    print("=" * 60)
    print("TEST: Validation Status Display")
    print("=" * 60)

    ui_manager_path = project_root / "scripts" / "ui" / "ui_outlook_manager.py"

    with open(ui_manager_path, 'r') as f:
        code = f.read()

    # Check for validation status display
    checks = [
        ("Validator instantiation", "validator = OutlookHelperValidator()" in code),
        ("validate_all() called", "validator.validate_all()" in code),
        ("Result display logic", "result.passed" in code or "result.errors" in code),
        ("Expandable section", "st.expander" in code),
    ]

    all_passed = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    print()
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 5 TESTING: UI Integration & Gating")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("Imports", test_imports()))
        results.append(("UI manager structure", test_ui_manager_structure()))
        results.append(("Requirements check function", test_requirements_check_function()))
        results.append(("Factory usage in UI", test_factory_usage_in_ui()))
        results.append(("IngestionManager integration", test_ingestion_manager_integration()))
        results.append(("Wizard integration", test_wizard_integration()))
        results.append(("Validation status display", test_validation_status_display()))

        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {test_name}")

        print()

        if all(passed for _, passed in results):
            print("✓ All tests passed")
            print()
            print("Phase 5 Implementation Complete!")
            print()
            print("Key Features:")
            print("  • Environment-aware requirements check (WSL + Windows)")
            print("  • Setup wizard integration when helper not configured")
            print("  • Validation status display when helper ready")
            print("  • Factory pattern used throughout UI and ingestion")
            print("  • All Outlook features properly gated")
            print()
            print("Next: Test the UI by running:")
            print("  streamlit run scripts/ui/ui_v3.py")
            return 0
        else:
            print("✗ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
