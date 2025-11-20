#!/usr/bin/env python3
"""
Test script for Phase 4: Setup Wizard UI

Tests:
- Wizard module structure
- Function definitions
- Step rendering functions
- Helper functions
"""

import sys
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that wizard module imports correctly."""
    print("=" * 60)
    print("TEST: Imports")
    print("=" * 60)

    try:
        from scripts.ui.ui_outlook_setup_wizard import (
            render_outlook_setup_wizard
        )
        print("✓ render_outlook_setup_wizard imported")

        import streamlit
        print("✓ streamlit available")

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    print()
    return True


def test_wizard_structure():
    """Test wizard module structure."""
    print("=" * 60)
    print("TEST: Wizard Structure")
    print("=" * 60)

    wizard_path = project_root / "scripts" / "ui" / "ui_outlook_setup_wizard.py"

    if not wizard_path.exists():
        print(f"✗ Wizard file not found: {wizard_path}")
        return False

    print(f"✓ Wizard file found: {wizard_path}")

    # Read and parse file
    with open(wizard_path, 'r') as f:
        code = f.read()

    # Parse AST
    tree = ast.parse(code)

    # Find all function definitions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    required_functions = [
        "render_outlook_setup_wizard",
        "_render_environment_check",
        "_render_python_config",
        "_render_helper_deployment",
        "_deploy_helper_script",
        "_render_dependency_check",
        "_render_final_validation",
        "_render_completion"
    ]

    print(f"✓ Total lines: {len(code.split(chr(10)))}")
    print()
    print("Required functions:")

    all_found = True
    for func in required_functions:
        if func in functions:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} (MISSING)")
            all_found = False

    print()
    return all_found


def test_step_functions():
    """Test that step rendering functions exist."""
    print("=" * 60)
    print("TEST: Step Functions")
    print("=" * 60)

    steps = [
        ("Step 0", "_render_environment_check", "Environment Detection"),
        ("Step 1", "_render_python_config", "Python Configuration"),
        ("Step 2", "_render_helper_deployment", "Helper Deployment"),
        ("Step 3", "_render_dependency_check", "Dependency Check"),
        ("Step 4", "_render_final_validation", "Final Validation"),
        ("Step 5", "_render_completion", "Completion"),
    ]

    from scripts.ui import ui_outlook_setup_wizard

    all_exist = True
    for step_num, func_name, description in steps:
        if hasattr(ui_outlook_setup_wizard, func_name):
            print(f"✓ {step_num}: {description} ({func_name})")
        else:
            print(f"✗ {step_num}: {description} ({func_name}) - MISSING")
            all_exist = False

    print()
    return all_exist


def test_helper_function():
    """Test helper deployment function exists."""
    print("=" * 60)
    print("TEST: Helper Functions")
    print("=" * 60)

    from scripts.ui import ui_outlook_setup_wizard

    if hasattr(ui_outlook_setup_wizard, "_deploy_helper_script"):
        print("✓ _deploy_helper_script function exists")
    else:
        print("✗ _deploy_helper_script function missing")
        return False

    print()
    return True


def test_integration_with_validator():
    """Test integration with OutlookHelperValidator."""
    print("=" * 60)
    print("TEST: Integration with Validator")
    print("=" * 60)

    wizard_path = project_root / "scripts" / "ui" / "ui_outlook_setup_wizard.py"

    with open(wizard_path, 'r') as f:
        code = f.read()

    # Check for OutlookHelperValidator import
    checks = [
        ("OutlookHelperValidator import", "OutlookHelperValidator" in code),
        ("ValidationResult import", "ValidationResult" in code),
        ("validator.is_wsl()", "validator.is_wsl()" in code),
        ("validator.can_access_windows_filesystem()", "validator.can_access_windows_filesystem()" in code),
        ("validator.auto_detect_windows_python()", "validator.auto_detect_windows_python()" in code),
        ("validator.validate_all()", "validator.validate_all()" in code),
    ]

    all_passed = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    print()
    return all_passed


def test_streamlit_elements():
    """Test that Streamlit elements are used."""
    print("=" * 60)
    print("TEST: Streamlit Elements")
    print("=" * 60)

    wizard_path = project_root / "scripts" / "ui" / "ui_outlook_setup_wizard.py"

    with open(wizard_path, 'r') as f:
        code = f.read()

    # Check for Streamlit elements
    elements = [
        ("st.markdown", "st.markdown" in code),
        ("st.success", "st.success" in code),
        ("st.error", "st.error" in code),
        ("st.warning", "st.warning" in code),
        ("st.info", "st.info" in code),
        ("st.button", "st.button" in code),
        ("st.progress", "st.progress" in code),
        ("st.spinner", "st.spinner" in code),
        ("st.text_input", "st.text_input" in code),
        ("st.code", "st.code" in code),
    ]

    all_found = True
    for element, found in elements:
        status = "✓" if found else "✗"
        print(f"  {status} {element}")
        if not found:
            all_found = False

    print()
    return all_found


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 4 TESTING: Setup Wizard UI")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("Imports", test_imports()))
        results.append(("Wizard structure", test_wizard_structure()))
        results.append(("Step functions", test_step_functions()))
        results.append(("Helper functions", test_helper_function()))
        results.append(("Integration with validator", test_integration_with_validator()))
        results.append(("Streamlit elements", test_streamlit_elements()))

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
            print("Note: Full wizard functionality requires Streamlit UI to be running.")
            print("To test the wizard:")
            print("  1. Run: streamlit run scripts/ui/ui_v3.py")
            print("  2. Navigate to Outlook Integration tab")
            print("  3. Follow the wizard steps")
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
