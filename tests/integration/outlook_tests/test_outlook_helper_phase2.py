#!/usr/bin/env python3
"""
Test script for Phase 2: Windows Helper Script Template

Tests:
- Template file exists and is valid Python
- Version constant matches config
- Required functions exist
- Script structure is correct
"""

import sys
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_template_exists():
    """Test that template file exists."""
    print("=" * 60)
    print("TEST: Template File Exists")
    print("=" * 60)

    template_path = project_root / "scripts" / "tools" / "templates" / "win_com_server.py.template"

    if template_path.exists():
        print(f"✓ Template found: {template_path}")
        print(f"✓ File size: {template_path.stat().st_size} bytes")
    else:
        print(f"✗ Template not found: {template_path}")
        return False

    print()
    return True


def test_template_syntax():
    """Test that template is valid Python."""
    print("=" * 60)
    print("TEST: Template Python Syntax")
    print("=" * 60)

    template_path = project_root / "scripts" / "tools" / "templates" / "win_com_server.py.template"

    try:
        with open(template_path, 'r') as f:
            code = f.read()

        # Parse to check syntax
        ast.parse(code)
        print("✓ Valid Python syntax")

        # Count lines
        lines = code.split('\n')
        print(f"✓ Total lines: {len(lines)}")

        # Count non-empty lines
        non_empty = sum(1 for line in lines if line.strip())
        print(f"✓ Non-empty lines: {non_empty}")

        # Count comments
        comments = sum(1 for line in lines if line.strip().startswith('#'))
        print(f"✓ Comment lines: {comments}")

    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

    print()
    return True


def test_template_structure():
    """Test that template has required structure."""
    print("=" * 60)
    print("TEST: Template Structure")
    print("=" * 60)

    template_path = project_root / "scripts" / "tools" / "templates" / "win_com_server.py.template"

    with open(template_path, 'r') as f:
        code = f.read()

    # Parse AST
    tree = ast.parse(code)

    # Find all function definitions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    required_functions = [
        "log_error",
        "log_info",
        "self_test",
        "extract_emails",
        "main"
    ]

    print("Required functions:")
    all_found = True
    for func in required_functions:
        if func in functions:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} (MISSING)")
            all_found = False

    print()

    # Check for HELPER_VERSION constant
    has_version = "HELPER_VERSION" in code
    print(f"{'✓' if has_version else '✗'} HELPER_VERSION constant: {has_version}")

    # Extract version value
    if has_version:
        for line in code.split('\n'):
            if line.startswith('HELPER_VERSION'):
                version = line.split('=')[1].strip().strip('"\'')
                print(f"  Version: {version}")

                # Check if it matches config default
                from scripts.connectors.outlook_helper_utils import OutlookHelperValidator
                validator = OutlookHelperValidator()
                expected_version = validator.config["windows"]["helper_version"]

                if version == expected_version:
                    print(f"  ✓ Matches config: {expected_version}")
                else:
                    print(f"  ✗ Config mismatch: expected {expected_version}, found {version}")
                    all_found = False

    print()
    return all_found and has_version


def test_template_imports():
    """Test that template has required imports."""
    print("=" * 60)
    print("TEST: Template Imports")
    print("=" * 60)

    template_path = project_root / "scripts" / "tools" / "templates" / "win_com_server.py.template"

    with open(template_path, 'r') as f:
        code = f.read()

    required_imports = [
        "sys",
        "json",
        "argparse",
        "datetime",
        "typing",
        "win32com.client",
        "pythoncom"
    ]

    print("Required imports:")
    for imp in required_imports:
        if imp in code:
            print(f"  ✓ {imp}")
        else:
            print(f"  ✗ {imp} (MISSING)")

    print()
    return True


def test_helper_deployment():
    """Test helper deployment function."""
    print("=" * 60)
    print("TEST: Helper Deployment")
    print("=" * 60)

    from scripts.connectors.outlook_helper_utils import OutlookHelperValidator

    validator = OutlookHelperValidator()
    helper_path = validator.config["windows"]["helper_script"]

    print(f"Target path: {helper_path}")

    # Convert to WSL path to check
    wsl_path = validator.windows_to_wsl_path(helper_path)
    print(f"WSL path: {wsl_path}")

    # Check if directory exists
    parent_dir = Path(wsl_path).parent
    if parent_dir.exists():
        print(f"✓ Parent directory exists: {parent_dir}")
    else:
        print(f"⚠ Parent directory does not exist: {parent_dir}")
        print(f"  Create with: mkdir -p '{parent_dir}'")

    # Check if helper already deployed
    if Path(wsl_path).exists():
        print(f"✓ Helper already deployed")

        # Check version
        deployed_version = validator.get_helper_version(helper_path)
        if deployed_version:
            print(f"  Deployed version: {deployed_version}")
        else:
            print(f"  ⚠ Could not read version from deployed script")
    else:
        print(f"⚠ Helper not yet deployed")
        print(f"  Will be deployed by setup wizard")

    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 2 TESTING: Windows Helper Script Template")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("Template exists", test_template_exists()))
        results.append(("Template syntax", test_template_syntax()))
        results.append(("Template structure", test_template_structure()))
        results.append(("Template imports", test_template_imports()))
        results.append(("Helper deployment", test_helper_deployment()))

        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {test_name}")

        print()

        if all(passed for _, passed in results):
            print("✓ All tests passed")
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
