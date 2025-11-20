#!/usr/bin/env python3
"""
Deploy Outlook Helper Script

Deploys the helper script template to the Windows filesystem.
Can be used standalone or imported by the setup wizard.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.connectors.outlook_helper_utils import OutlookHelperValidator


def deploy_helper_script(
    target_path: str = None,
    validator: OutlookHelperValidator = None
) -> bool:
    """
    Deploy helper script to Windows location.

    Args:
        target_path: Windows path to deploy to (optional, uses config if None)
        validator: OutlookHelperValidator instance (optional, creates if None)

    Returns:
        True if successful, False otherwise
    """
    if validator is None:
        validator = OutlookHelperValidator()

    if target_path is None:
        target_path = validator.config["windows"]["helper_script"]

    # Get template path
    template_path = project_root / "scripts" / "tools" / "templates" / "win_com_server.py.template"

    if not template_path.exists():
        print(f"✗ Template not found: {template_path}")
        return False

    # Convert Windows path to WSL path
    wsl_target_path = validator.windows_to_wsl_path(target_path)
    wsl_target = Path(wsl_target_path)

    # Create parent directory if needed
    try:
        wsl_target.parent.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {wsl_target.parent}")
    except Exception as e:
        print(f"✗ Failed to create directory: {e}")
        return False

    # Copy template
    try:
        shutil.copy(template_path, wsl_target)
        print(f"✓ Deployed helper script to: {target_path}")
        print(f"  (WSL path: {wsl_target_path})")
        return True
    except Exception as e:
        print(f"✗ Failed to deploy: {e}")
        return False


def main():
    """Command-line interface for deployment."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Outlook helper script")
    parser.add_argument(
        "--target",
        help="Target Windows path (default: from config)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing script"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Outlook Helper Script Deployment")
    print("=" * 60)
    print()

    # Initialize validator
    validator = OutlookHelperValidator()

    target_path = args.target or validator.config["windows"]["helper_script"]
    wsl_path = validator.windows_to_wsl_path(target_path)

    # Check if already exists
    if Path(wsl_path).exists():
        if not args.force:
            print(f"⚠ Helper script already exists at: {target_path}")
            print(f"  Use --force to overwrite")
            print()

            # Show current version
            version = validator.get_helper_version(target_path)
            if version:
                print(f"  Current version: {version}")

            return 1

        print(f"⚠ Overwriting existing script (--force specified)")
        print()

    # Deploy
    success = deploy_helper_script(target_path, validator)

    print()

    if success:
        print("✓ Deployment successful")

        # Verify deployment
        deployed_version = validator.get_helper_version(target_path)
        if deployed_version:
            print(f"  Deployed version: {deployed_version}")

        print()
        print("Next steps:")
        print("1. Ensure pywin32 is installed on Windows Python")
        print(f"   {validator.config['windows']['python_path']} -m pip install pywin32")
        print()
        print("2. Test the helper:")
        print(f"   {validator.config['windows']['python_path']} {target_path} --self-test")

        return 0
    else:
        print("✗ Deployment failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
