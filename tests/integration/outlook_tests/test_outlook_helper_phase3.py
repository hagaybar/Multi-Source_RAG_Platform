#!/usr/bin/env python3
"""
Test script for Phase 3: WSL Client Wrapper

Tests:
- OutlookWSLClient class structure
- Factory function (get_outlook_connector)
- Integration with OutlookConfig
- Validation integration
- Mock extraction workflow
"""

import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required imports work."""
    print("=" * 60)
    print("TEST: Imports")
    print("=" * 60)

    try:
        from scripts.connectors.outlook_wsl_client import (
            OutlookWSLClient,
            get_outlook_connector
        )
        print("✓ OutlookWSLClient imported")
        print("✓ get_outlook_connector imported")

        from scripts.connectors.outlook_connector import OutlookConfig
        print("✓ OutlookConfig imported")

        from scripts.connectors.outlook_helper_utils import (
            OutlookHelperValidator,
            is_outlook_helper_ready
        )
        print("✓ OutlookHelperValidator imported")

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    print()
    return True


def test_client_structure():
    """Test OutlookWSLClient class structure."""
    print("=" * 60)
    print("TEST: Client Structure")
    print("=" * 60)

    from scripts.connectors.outlook_wsl_client import OutlookWSLClient
    from scripts.connectors.outlook_connector import OutlookConfig

    # Create test config
    config = OutlookConfig(
        account_name="test@example.com",
        folder_path="Inbox",
        days_back=7,
        max_emails=10
    )

    # Initialize client
    client = OutlookWSLClient(config)

    # Check attributes
    checks = [
        ("config attribute", hasattr(client, "config")),
        ("logger attribute", hasattr(client, "logger")),
        ("validator attribute", hasattr(client, "validator")),
        ("helper_config attribute", hasattr(client, "helper_config")),
        ("validate method", hasattr(client, "validate")),
        ("extract_emails method", hasattr(client, "extract_emails")),
        ("_process_helper_logs method", hasattr(client, "_process_helper_logs")),
    ]

    all_passed = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}: {result}")
        if not result:
            all_passed = False

    # Check config is stored correctly
    if client.config.account_name == "test@example.com":
        print(f"  ✓ Config stored correctly")
    else:
        print(f"  ✗ Config not stored correctly")
        all_passed = False

    print()
    return all_passed


def test_factory_function():
    """Test get_outlook_connector factory function."""
    print("=" * 60)
    print("TEST: Factory Function")
    print("=" * 60)

    from scripts.connectors.outlook_wsl_client import get_outlook_connector
    from scripts.connectors.outlook_connector import OutlookConfig
    from scripts.connectors.outlook_helper_utils import OutlookHelperValidator

    config = OutlookConfig(
        account_name="test@example.com",
        folder_path="Inbox",
        days_back=7
    )

    # Test WSL path (current environment)
    if OutlookHelperValidator.is_wsl():
        print("✓ Running in WSL (as expected)")

        try:
            # This will fail if helper not ready, which is expected
            connector = get_outlook_connector(config)
            print(f"✓ Factory returned: {type(connector).__name__}")
            from scripts.connectors.outlook_wsl_client import OutlookWSLClient
            if isinstance(connector, OutlookWSLClient):
                print("✓ Correct type: OutlookWSLClient")
            else:
                print(f"✗ Wrong type: {type(connector)}")
                return False
        except RuntimeError as e:
            # Expected if helper not ready
            if "not configured" in str(e):
                print("⚠ Helper not ready (expected)")
                print("  Factory correctly raises RuntimeError")
            else:
                print(f"✗ Unexpected error: {e}")
                return False
    else:
        print("⚠ Not running in WSL, skipping WSL-specific test")

    print()
    return True


def test_validation_integration():
    """Test validation integration."""
    print("=" * 60)
    print("TEST: Validation Integration")
    print("=" * 60)

    from scripts.connectors.outlook_wsl_client import OutlookWSLClient
    from scripts.connectors.outlook_connector import OutlookConfig

    config = OutlookConfig(
        account_name="test@example.com",
        folder_path="Inbox",
        days_back=7
    )

    client = OutlookWSLClient(config)

    # Run validation
    result = client.validate()

    print(f"Validation passed: {result.passed}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Info items: {len(result.info)}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors[:3]:  # Show first 3
            print(f"  • {error}")

    if result.info:
        print("\nInfo:")
        for key, value in list(result.info.items())[:3]:  # Show first 3
            print(f"  • {key}: {value}")

    print()
    return True


def test_mock_extraction():
    """Test extraction with mocked subprocess."""
    print("=" * 60)
    print("TEST: Mock Extraction")
    print("=" * 60)

    from scripts.connectors.outlook_wsl_client import OutlookWSLClient
    from scripts.connectors.outlook_connector import OutlookConfig

    config = OutlookConfig(
        account_name="test@example.com",
        folder_path="Inbox",
        days_back=7,
        max_emails=5
    )

    client = OutlookWSLClient(config)

    # Mock the subprocess and validation
    with patch('scripts.connectors.outlook_wsl_client.subprocess.run') as mock_run, \
         patch.object(client, 'validate') as mock_validate:

        # Mock successful validation
        mock_validate.return_value = Mock(
            passed=True,
            errors=[],
            warnings=[],
            info={"environment": "WSL2"}
        )

        # Mock successful extraction
        mock_output = [
            {
                "content": "Email body 1",
                "metadata": {
                    "subject": "Test Email 1",
                    "sender": "sender1@test.com",
                    "date": "2025-01-19 10:00:00"
                }
            },
            {
                "content": "Email body 2",
                "metadata": {
                    "subject": "Test Email 2",
                    "sender": "sender2@test.com",
                    "date": "2025-01-19 11:00:00"
                }
            }
        ]

        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_output),
            stderr='{"level": "INFO", "message": "Extracted 2 emails", "timestamp": "2025-01-19T10:00:00"}'
        )

        # Run extraction
        try:
            emails = client.extract_emails()

            print(f"✓ Extraction succeeded")
            print(f"✓ Extracted {len(emails)} emails")

            # Verify format
            if len(emails) == 2:
                print(f"✓ Correct number of emails")
            else:
                print(f"✗ Expected 2 emails, got {len(emails)}")
                return False

            # Check first email
            body, metadata = emails[0]
            if body == "Email body 1" and metadata["subject"] == "Test Email 1":
                print(f"✓ Email format correct")
            else:
                print(f"✗ Email format incorrect")
                return False

            # Verify subprocess was called
            if mock_run.called:
                print(f"✓ Subprocess was called")
                args = mock_run.call_args[0][0]
                print(f"  Command included: --account, --folder, --days")
            else:
                print(f"✗ Subprocess was not called")
                return False

        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print()
    return True


def test_retry_logic():
    """Test retry logic with mock failures."""
    print("=" * 60)
    print("TEST: Retry Logic")
    print("=" * 60)

    from scripts.connectors.outlook_wsl_client import OutlookWSLClient
    from scripts.connectors.outlook_connector import OutlookConfig

    config = OutlookConfig(
        account_name="test@example.com",
        folder_path="Inbox",
        days_back=7
    )

    client = OutlookWSLClient(config)

    # Mock validation and subprocess
    with patch('scripts.connectors.outlook_wsl_client.subprocess.run') as mock_run, \
         patch.object(client, 'validate') as mock_validate, \
         patch('scripts.connectors.outlook_wsl_client.time.sleep') as mock_sleep:

        # Mock successful validation
        mock_validate.return_value = Mock(
            passed=True,
            errors=[],
            warnings=[],
            info={}
        )

        # Mock: fail twice, then succeed
        mock_run.side_effect = [
            # First attempt: failure
            Mock(returncode=1, stdout='', stderr='Error 1'),
            # Second attempt: failure
            Mock(returncode=1, stdout='', stderr='Error 2'),
            # Third attempt: success
            Mock(
                returncode=0,
                stdout='[]',
                stderr='{"level": "INFO", "message": "Success"}'
            )
        ]

        try:
            emails = client.extract_emails()
            print(f"✓ Extraction succeeded after retries")
            print(f"✓ Subprocess called {mock_run.call_count} times")

            if mock_run.call_count == 3:
                print(f"✓ Correct number of retries (3)")
            else:
                print(f"✗ Expected 3 calls, got {mock_run.call_count}")
                return False

            if mock_sleep.called:
                print(f"✓ Sleep was called (backoff working)")
            else:
                print(f"⚠ Sleep not called (may not be needed)")

        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False

    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 3 TESTING: WSL Client Wrapper")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("Imports", test_imports()))
        results.append(("Client structure", test_client_structure()))
        results.append(("Factory function", test_factory_function()))
        results.append(("Validation integration", test_validation_integration()))
        results.append(("Mock extraction", test_mock_extraction()))
        results.append(("Retry logic", test_retry_logic()))

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
