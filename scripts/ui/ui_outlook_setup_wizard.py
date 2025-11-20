"""
Outlook Helper Setup Wizard for Streamlit UI

Guides users through setting up the Windows helper for Outlook
extraction from WSL.
"""

import streamlit as st
import shutil
import time
from pathlib import Path
from scripts.connectors.outlook_helper_utils import (
    OutlookHelperValidator,
    ValidationResult
)


def render_outlook_setup_wizard():
    """
    Render Outlook helper setup wizard.

    Guides user through:
    0. Environment detection
    1. Python path configuration
    2. Helper script deployment
    3. Dependency installation
    4. Final validation
    5. Completion
    """
    st.subheader("🔧 Outlook Helper Setup Wizard")

    validator = OutlookHelperValidator()

    # Initialize session state
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0

    # Progress indicator
    total_steps = 6
    st.progress((st.session_state.wizard_step + 1) / total_steps)
    st.caption(f"Step {st.session_state.wizard_step + 1} of {total_steps}")

    # Render current step
    if st.session_state.wizard_step == 0:
        _render_environment_check(validator)
    elif st.session_state.wizard_step == 1:
        _render_python_config(validator)
    elif st.session_state.wizard_step == 2:
        _render_helper_deployment(validator)
    elif st.session_state.wizard_step == 3:
        _render_dependency_check(validator)
    elif st.session_state.wizard_step == 4:
        _render_final_validation(validator)
    elif st.session_state.wizard_step == 5:
        _render_completion()


def _render_environment_check(validator: OutlookHelperValidator):
    """Step 0: Check WSL environment."""
    st.markdown("### Step 1: Environment Detection")
    st.markdown("Checking if your environment is compatible...")

    # Check WSL
    if not validator.is_wsl():
        st.error("❌ Not running in WSL")
        st.info(
            "💡 **This wizard is for WSL users.** If you're on Windows, "
            "you can use the native Outlook connector without the helper."
        )
        st.markdown("---")
        if st.button("← Exit Wizard"):
            st.session_state.wizard_step = 0
            st.rerun()
        return

    st.success("✅ Running in WSL2")

    # Check filesystem access
    if not validator.can_access_windows_filesystem():
        st.error("❌ Cannot access Windows filesystem (/mnt/c/)")
        st.code("# Try remounting:\nsudo mkdir -p /mnt/c\nsudo mount -t drvfs C: /mnt/c")
        st.markdown("---")
        if st.button("🔄 Recheck"):
            st.rerun()
        return

    st.success("✅ Windows filesystem accessible")

    st.markdown("---")
    st.info("✨ Environment is compatible! Ready to proceed.")

    if st.button("Next: Configure Python Path →"):
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
            if validator.is_python_version_compatible(version):
                st.success(f"✅ Version: {version} (compatible)")
            else:
                st.warning(f"⚠️ Version: {version} (may not be compatible, recommended: 3.11+)")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Use This Python"):
                validator.config["windows"]["python_path"] = detected_path
                validator.save_config()
                st.success("✅ Saved configuration")
                time.sleep(1)
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("← Back", key="back_python_auto"):
                st.session_state.wizard_step = 0
                st.rerun()

    # Manual entry
    st.markdown("---")
    st.markdown("**Or enter manually:**")

    current_path = validator.config["windows"]["python_path"]
    python_path = st.text_input(
        "Windows Python Path",
        value=current_path if current_path else "",
        placeholder="C:/Users/YourName/AppData/Local/Programs/Python/Python311/python.exe",
        help="Enter the full path to python.exe on Windows"
    )

    if python_path:
        if validator.validate_windows_python(python_path):
            st.success("✅ Python executable found")
            version = validator.get_python_version(python_path)
            if version:
                if validator.is_python_version_compatible(version):
                    st.success(f"✅ Version: {version} (compatible)")
                else:
                    st.warning(f"⚠️ Version: {version} (may not be compatible)")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Save and Continue →"):
                    validator.config["windows"]["python_path"] = python_path
                    validator.save_config()
                    st.success("✅ Saved configuration")
                    time.sleep(1)
                    st.session_state.wizard_step = 2
                    st.rerun()
            with col2:
                if st.button("← Back", key="back_manual"):
                    st.session_state.wizard_step = 0
                    st.rerun()
        else:
            st.error("❌ Python not found at this path")
            st.info("💡 **Suggestions:**")
            for suggestion in validator.suggest_python_paths():
                st.code(suggestion)

            if st.button("← Back", key="back_invalid"):
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
            st.info(f"📌 Version: {version}")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back", key="back_helper_exists"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("🔄 Re-deploy"):
                if _deploy_helper_script(helper_path, validator):
                    st.success("✅ Re-deployed successfully")
                    time.sleep(1)
                    st.rerun()
        with col3:
            if st.button("Next →"):
                st.session_state.wizard_step = 3
                st.rerun()
    else:
        st.warning("⚠️ Helper script not found")
        st.markdown("The helper script needs to be deployed to the Windows filesystem.")

        if st.button("📥 Deploy Helper Script"):
            with st.spinner("Deploying helper script..."):
                if _deploy_helper_script(helper_path, validator):
                    st.success("✅ Deployed successfully!")
                    time.sleep(1)
                    st.session_state.wizard_step = 3
                    st.rerun()
                else:
                    st.error("❌ Deployment failed")

        st.markdown("---")
        if st.button("← Back", key="back_nodeploy"):
            st.session_state.wizard_step = 1
            st.rerun()


def _deploy_helper_script(helper_path: str, validator: OutlookHelperValidator) -> bool:
    """
    Deploy helper script to Windows location.

    Args:
        helper_path: Windows path to helper script
        validator: OutlookHelperValidator instance

    Returns:
        True if successful
    """
    try:
        # Get template
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "tools" / "templates" / "win_com_server.py.template"

        if not template_path.exists():
            st.error(f"Template not found: {template_path}")
            return False

        # Convert to WSL path
        wsl_path = validator.windows_to_wsl_path(helper_path)

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

    if not python_path:
        st.error("❌ Python path not configured")
        if st.button("← Back to configure"):
            st.session_state.wizard_step = 1
            st.rerun()
        return

    st.info(f"🔍 Checking packages in: `{python_path}`")

    missing = validator.check_required_packages(python_path)

    if not missing:
        st.success("✅ All required packages installed")
        st.markdown("**Required packages:**")
        for pkg in validator.config["validation"]["required_packages"]:
            st.markdown(f"- ✅ {pkg}")

        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", key="back_deps_ok"):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("Next: Final Validation →"):
                st.session_state.wizard_step = 4
                st.rerun()
    else:
        st.error(f"❌ Missing packages: {', '.join(missing)}")

        st.markdown("**Installation Instructions:**")
        st.markdown("Copy and run this command in **Windows PowerShell**:")
        st.code(
            f"{python_path} -m pip install {' '.join(missing)}",
            language="powershell"
        )

        st.info(
            "💡 **Steps:**\n"
            "1. Copy the command above\n"
            "2. Open Windows PowerShell\n"
            "3. Paste and run the command\n"
            "4. Come back and click 'Check Again'"
        )

        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", key="back_deps_missing"):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("🔄 Check Again"):
                st.rerun()


def _render_final_validation(validator: OutlookHelperValidator):
    """Step 4: Final validation."""
    st.markdown("### Step 5: Final Validation")

    st.info("✨ **Almost there!** Click the button below to run final validation.")

    if st.button("🧪 Run Full Validation", type="primary"):
        with st.spinner("Validating configuration..."):
            result = validator.validate_all()

            if result.passed:
                st.success("✅ All checks passed!")

                # Show details
                with st.expander("📋 Validation Details", expanded=True):
                    for key, value in result.info.items():
                        st.markdown(f"**{key}**: {value}")

                if result.warnings:
                    st.warning("⚠️ Warnings:")
                    for warning in result.warnings:
                        st.warning(warning)

                time.sleep(2)
                st.session_state.wizard_step = 5
                st.rerun()
            else:
                st.error("❌ Validation failed")

                if result.errors:
                    st.markdown("**Errors:**")
                    for error in result.errors:
                        st.error(error)

                # Show detailed self-test output if helper self-test failed
                if any("self-test" in err.lower() for err in result.errors):
                    st.markdown("**Detailed Self-Test Output:**")
                    python_path = validator.config["windows"]["python_path"]
                    helper_path = validator.config["windows"]["helper_script"]
                    success, output = validator.run_helper_self_test_detailed(python_path, helper_path)

                    if output:
                        st.code(output, language="text")

                    st.info(
                        "💡 **Common Issues:**\n"
                        "- **Outlook not configured:** Open Outlook and add an email account\n"
                        "- **Outlook permissions:** Make sure Outlook is allowed to run COM automation\n"
                        "- **First-time setup:** Outlook may need to complete its first-run wizard\n"
                        "- **Antivirus/Security:** Some security software blocks COM automation"
                    )

                st.markdown("---")
                if st.button("← Back to Fix Issues"):
                    # Determine which step to go back to based on errors
                    error_str = " ".join(result.errors).lower()
                    if "python" in error_str:
                        st.session_state.wizard_step = 1
                    elif "helper script" in error_str:
                        st.session_state.wizard_step = 2
                    elif "package" in error_str or "pywin32" in error_str:
                        st.session_state.wizard_step = 3
                    else:
                        st.session_state.wizard_step = 0
                    st.rerun()

    st.markdown("---")
    if st.button("← Back", key="back_validation"):
        st.session_state.wizard_step = 3
        st.rerun()


def _render_completion():
    """Step 5: Setup complete."""
    st.markdown("### ✅ Setup Complete!")

    st.success(
        "🎉 **Congratulations!** Your Outlook helper is configured and ready to use."
    )

    st.markdown("---")
    st.markdown("### 📋 What's Next?")

    st.info(
        "**You can now:**\n"
        "1. Go to the **Outlook Integration** tab\n"
        "2. Create an Outlook project\n"
        "3. Extract emails from your Outlook folders\n"
        "4. Run the full RAG pipeline (ingest → ask)"
    )

    st.markdown("---")
    st.markdown("### 🧪 Test Your Setup")

    st.code("""
# Test extraction with a small sample
from scripts.connectors.outlook_wsl_client import get_outlook_connector
from scripts.connectors.outlook_connector import OutlookConfig

config = OutlookConfig(
    account_name="your-email@company.com",
    folder_path="Inbox",
    days_back=7,
    max_emails=5
)

connector = get_outlook_connector(config)
emails = connector.extract_emails()
print(f"Extracted {len(emails)} emails")
""", language="python")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Reset Wizard"):
            st.session_state.wizard_step = 0
            st.rerun()
    with col2:
        if st.button("✅ Done"):
            # Clear wizard state and return to main UI
            del st.session_state.wizard_step
            st.rerun()
