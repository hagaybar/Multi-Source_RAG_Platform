# Plan: Outlook Connector Interop (WSL client + Windows helper)

## Goals
- Keep existing `scripts/connectors/outlook_connector.py` untouched for native Windows use.
- Add WSL-compatible pathway that delegates COM work to a shared Windows helper script.
- Provide configuration + Streamlit UX for validating/setting up the helper from WSL.

## Phases

### 1. Windows Helper Script
1.1 Create `tools/windows/win_com_server.py` (lives on Windows filesystem; document expected location).
1.2 Script responsibilities:
- Parse CLI args (account, folder path, days, max emails, attachment flag, logging path, output format).
- Initialize COM (`pythoncom.CoInitializeEx(0)`), run Outlook extraction (code copied from existing connector), emit JSON with `[{"body": "...", "metadata": {...}}, ...]` to stdout.
- Log errors to stderr and return non-zero exit codes.

1.3 Document dependency expectations (Python interpreter path, `pywin32`, Outlook availability).

### 2. Shared Configuration & Detection
2.1 Define new config section (e.g., `configs/outlook_helper.yaml`):
```
windows_python: "C:/path/to/venv/Scripts/python.exe"
helper_script: "C:/MultiSourceRAG/tools/win_com_server.py"
required_packages: ["pywin32"]
```
2.2 Add loader utility (e.g., `scripts/connectors/outlook_helper_config.py`) to read config, validate existence of paths via `/mnt/c/...` translation.
2.3 Provide CLI command `python scripts/tools/check_outlook_helper.py --config ...` that prints readiness status; used by CI/tests.

### 3. WSL Client Wrapper
3.1 Add new module `scripts/connectors/outlook_wsl_client.py`:
- Detect environment (`is_wsl()` helper).
- If running under WSL, use the helper config to build subprocess command and call Windows helper.
- Parse JSON output and convert back to `List[Tuple[str, dict]]`.
- Reuse `OutlookConfig` dataclass for inputs.

3.2 Update ingestion pipeline to choose connector implementation:
- Keep existing `OutlookConnector` for pure Windows.
- Introduce factory `get_outlook_connector(config)` that returns either native connector or WSL client based on environment/config readiness.

### 4. Streamlit UI Integration
4.1 Extend UI forms to collect/save helper configuration (paths to Windows Python + helper script). Store in new config file.
4.2 Add validation status panel before enabling Outlook source:
- Checks: files present, helper script reachable, Windows Python executable exists, pywin32 installed (use helper CLI `--self-test`).
- Provide call-to-action buttons: "Create helper script" (writes template to Windows path), "Set Python path" (opens file picker or prompts).

4.3 On failure, show actionable messages; on success, allow running extraction.

### 5. Helper Script Deployment Support
5.1 Provide template content for helper script (same COM logic; instructions in `docs/outlook_integration_plan.md`).
5.2 Offer script generator command `python scripts/tools/create_outlook_helper.py --dest /mnt/c/...` that writes template and ensures permissions.

### 6. Testing
6.1 Unit tests for new config loader, environment detection, subprocess wrapper (use mocks to simulate helper output/errors).
6.2 Integration test stub verifying factory selection when `is_wsl()` returns True/False.
6.3 Manual test checklist: run helper on Windows, call WSL client, confirm Streamlit UI shows readiness states.

### 7. Documentation
7.1 Update `docs/outlook_integration_plan.md` with WSL workflow, helper setup steps, config file format, troubleshooting.
7.2 Add README section summarizing dual-mode connector.

## Deliverables
- `tools/windows/win_com_server.py` (Windows helper script).
- `scripts/connectors/outlook_wsl_client.py` (WSL subprocess client).
- Config loader and CLI checks.
- Streamlit UI changes to configure/validate helper.
- Tests + documentation updates.
