# Evaluation: Outlook WSL-Windows Helper Plan

**Date:** 2025-01-19
**Evaluator:** Claude Code
**Status:** Critical Path Decision

---

## Executive Summary

**The Plan:** Split Outlook connector into Windows-only helper (COM access) and WSL client (subprocess caller) to overcome the dual incompatibility:
- Windows: FAISS + OpenMP DLL conflict in retrieve/ask steps
- WSL: No COM access for Outlook

**Verdict:** ✅ **APPROVED WITH MODIFICATIONS**

**Confidence:** High (85%)

**Recommendation:** Implement with the suggested improvements below. This is a pragmatic solution to a genuine cross-platform constraint.

---

## Problem Statement Validation

### Root Causes Confirmed

**Issue 1: Windows OpenMP Conflict**
```
Error: Two different OpenMP DLLs in same process
- FAISS brings in LLVM's OpenMP runtime
- Another library brings in different OpenMP version
- Process crash during retrieve/ask steps
```

**Issue 2: WSL COM Limitation**
```
Error: COM not available in WSL
- win32com.client requires Windows native COM
- WSL is Linux; no COM interop (unless using WSLg with specific setup)
- Cannot run outlook_connector.py directly from WSL
```

### Is This a Real Constraint?

✅ **YES** - These are genuine technical limitations:

1. **OpenMP DLL Conflict**: Real issue in Windows Python when multiple packages bundle OpenMP
   - Common with scientific computing libraries (numpy, scipy, FAISS)
   - Known problem in ML/DS community
   - Workarounds: environment variables, conda environments, or process isolation

2. **WSL COM Access**: WSL is a Linux subsystem, COM is Windows-only
   - COM requires Windows Process/Thread model
   - WSL2 uses Hyper-V VM, no direct COM access
   - WSLg adds GUI support but not full COM

### Alternative Solutions Considered?

Before accepting the helper plan, consider:

**Option A: Fix OpenMP Conflict on Windows**
- Set `KMP_DUPLICATE_LIB_OK=TRUE` environment variable
- Use conda instead of pip (better DLL management)
- Build FAISS without OpenMP
- **Pros:** Simpler, no cross-process communication
- **Cons:** Fragile, may not work, conda requirement

**Option B: Run Everything in WSL with Outlook Web Access**
- Use Microsoft Graph API instead of local COM
- OAuth2 authentication to Exchange/Outlook.com
- **Pros:** Cross-platform, no COM needed
- **Cons:** Requires API credentials, network access, different feature set

**Option C: Docker with Windows Containers**
- Run Outlook extraction in Windows container
- Run pipeline in Linux container
- **Pros:** Clean separation, reproducible
- **Cons:** Complex setup, requires Docker Desktop, Windows containers rare

**Option D: Dual Installation (Current State)**
- Windows: Extract emails only
- WSL: Run full pipeline (ingest → embed → ask)
- **Pros:** Already works, simple
- **Cons:** Manual file transfer between environments

**Why Helper Plan is Better than Alternatives:**
- More robust than Option A (env var hacks)
- Simpler than Option B (no API setup required)
- Lighter than Option C (no Docker complexity)
- More integrated than Option D (automated, not manual)

---

## Plan Architecture Evaluation

### Design Strengths ✅

1. **Separation of Concerns**
   - Windows helper: Pure COM extraction logic
   - WSL client: Pure subprocess orchestration
   - Clear boundaries, easy to test

2. **Backward Compatibility**
   - Keeps existing `outlook_connector.py` untouched
   - Existing Windows-only users unaffected
   - Factory pattern allows graceful fallback

3. **Configuration-Driven**
   - Paths stored in config file
   - Validation before execution
   - User can update paths without code changes

4. **Testability**
   - Helper can be tested independently on Windows
   - WSL client can use mocked subprocess
   - Factory can be unit tested for environment detection

5. **Error Handling**
   - Separate stderr for errors
   - Non-zero exit codes for failures
   - JSON parsing errors caught in WSL client

### Design Weaknesses ⚠️

1. **Cross-Process Complexity**
   - Adds subprocess management overhead
   - JSON serialization/deserialization
   - Potential encoding issues (Windows-1252 vs UTF-8)

2. **Path Translation Fragility**
   - `/mnt/c/...` ↔ `C:\...` conversions
   - Spaces in paths (Windows common)
   - Case sensitivity differences

3. **Dependency Duplication**
   - Two Python environments (WSL + Windows)
   - pywin32 must be installed in Windows Python
   - Version skew risk

4. **Configuration Burden**
   - User must set up Windows Python path
   - Helper script must be manually deployed
   - No auto-discovery mechanism

5. **Performance Overhead**
   - Process spawn on every extraction
   - JSON encoding/decoding
   - No connection pooling (COM init every time)

6. **Security Considerations**
   - Executing Windows scripts from WSL
   - JSON injection if args not sanitized
   - No authentication/authorization between WSL ↔ Windows

---

## Phase-by-Phase Analysis

### Phase 1: Windows Helper Script ✅ APPROVED

**Evaluation:**
- ✅ Clear responsibilities
- ✅ JSON output to stdout is standard
- ✅ Reuses existing connector logic
- ⚠️ Should include `--version` flag for compatibility checks
- ⚠️ Should support `--dry-run` for testing without extraction

**Recommended Changes:**
1. Add `--self-test` flag that validates:
   - Python version
   - pywin32 installed
   - Outlook installed and accessible
   - COM initialization works

2. Add `--version` flag to check compatibility

3. Use structured logging to stderr (not just errors):
   ```python
   import json, sys
   def log(level, message, **kwargs):
       json.dump({"level": level, "message": message, **kwargs}, sys.stderr)
       sys.stderr.write("\n")
   ```

4. Add timeout parameter (default 60s) to prevent hangs

### Phase 2: Shared Configuration & Detection ✅ APPROVED

**Evaluation:**
- ✅ YAML config is appropriate
- ✅ Path validation is critical
- ⚠️ Config file location not specified (should be in project root or user home)
- ⚠️ No version checking between WSL client and Windows helper

**Recommended Changes:**
1. Config file location: `~/.outlook_helper.yaml` (user-specific) or `configs/outlook_helper.yaml` (project-wide)

2. Add version field to config:
   ```yaml
   version: "1.0"
   windows_python: "C:/Users/hagay/AppData/Local/Programs/Python/Python311/python.exe"
   helper_script: "C:/MultiSourceRAG/tools/win_com_server.py"
   helper_version: "1.0"  # ← Must match helper script version
   timeout: 60
   ```

3. Auto-detection for Windows Python:
   ```python
   # Try common locations first:
   common_paths = [
       "C:/Python311/python.exe",
       "C:/Program Files/Python311/python.exe",
       "C:/Users/{user}/AppData/Local/Programs/Python/Python311/python.exe"
   ]
   for path in common_paths:
       wsl_path = f"/mnt/c{path[2:].replace('\\', '/')}"
       if os.path.exists(wsl_path):
           return path
   ```

4. Validation should check:
   - File existence via WSL path (`/mnt/c/...`)
   - File executability
   - Python version compatibility (>= 3.11)
   - Helper script imports (`import win32com.client`)

### Phase 3: WSL Client Wrapper ⚠️ NEEDS IMPROVEMENT

**Evaluation:**
- ✅ Clean abstraction
- ✅ Reuses `OutlookConfig`
- ⚠️ Subprocess error handling needs more detail
- ⚠️ No retry logic for transient failures
- ⚠️ Encoding issues not addressed

**Critical Issues:**

1. **Environment Detection**
   ```python
   def is_wsl():
       # Current plan: Check for WSL
       # Problem: What if running in pure Linux? Docker?
   ```

   **Better approach:**
   ```python
   def is_wsl():
       """Detect if running in WSL (not just Linux)."""
       try:
           with open('/proc/version', 'r') as f:
               return 'microsoft' in f.read().lower()
       except:
           return False

   def can_use_windows_helper():
       """Check if Windows filesystem is accessible."""
       return os.path.exists('/mnt/c/') and is_wsl()
   ```

2. **Subprocess Command Building**
   ```python
   # Potential issues:
   # - Spaces in paths
   # - Special characters in args (quotes, etc.)
   # - Windows path format vs WSL path format
   ```

   **Better approach:**
   ```python
   import shlex

   cmd = [
       config.windows_python,
       config.helper_script,
       "--account", shlex.quote(config.account_name),
       "--folder", shlex.quote(config.folder_path),
       "--days", str(config.days_back),
       # ... etc
   ]

   # Convert WSL paths to Windows paths for helper
   helper_script_win = wsl_to_windows_path(config.helper_script)
   cmd[1] = helper_script_win
   ```

3. **Encoding Handling**
   ```python
   # Windows uses UTF-16LE or Windows-1252 by default
   # JSON should be UTF-8

   proc = subprocess.run(
       cmd,
       capture_output=True,
       encoding='utf-8',  # ← Explicit UTF-8
       errors='replace',   # ← Handle decode errors gracefully
       timeout=config.timeout
   )
   ```

4. **Error Parsing**
   ```python
   if proc.returncode != 0:
       # Parse stderr for structured errors
       try:
           error_logs = [json.loads(line) for line in proc.stderr.split('\n') if line]
           for log in error_logs:
               if log.get('level') == 'ERROR':
                   raise OutlookExtractionError(log['message'])
       except json.JSONDecodeError:
           # Fallback to raw stderr
           raise OutlookExtractionError(proc.stderr)
   ```

**Recommended Changes:**

1. Add retry logic with exponential backoff:
   ```python
   max_retries = 3
   for attempt in range(max_retries):
       try:
           result = subprocess.run(...)
           break
       except subprocess.TimeoutExpired:
           if attempt == max_retries - 1:
               raise
           time.sleep(2 ** attempt)  # 1s, 2s, 4s
   ```

2. Add progress reporting:
   - Helper script emits progress to stderr
   - WSL client parses and displays in UI

3. Add connection pooling (future enhancement):
   - Keep helper process alive
   - Use stdin/stdout for requests/responses
   - Avoid COM init overhead on each extraction

### Phase 4: Streamlit UI Integration ✅ APPROVED

**Evaluation:**
- ✅ Validation before use is critical
- ✅ Actionable error messages
- ✅ Self-test capability
- ⚠️ "Create helper script" button might be too magical

**Recommended Changes:**

1. **Configuration Wizard Flow:**
   ```
   Step 1: Detect Environment
   - [ ] Running in WSL? → YES
   - [ ] Can access /mnt/c/? → YES

   Step 2: Locate Windows Python
   - Auto-detect: [Found at C:\Python311\python.exe]
   - Or manually enter path: [_______________]

   Step 3: Deploy Helper Script
   - Copy to Windows: [/mnt/c/MultiSourceRAG/tools/]
   - [Test Helper] button

   Step 4: Validate Setup
   - [✓] Helper script exists
   - [✓] Python version compatible (3.11.2)
   - [✓] pywin32 installed
   - [✓] Outlook accessible
   - [✓] Self-test passed

   [Save Configuration] [Skip - Use Windows Native]
   ```

2. **Status Panel:**
   ```
   Outlook Helper Status:
   ┌─────────────────────────────────────┐
   │ Environment: WSL2                   │
   │ Helper: Configured ✓                │
   │ Last Test: 2025-01-19 14:30 ✓       │
   │ Version: 1.0 (compatible)           │
   │                                     │
   │ [Test Connection] [Reconfigure]     │
   └─────────────────────────────────────┘
   ```

3. **Error Messages:**
   - ❌ "Helper script not found at C:\path\to\script.py"
     → "Create script" button OR "Choose different location"

   - ❌ "pywin32 not installed in Windows Python"
     → "Run in Windows terminal: pip install pywin32"

   - ❌ "Outlook not accessible"
     → "Ensure Outlook is installed and configured"

### Phase 5: Helper Script Deployment ⚠️ RISKY

**Evaluation:**
- ⚠️ Automated deployment is convenient but risky
- ⚠️ Writing to Windows filesystem from WSL can have permission issues
- ⚠️ Version updates require re-deployment

**Concerns:**

1. **Security:** Writing executable Python scripts from WSL to Windows
2. **Permissions:** `/mnt/c/` might have restricted write access
3. **Overwrite:** What if user customized the helper script?

**Recommended Changes:**

1. **Manual Deployment Preferred:**
   - Provide downloadable `win_com_server.py` template
   - User manually copies to Windows location
   - Less magic, more transparent

2. **If Automated Deployment:**
   ```python
   def deploy_helper_script(dest_path):
       """Deploy helper script with safety checks."""
       # 1. Check if dest already exists
       if os.path.exists(dest_path):
           # Prompt: Overwrite? Backup existing?
           backup_path = f"{dest_path}.backup.{int(time.time())}"
           shutil.copy(dest_path, backup_path)
           logger.warning(f"Backed up existing to {backup_path}")

       # 2. Write with proper permissions
       template = load_template("win_com_server.py.template")
       with open(dest_path, 'w', encoding='utf-8') as f:
           f.write(template)

       # 3. Validate deployment
       if not validate_helper_script(dest_path):
           os.remove(dest_path)
           raise DeploymentError("Deployed script failed validation")

       return True
   ```

3. **Version Management:**
   - Embed version string in helper script
   - Check version compatibility before use
   - Warn user if version mismatch

### Phase 6: Testing ✅ APPROVED

**Evaluation:**
- ✅ Good test coverage plan
- ⚠️ Integration tests need more detail

**Recommended Additions:**

1. **Unit Tests:**
   ```python
   # test_outlook_wsl_client.py
   def test_is_wsl():
       # Mock /proc/version
       ...

   def test_command_building():
       # Test with spaces in paths, special chars
       ...

   def test_json_parsing():
       # Test with valid/invalid JSON
       ...

   def test_error_handling():
       # Test non-zero exit codes, timeouts
       ...
   ```

2. **Integration Tests:**
   ```python
   # test_outlook_helper_integration.py
   @pytest.mark.skipif(not is_wsl(), reason="WSL only")
   def test_wsl_to_windows_extraction():
       # Requires Windows helper installed
       config = load_test_config()
       client = OutlookWSLClient(config)
       emails = client.extract_emails()
       assert len(emails) > 0

   @pytest.mark.skipif(is_wsl(), reason="Windows only")
   def test_windows_native_extraction():
       # Test native connector
       ...
   ```

3. **Manual Test Checklist:**
   ```
   [ ] Windows native: Extract emails directly
   [ ] WSL client: Extract emails via helper
   [ ] Config wizard: Complete setup flow
   [ ] Self-test: Run helper --self-test
   [ ] Error handling: Invalid paths, missing pywin32
   [ ] Encoding: Test Hebrew email subjects
   [ ] Performance: Extract 100 emails via helper
   [ ] UI: Status panel updates correctly
   ```

### Phase 7: Documentation ✅ APPROVED

**Evaluation:**
- ✅ Documentation is critical for this complexity
- ⚠️ Should include troubleshooting guide

**Recommended Additions:**

1. **Architecture Diagram:**
   ```
   ┌──────────────────────────────────────────────┐
   │           WSL2 Linux Environment             │
   │                                              │
   │  ┌────────────────────────────────────────┐  │
   │  │  Streamlit UI (ui_v3.py)               │  │
   │  │    ↓                                   │  │
   │  │  OutlookWSLClient                      │  │
   │  │    ↓                                   │  │
   │  │  subprocess.run(                       │  │
   │  │    cmd=[windows_python, helper, ...]  │  │
   │  │  )                                     │  │
   │  └─────────────│──────────────────────────┘  │
   │                ↓ JSON over stdout/stderr      │
   └────────────────│──────────────────────────────┘
                    │ Cross-process boundary
   ┌────────────────│──────────────────────────────┐
   │                ↓    Windows Environment       │
   │  ┌────────────────────────────────────────┐  │
   │  │  win_com_server.py                     │  │
   │  │    ↓                                   │  │
   │  │  pythoncom.CoInitializeEx()            │  │
   │  │    ↓                                   │  │
   │  │  Outlook.Application COM Object        │  │
   │  │    ↓                                   │  │
   │  │  Extract emails via MAPI               │  │
   │  │    ↓                                   │  │
   │  │  JSON.dump(emails) → stdout            │  │
   │  └────────────────────────────────────────┘  │
   └──────────────────────────────────────────────┘
   ```

2. **Troubleshooting Guide:**
   ```markdown
   ## Common Issues

   ### "Helper script not found"
   - Check path in outlook_helper.yaml
   - Verify /mnt/c/ is accessible: ls /mnt/c/
   - Convert path: C:\path → /mnt/c/path

   ### "pywin32 not installed"
   - Open Windows terminal (not WSL)
   - Run: pip install pywin32
   - Verify: python -c "import win32com.client"

   ### "Timeout expired"
   - Increase timeout in config.yaml
   - Check Outlook is not frozen
   - Try with smaller max_emails

   ### "JSON decode error"
   - Check helper script logs in Windows
   - Verify UTF-8 encoding
   - Test helper directly: python helper.py --self-test
   ```

3. **Setup Guide:**
   ```markdown
   ## WSL + Windows Setup

   ### Prerequisites
   1. WSL2 with Ubuntu
   2. Windows with Outlook installed
   3. Python 3.11+ in both WSL and Windows

   ### Step 1: Install WSL Dependencies
   ```bash
   cd /path/to/project
   poetry install
   ```

   ### Step 2: Install Windows Dependencies
   ```powershell
   # In Windows PowerShell
   cd C:\path\to\project
   pip install pywin32
   ```

   ### Step 3: Deploy Helper Script
   ```bash
   # From WSL
   python scripts/tools/create_outlook_helper.py \
     --dest /mnt/c/MultiSourceRAG/tools/win_com_server.py
   ```

   ### Step 4: Configure Paths
   Edit `configs/outlook_helper.yaml`:
   ```yaml
   windows_python: "C:/Python311/python.exe"
   helper_script: "C:/MultiSourceRAG/tools/win_com_server.py"
   ```

   ### Step 5: Test Setup
   ```bash
   python scripts/tools/check_outlook_helper.py --config configs/outlook_helper.yaml
   ```
   ```

---

## Risk Assessment

### High Risks 🔴

1. **Configuration Complexity**
   - Users must set up two Python environments
   - Path translations are error-prone
   - **Mitigation:** Auto-detection, wizard, clear error messages

2. **Version Skew**
   - WSL client and Windows helper must stay compatible
   - Dependency version mismatches
   - **Mitigation:** Version checking, explicit compatibility matrix

3. **Subprocess Reliability**
   - Timeouts, hangs, encoding errors
   - COM initialization failures
   - **Mitigation:** Retry logic, timeouts, structured error handling

### Medium Risks 🟡

4. **Performance Overhead**
   - Process spawn on every extraction (1-2s overhead)
   - JSON serialization/deserialization
   - **Mitigation:** Accept for MVP, optimize later with persistent helper

5. **Security Concerns**
   - Executing Windows scripts from WSL
   - No authentication between environments
   - **Mitigation:** Validate paths, sanitize args, document risks

6. **Maintenance Burden**
   - Two codepaths to maintain (native + WSL)
   - Testing requires both environments
   - **Mitigation:** Factory pattern reduces coupling, shared tests

### Low Risks 🟢

7. **UI Complexity**
   - Configuration wizard might be overwhelming
   - **Mitigation:** Progressive disclosure, good UX

8. **Documentation Gaps**
   - Users might not understand setup
   - **Mitigation:** Comprehensive docs with screenshots

---

## Comparison to Alternatives

| Criterion | Helper Plan | Fix OpenMP | Graph API | Docker | Dual Install |
|-----------|-------------|------------|-----------|---------|--------------|
| **Complexity** | Medium | Low | High | High | Low |
| **Reliability** | Medium | Low | High | High | High |
| **Setup Time** | 10-15 min | 1 min | 30-60 min | 60+ min | 5 min |
| **Maintenance** | Medium | Low | Medium | High | Low |
| **Cross-Platform** | WSL+Win | Win only | All | All | Win+WSL |
| **User Experience** | Good | Fragile | Good | Complex | Manual |
| **Performance** | Medium | High | Medium | High | High |
| **Cost** | Free | Free | API costs | Free | Free |

**Verdict:** Helper plan is the best balance of complexity, reliability, and user experience for this specific constraint.

---

## Recommendations

### Critical Changes (Must Have) 🔴

1. **Add version compatibility checking**
   - Helper script version must match client expectations
   - Fail early with clear message if mismatch

2. **Improve subprocess error handling**
   - Structured logging from helper
   - Retry logic with exponential backoff
   - Timeout protection

3. **Path validation and translation**
   - Robust `/mnt/c/` ↔ `C:\` conversion
   - Handle spaces, special characters
   - Validate before execution

4. **Auto-detection for Windows Python**
   - Check common installation locations
   - Reduce configuration burden
   - Fallback to manual entry

### Important Improvements (Should Have) 🟡

5. **Configuration wizard in UI**
   - Step-by-step setup flow
   - Validation at each step
   - Clear success/failure indicators

6. **Self-test capability**
   - Helper script `--self-test` flag
   - Validates environment end-to-end
   - Useful for CI/CD

7. **Comprehensive documentation**
   - Architecture diagram
   - Setup guide with screenshots
   - Troubleshooting section

8. **Integration tests**
   - Test WSL client with mocked helper
   - Test helper script directly on Windows
   - Test factory selection logic

### Nice to Have (Could Have) 🟢

9. **Connection pooling** (future enhancement)
   - Keep helper process alive
   - Reuse COM connection
   - Reduce overhead

10. **Progress reporting**
    - Helper emits progress to stderr
    - UI shows real-time status
    - Better UX for large extractions

11. **Automatic helper updates**
    - Check for newer helper version
    - Prompt user to update
    - Self-update mechanism

---

## Implementation Strategy

### Recommended Approach: **Phased with Quick Wins**

**Week 1: MVP (Phases 1-3)**
- ✅ Windows helper script with basic JSON output
- ✅ WSL client with subprocess wrapper
- ✅ Factory pattern for connector selection
- ✅ Manual configuration (YAML file)
- **Goal:** Basic extraction works from WSL

**Week 2: UX (Phase 4)**
- ✅ Configuration wizard in Streamlit
- ✅ Validation and self-test
- ✅ Error messages and troubleshooting
- **Goal:** Users can set up without reading docs

**Week 3: Polish (Phases 5-7)**
- ✅ Helper deployment automation
- ✅ Comprehensive testing
- ✅ Documentation with examples
- **Goal:** Production-ready, maintainable

### Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| 1. Windows Helper | 4-6 hours | Low |
| 2. Configuration | 2-3 hours | Medium |
| 3. WSL Client | 6-8 hours | High |
| 4. UI Integration | 6-8 hours | Medium |
| 5. Deployment | 3-4 hours | Medium |
| 6. Testing | 6-8 hours | Low |
| 7. Documentation | 4-6 hours | Low |
| **Total** | **31-43 hours** | **Medium** |

**Equivalent:** 4-5 working days for one developer

---

## Decision Criteria

### When to Implement This Plan

✅ **YES, implement if:**
- OpenMP conflict cannot be resolved with environment variables
- Graph API is not an option (security, network, features)
- Docker is too heavy for your workflow
- You frequently need to extract fresh emails from Outlook
- Development will continue in WSL long-term

❌ **NO, reconsider if:**
- OpenMP conflict can be fixed with `KMP_DUPLICATE_LIB_OK=TRUE`
- You can switch to Graph API for email access
- Email extraction is infrequent (manual file transfer is fine)
- You can develop fully on Windows (no WSL requirement)

### Success Criteria

**MVP Success:**
- [ ] Extract emails from WSL using Windows helper
- [ ] Configuration works for at least one user
- [ ] Error messages are actionable
- [ ] Self-test validates setup

**Production Success:**
- [ ] Setup time < 15 minutes for new users
- [ ] Extraction performance acceptable (< 2x slower than native)
- [ ] Fewer than 5 support requests about setup per month
- [ ] Both native and WSL paths work reliably

---

## Final Verdict

### ✅ APPROVE WITH MODIFICATIONS

**Rationale:**
1. The plan addresses a real, unsolvable technical constraint
2. Architecture is sound with clear separation of concerns
3. Backward compatibility preserved (native Windows unchanged)
4. Recommended improvements address all major risks
5. Effort is reasonable (4-5 days) for the value provided

**Key Modifications Required:**
1. Add version compatibility checking
2. Improve subprocess error handling and retries
3. Implement auto-detection for Windows Python
4. Add self-test capability to helper
5. Create configuration wizard in UI
6. Write comprehensive troubleshooting guide

**Proceed?** Yes, implement with the recommended modifications.

**Confidence Level:** 85% - This is a pragmatic solution given the constraints. The main risk is configuration complexity, which can be mitigated with good UX and documentation.

---

## Next Steps

1. **Review this evaluation** with stakeholders
2. **Decide:** Implement helper plan OR explore OpenMP fix further
3. **If approved:** Start with Week 1 MVP (Phases 1-3)
4. **Validate:** Test MVP with real Outlook extraction
5. **Iterate:** Add UX improvements based on feedback

**Ready to proceed?** Let me know your decision.
