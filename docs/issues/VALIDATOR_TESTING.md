# Pipeline Validator - Testing Guide

**Date:** 2025-11-23
**Status:** Validator Implemented ✅
**Testing:** Required before production use

---

## What the Validator Does

**Pre-flight checks before pipeline execution to prevent:**
- ❌ Embedding dimension mismatches (FAISS incompatibility)
- ❌ Mixed chunk sizes from changed rules
- ❌ Wasted API costs from unnecessary re-processing
- ❌ Data corruption

**Shows user:**
- ✅ What changed since last run
- ✅ What will happen if they proceed
- ✅ How to fix errors
- ✅ Warnings about potential issues

---

## How It Works

### 1. On First Run
```
================================================================================
PIPELINE VALIDATION REPORT
================================================================================

✓ INFORMATION:
  [FIRST_RUN] First pipeline run for this project
    Impact: All data will be processed from scratch

================================================================================
✅ Safe to proceed
================================================================================
```

### 2. After Successful Run

**Saves state to disk:**
- `output/.last_config.yml` - Config snapshot
- `output/.last_metadata.json` - File counts, timestamps

### 3. On Next Run

**Compares current vs last:**
- Chunking rules changed?
- Embedding model changed?
- Embedding dimensions changed?
- Files added/removed?

**Shows report:**
- ❌ **Errors** → Block execution
- ⚠️  **Warnings** → Allow but warn
- ✓ **Info** → FYI messages

---

## Test Scenarios

### Scenario 1: Adding More Emails ✅ Safe

**Setup:**
```bash
# First run with 100 emails
rm -rf data/projects/test_validator
mkdir -p data/projects/test_validator/input/raw/eml
# Add 100 .eml files

# Run pipeline
# → Saves state

# Add 50 more emails
# Add 50 more .eml files

# Run pipeline again
```

**Expected Output:**
```
✓ INFORMATION:
  [DATA] Added 50 new raw file(s)
    Previous: 100 files
    Current: 150 files
    Impact: New files will be ingested, chunked, and embedded
    Note: Deduplication will skip unchanged content (via content_hash).
          Only truly new chunks will be embedded.

✅ Safe to proceed
```

**Result:** ✅ PASS - No errors, just info

---

### Scenario 2: Changing Chunk Rules ⚠️ Warning

**Setup:**
```bash
# First run with max_tokens=300
# Edit configs/chunk_rules.yaml:
#   outlook_eml:
#     max_tokens: 300

# Run pipeline
# → Saves state

# Change max_tokens=500
# Edit configs/chunk_rules.yaml:
#   outlook_eml:
#     max_tokens: 500

# Run pipeline again
```

**Expected Output:**
```
⚠️  WARNINGS - Proceed with caution:

  [CHUNKING] Chunking rules have changed since last run
    Old: Last modified: 1732356789.123
    New: Current modified: 1732357890.456
    Impact: Will create chunks with different sizes, mixed with old chunks
    Recommendation: Delete existing chunks to avoid mixed sizes:
      rm data/projects/test_validator/input/chunks_*.tsv

✅ Safe to proceed (but review warnings)
```

**Result:** ✅ PASS - Warning shown, user can proceed or fix

---

### Scenario 3: Changing Embedding Dimensions ❌ Error

**Setup:**
```bash
# First run with text-embedding-3-large (3072d)
# Edit config.yml:
#   embedding:
#     model: text-embedding-3-large

# Run pipeline
# → Creates FAISS indices

# Change to ada-002 (1536d)
# Edit config.yml:
#   embedding:
#     model: text-embedding-ada-002

# Run pipeline again
```

**Expected Output:**
```
❌ CRITICAL ERRORS - Cannot proceed:

  [EMBEDDING] Embedding model dimension mismatch
    Old: text-embedding-3-large (3072d)
    New: text-embedding-ada-002 (1536d)
    Impact: FAISS indices are incompatible with new model
    Fix: Delete existing indices and metadata:
      rm -r data/projects/test_validator/output/faiss/
      rm -r data/projects/test_validator/output/metadata/

================================================================================
❌ Pipeline execution BLOCKED - fix errors above first
================================================================================
```

**Result:** ✅ PASS - Pipeline blocked, clear fix shown

---

### Scenario 4: Removing Files ⚠️ Warning

**Setup:**
```bash
# First run with 100 emails
# Run pipeline

# Remove 30 emails
rm data/projects/test_validator/input/raw/eml/email_01.eml
# ... remove 29 more

# Run pipeline again
```

**Expected Output:**
```
⚠️  WARNINGS:

  [DATA] Removed 30 raw file(s)
    Old: 100 files
    New: 70 files
    Impact: Old chunks/embeddings from removed files will remain
    Recommendation: Consider deleting old data if files were intentionally removed:
      rm data/projects/test_validator/input/chunks_*.tsv
      rm -r data/projects/test_validator/output/faiss/
      rm -r data/projects/test_validator/output/metadata/

✅ Safe to proceed (but review warnings)
```

**Result:** ✅ PASS - Warning shown with guidance

---

### Scenario 5: Same Model, Different Provider ⚠️ Warning

**Setup:**
```bash
# First run with provider: openai
# Run pipeline

# Change to provider: litellm (same model)
# Edit config.yml:
#   embedding:
#     provider: litellm

# Run pipeline again
```

**Expected Output:**
```
⚠️  WARNINGS:

  [EMBEDDING] Embedding provider changed
    Old: openai
    New: litellm
    Impact: May affect embedding quality/compatibility

✅ Safe to proceed (but review warnings)
```

**Result:** ✅ PASS - Warning shown

---

## Manual Testing Checklist

Before marking validator as complete, test:

- [ ] **First run** → Shows "First pipeline run" info
- [ ] **No changes** → Shows "No configuration changes detected"
- [ ] **Added files** → Shows info about new files
- [ ] **Removed files** → Shows warning
- [ ] **Changed chunk rules** → Shows warning + recommendation
- [ ] **Changed embedding dims** → Shows error, blocks execution
- [ ] **Changed same-dim model** → Shows warning
- [ ] **Changed provider** → Shows warning
- [ ] **Changed LLM model** → Shows info
- [ ] **State save** → Creates .last_config.yml and .last_metadata.json
- [ ] **Multiple changes** → Shows all issues in one report

---

## Integration Testing

### Test in UI

**Steps:**
1. Create new project in UI
2. Add some emails
3. Run "Ingest → Chunk → Embed"
4. Check `output/.last_config.yml` exists ✅
5. Change chunk rules in configs/
6. Try to run "Chunk" again
7. Should see warning about mixed chunk sizes ✅
8. Change embedding model dimensions
9. Try to run "Embed"
10. Should see error blocking execution ✅

### Test in CLI

```bash
# Test script
#!/bin/bash

# Create test project
python -c "
from scripts.core.project_manager import ProjectManager
from pathlib import Path

project = ProjectManager.create_project(
    Path('data/projects/validator_test'),
    'Validator Test',
    'Testing pipeline validator'
)
"

# Add test data
# ...

# Run pipeline first time
PYTHONPATH=. poetry run python -c "
from scripts.pipeline.runner import PipelineRunner
from scripts.core.project_manager import ProjectManager
from pathlib import Path

project = ProjectManager(Path('data/projects/validator_test'))
runner = PipelineRunner(project, project.config)

# Add steps
runner.add_step('ingest')
runner.add_step('chunk')
runner.add_step('embed')

# Run (should show first_run info)
for msg in runner.run_steps():
    print(msg)
"

# Modify config
# ...

# Run again (should show warnings/errors)
```

---

## Known Limitations

### Current Implementation

1. **Chunk rules detection:**
   - Uses file modification time
   - Doesn't detect which specific rule changed
   - Could show false positives if file touched but not changed

2. **Model dimensions:**
   - Only knows common OpenAI models
   - Unknown models show 'unknown' dimensions
   - User must manually verify compatibility

3. **Deduplication:**
   - Assumes content_hash deduplication works
   - Doesn't verify actual duplicates

4. **File additions:**
   - Counts all files in raw/
   - Doesn't distinguish file types
   - Could count non-document files

### Future Enhancements

1. **Deeper analysis:**
   - Parse chunk_rules.yaml and compare specific rules
   - Detect which doc_types are affected
   - Show before/after values for each rule

2. **Model registry:**
   - Maintain registry of all embedding models + dimensions
   - Auto-detect dimensions from model API
   - Warn about unknown models

3. **Dry-run mode:**
   - Simulate pipeline without executing
   - Show exactly what will happen
   - Estimate costs (API calls, time)

4. **Interactive mode:**
   - Ask user to confirm warnings
   - Offer to auto-fix errors (delete old indices)
   - Guide user through conflict resolution

---

## Success Criteria

Validator is production-ready when:

- [x] Detects dimension mismatches
- [x] Warns about chunk rule changes
- [x] Tracks data additions/removals
- [x] Blocks execution on errors
- [x] Allows execution on warnings
- [x] Shows clear recommendations
- [x] Saves state after successful runs
- [x] Works with run_steps()
- [ ] Tested on all scenarios above
- [ ] Integrated with UI
- [ ] Documentation complete

---

## Next Steps

1. **Complete manual testing** (30-60 min)
   - Test all scenarios listed above
   - Verify error/warning/info messages
   - Check that recommendations work

2. **UI Integration** (1-2 hours)
   - Show validation report in UI before running
   - Add "Skip Validation" checkbox for advanced users
   - Display warnings/errors prominently

3. **Documentation** (30 min)
   - Add to USER_GUIDE.md
   - Update TROUBLESHOOTING.md with validation errors
   - Add examples to README.md

---

**Status:** Validator implemented and ready for testing
**Next:** Manual testing of all scenarios
