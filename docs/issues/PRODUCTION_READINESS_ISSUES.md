# Production Readiness Issues - Analysis & Solutions

**Date:** 2025-11-23
**Priority:** HIGH
**Status:** Identified - Solutions Designed

---

## Issue 1: Config `prompt_strategy` Not Being Respected ⚠️

### Problem

**Config exists but is ignored:**
```yaml
# data/projects/Primo_List_2/config.yml
llm:
  prompt_strategy: default  # ← This is IGNORED!
```

**Current behavior:**
- PromptBuilder uses **auto-detection** logic (lines 238-251)
- If >50% of chunks are emails → uses EMAIL_PROMPT_TEMPLATE
- If ≤50% of chunks are emails → uses DEFAULT_PROMPT_TEMPLATE_V2
- **Config is never consulted**

**Why this happens:**
```python
# scripts/prompting/prompt_builder.py:238-251
# Auto-select template based on content type
if context_chunks and email_chunk_count > len(context_chunks) / 2:
    selected_template = self.email_template  # Auto-detection!
else:
    selected_template = self.template
```

No code reads `config['llm']['prompt_strategy']`!

### Impact

**User expectations violated:**
- Set `prompt_strategy: default` expecting default prompts
- System ignores config and auto-selects email prompts
- No visibility into which template is being used

**Debugging difficulty:**
- User can't control prompt selection
- Can't force email prompts for mixed datasets
- Can't force default prompts for email datasets

### Solution Options

#### Option A: Respect Config (Recommended) ⭐

**Modify PromptBuilder to load config:**
```python
class PromptBuilder:
    def __init__(self, template: str | None = None, run_id: Optional[str] = None,
                 project=None, strategy: str = None):
        """
        Args:
            strategy: Override prompt strategy. Options:
                - 'auto': Auto-detect based on content (default)
                - 'email': Force email template
                - 'default': Force default template
                - 'v2': Force default_v2 template
        """
        # Load strategy from config if not provided
        if strategy is None and project:
            strategy = project.config.get('llm', {}).get('prompt_strategy', 'auto')

        self.strategy = strategy or 'auto'
        self.template = template or DEFAULT_PROMPT_TEMPLATE_V2
        self.email_template = EMAIL_PROMPT_TEMPLATE

    def build_prompt(self, query: str, context_chunks: List[Chunk]) -> str:
        # Count email chunks
        email_chunk_count = sum(1 for c in context_chunks
                               if c.meta.get("doc_type") == "outlook_eml")

        # Select template based on strategy
        if self.strategy == 'email':
            selected_template = self.email_template
        elif self.strategy == 'default' or self.strategy == 'v2':
            selected_template = self.template
        elif self.strategy == 'auto':
            # Auto-detect (current behavior)
            if email_chunk_count > len(context_chunks) / 2:
                selected_template = self.email_template
            else:
                selected_template = self.template
        else:
            # Unknown strategy, fall back to auto
            selected_template = (self.email_template
                                if email_chunk_count > len(context_chunks) / 2
                                else self.template)

        # ... rest of method
```

**Benefits:**
- ✅ Config is respected
- ✅ User has control
- ✅ Backward compatible (default to 'auto')
- ✅ Clear logging of which template is used

**Files to modify:**
1. `scripts/prompting/prompt_builder.py` - Add strategy parameter
2. `scripts/pipeline/runner.py` - Pass config to PromptBuilder
3. `scripts/ui/ui_v3.py` - Pass config to PromptBuilder

#### Option B: Document Auto-Detection

**Just document current behavior:**
- Update config.yml comments
- Explain 'auto' detection logic
- Remove misleading `prompt_strategy` config

**Pros:** No code changes
**Cons:** User has no control

### Recommendation

✅ **Implement Option A** (2-3 hours)
- Gives users control they expect
- Honors config values
- Maintains auto-detection as default

---

## Issue 2: Pipeline Step Dependencies - Lack of Flexibility ⚠️

### Problem

**Rigid in-memory dependency:**
```python
# scripts/pipeline/runner.py
def step_chunk(self):
    if not self.raw_docs:
        print("❌ No raw documents available. Run 'ingest' first.")
        return
```

**Current behavior:**
- `ingest` creates `self.raw_docs` (in memory)
- `chunk` expects `self.raw_docs` to exist
- **If you run UI steps separately**, data is lost between steps!

**Example failure:**
1. User runs "Ingest" in UI → creates `self.raw_docs`
2. UI refreshes (or user navigates away)
3. User runs "Chunk" in UI → NEW PipelineRunner instance
4. `self.raw_docs` is None → ❌ Error!

**But the data EXISTS on disk:**
- Raw files: `input/raw/<extension>/`
- Should be able to chunk from disk!

### Impact

**Poor UX:**
- Can't run steps independently
- Must run full pipeline every time
- Loses data between UI interactions

**Development friction:**
- Can't test single steps
- Can't re-chunk after changing rules
- Can't re-embed after changing model

### Solution: Smart Fallback to Disk

**Modify each step to check disk if memory is empty:**

```python
def step_chunk(self):
    """Chunk documents (load from memory or disk)."""

    # Try memory first
    if not self.raw_docs:
        print("⚠️  No raw documents in memory, checking disk...")

        # Try loading from raw files
        raw_files = list(self.project.get_input_path("raw").rglob("*.*"))
        raw_files = [f for f in raw_files if f.is_file() and not f.name.startswith('.')]

        if raw_files:
            print(f"✓ Found {len(raw_files)} raw files on disk")
            print("  Running ingest first...")
            self.step_ingest()
        else:
            print("❌ No raw documents available.")
            print("   Options:")
            print("   1. Run 'ingest' first to load raw files")
            print("   2. Add files to input/raw/ directory")
            return

    # Now proceed with chunking...
    # (existing chunking logic)
```

**Similarly for other steps:**

```python
def step_embed(self):
    """Embed chunks (load from memory or disk)."""

    if not self.chunks:
        print("⚠️  No chunks in memory, checking disk...")

        # Try loading from chunk TSV files
        chunk_files = list(self.project.get_input_path().glob("chunks_*.tsv"))

        if chunk_files:
            print(f"✓ Found {len(chunk_files)} chunk files on disk")
            print("  Loading chunks from disk...")
            self.chunks = self._load_chunks_from_disk()
        else:
            print("❌ No chunks available.")
            print("   Run 'chunk' first or add chunk files")
            return

    # Proceed with embedding...
```

### Benefits

✅ **Flexible workflow:**
- Run steps independently
- Re-run any step without full pipeline
- Survives UI refreshes

✅ **Better error messages:**
- Shows what options user has
- Guides user to fix

✅ **Development friendly:**
- Test single steps
- Iterate on parameters

### Implementation

**Files to modify:**
1. `scripts/pipeline/runner.py`:
   - `step_chunk()` - Check disk for raw files
   - `step_embed()` - Check disk for chunk TSVs
   - `step_retrieve()` - Check disk for FAISS indices
   - Add `_load_chunks_from_disk()` helper

**Estimated effort:** 4-6 hours

---

## Issue 3: Change Management - No Dependency Tracking ⚠️ CRITICAL

### Problem

**Users making changes have no visibility into consequences:**

**Scenario 1: Adding more emails**
```
Current dataset: 600 emails, 600 chunks, 600 embeddings
User adds: 400 more emails

Questions:
- Will old chunks be deleted?          → No (good)
- Will new chunks deduplicate?         → Yes (good, via content_hash)
- Will I pay to re-embed old chunks?   → No (good, skip_duplicates)
- How do I know this will work?        → No visibility! ⚠️
```

**Scenario 2: Changing chunk rules**
```
Old rule: max_tokens=300
New rule: max_tokens=500

Questions:
- What happens to old chunks?          → They stay! (bad)
- Will I have mixed chunk sizes?       → YES (messy dataset)
- Will old + new chunks coexist?       → YES (inconsistent)
- Should I delete old chunks first?    → User doesn't know!
```

**Scenario 3: Changing embedding dimensions**
```
Old: text-embedding-3-large (3072 dims)
New: text-embedding-ada-002 (1536 dims)

Questions:
- Will FAISS index break?              → YES! ❌ Error
- Should I delete old index?           → Yes (but no warning)
- Will I lose all data?                → Depends (no visibility)
```

### Impact

**Production disasters:**
- Mixed chunk sizes in same project
- Incompatible FAISS indices
- Wasted API costs re-embedding
- Data corruption

**User confusion:**
- No idea what will happen
- Trial-and-error approach
- Fear of breaking things

### Solution: Dependency Validation System

**Design: Pre-flight checks before running pipeline**

```python
class PipelineValidator:
    """Validates pipeline changes and shows impact."""

    def __init__(self, project: ProjectManager):
        self.project = project
        self.warnings = []
        self.errors = []
        self.changes = []

    def validate_config_changes(self) -> Dict:
        """Check what changed in config since last run."""

        # Load last run config (if exists)
        last_config_path = self.project.root_dir / "output" / ".last_config.yml"
        if not last_config_path.exists():
            return {"first_run": True}

        with open(last_config_path) as f:
            last_config = yaml.safe_load(f)

        current_config = self.project.config
        changes = {}

        # Check chunking rules
        if last_config.get('chunking') != current_config.get('chunking'):
            changes['chunking'] = {
                'old': last_config.get('chunking'),
                'new': current_config.get('chunking'),
                'impact': 'CRITICAL',
                'warning': 'Chunking rules changed - will create mixed chunk sizes!',
                'recommendation': 'Delete old chunks first: rm input/chunks_*.tsv'
            }

        # Check embedding model
        old_model = last_config.get('embedding', {}).get('model')
        new_model = current_config.get('embedding', {}).get('model')
        if old_model != new_model:
            # Check dimensions
            old_dims = self._get_model_dims(old_model)
            new_dims = self._get_model_dims(new_model)

            if old_dims != new_dims:
                changes['embedding_dims'] = {
                    'old': f"{old_model} ({old_dims}d)",
                    'new': f"{new_model} ({new_dims}d)",
                    'impact': 'CRITICAL',
                    'error': 'Embedding dimension mismatch - FAISS index incompatible!',
                    'recommendation': 'Delete old index: rm -r output/faiss/ output/metadata/'
                }

        # Check data additions
        current_raw_count = len(list(self.project.get_input_path("raw").rglob("*.*")))
        last_raw_count = last_config.get('_metadata', {}).get('raw_file_count', 0)
        if current_raw_count > last_raw_count:
            changes['data_added'] = {
                'old_count': last_raw_count,
                'new_count': current_raw_count,
                'added': current_raw_count - last_raw_count,
                'impact': 'INFO',
                'info': f'Added {current_raw_count - last_raw_count} new files',
                'note': 'Deduplication will skip unchanged chunks (via content_hash)'
            }

        return changes

    def print_validation_report(self, changes: Dict):
        """Show user-friendly validation report."""

        if not changes:
            print("✅ No config changes detected - safe to proceed")
            return

        print("="*70)
        print("PIPELINE VALIDATION REPORT")
        print("="*70)

        # Critical errors (block execution)
        errors = [c for c in changes.values() if c.get('impact') == 'CRITICAL' and 'error' in c]
        if errors:
            print("\n❌ CRITICAL ERRORS - Cannot proceed:")
            for err in errors:
                print(f"\n  Problem: {err['error']}")
                print(f"  Recommendation: {err['recommendation']}")
            return False

        # Warnings (allow but warn)
        warnings = [c for c in changes.values() if c.get('impact') == 'CRITICAL' and 'warning' in c]
        if warnings:
            print("\n⚠️  WARNINGS - Proceed with caution:")
            for warn in warnings:
                print(f"\n  Warning: {warn['warning']}")
                print(f"  Recommendation: {warn['recommendation']}")

        # Info (just FYI)
        infos = [c for c in changes.values() if c.get('impact') == 'INFO']
        if infos:
            print("\n✓ INFO:")
            for info in infos:
                print(f"  • {info['info']}")
                if 'note' in info:
                    print(f"    Note: {info['note']}")

        print("\n" + "="*70)
        return True

# Usage in PipelineRunner:
def run_all(self):
    """Run full pipeline with validation."""

    # Validate before running
    validator = PipelineValidator(self.project)
    changes = validator.validate_config_changes()

    if not validator.print_validation_report(changes):
        print("\n❌ Pipeline execution blocked - fix errors first")
        return

    # Proceed with pipeline...
    self.step_ingest()
    self.step_chunk()
    # ...
```

### Example Output

**Scenario 1: Adding emails (safe)**
```
================================================================================
PIPELINE VALIDATION REPORT
================================================================================

✓ INFO:
  • Added 400 new files
    Note: Deduplication will skip unchanged chunks (via content_hash)

================================================================================
✅ Safe to proceed
```

**Scenario 2: Changed chunking (warning)**
```
================================================================================
PIPELINE VALIDATION REPORT
================================================================================

⚠️  WARNINGS - Proceed with caution:

  Warning: Chunking rules changed - will create mixed chunk sizes!
  Old: max_tokens=300
  New: max_tokens=500
  Recommendation: Delete old chunks first: rm input/chunks_*.tsv

================================================================================
⚠️  Proceed? [y/N]:
```

**Scenario 3: Changed embedding dims (error)**
```
================================================================================
PIPELINE VALIDATION REPORT
================================================================================

❌ CRITICAL ERRORS - Cannot proceed:

  Problem: Embedding dimension mismatch - FAISS index incompatible!
  Old: text-embedding-3-large (3072d)
  New: text-embedding-ada-002 (1536d)
  Recommendation: Delete old index: rm -r output/faiss/ output/metadata/

================================================================================
❌ Pipeline execution blocked - fix errors first
```

### Benefits

✅ **Safety:**
- Prevent data corruption
- Block incompatible changes
- Clear error messages

✅ **Visibility:**
- User knows what will happen
- Understands consequences
- Guided to fix issues

✅ **Confidence:**
- No fear of breaking things
- Clear next steps
- Professional UX

### Implementation

**Files to create:**
1. `scripts/pipeline/validator.py` - PipelineValidator class

**Files to modify:**
1. `scripts/pipeline/runner.py` - Call validator before each run
2. `scripts/ui/ui_v3.py` - Show validation in UI
3. Save config snapshot after successful run

**Estimated effort:** 6-8 hours

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1 day)

1. **Issue 1: Fix config respect** (2-3 hours)
   - High impact, low effort
   - Users get control they expect

2. **Issue 2: Add disk fallback to chunk/embed** (4-6 hours)
   - Immediate UX improvement
   - Enables independent step execution

**Total:** 6-9 hours

### Phase 2: Safety Net (1 day)

3. **Issue 3: Implement PipelineValidator** (6-8 hours)
   - Critical for production
   - Prevents disasters
   - Professional UX

**Total:** 6-8 hours

---

## Alternative: Quick Patches vs Full Solution

### Quick Patches (4 hours)

**Issue 1:** Document auto-detection, remove misleading config (30min)
**Issue 2:** Add better error messages with suggestions (2 hours)
**Issue 3:** Add warning banner in UI: "Changing config may require deleting old data" (1.5 hours)

**Pros:** Fast
**Cons:** Doesn't solve root problems

### Full Solution (12-17 hours)

**Implement all three solutions properly**

**Pros:** Production-ready, professional, safe
**Cons:** Takes 1.5-2 days

---

## Recommendation

✅ **Implement Full Solution** (Phases 1 & 2)

**Why:**
- These are production-readiness blockers
- Will save hours of debugging later
- Professional UX expected in production
- 12-17 hours is reasonable investment

**Priority:**
1. Issue 3 (validator) - Prevents disasters ⭐ CRITICAL
2. Issue 1 (config) - User expectations
3. Issue 2 (flexibility) - UX improvement

---

**Document Version:** 1.0
**Last Updated:** 2025-11-23
**Status:** Analysis Complete - Ready for Implementation
**Next Step:** Choose implementation approach
