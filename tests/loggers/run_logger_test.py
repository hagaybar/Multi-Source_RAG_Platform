
from pathlib import Path


from scripts.utils.run_logger import RunLogger

r = RunLogger(project_dir=Path('.'), run_name="demo_step1")
print("Run dir:", r.base_dir)

# Write a few files via the existing APIs:
r.log_metadata({"ok": True, "note": "step1 smoke"})
r.log_prompt("Hello from Step 1")
r.log_response("World from Step 1")

# List created files
print("Contents:", sorted(p.name for p in r.base_dir.iterdir()))
assert str(r.base_dir).endswith("logs/runs/demo_step1")
assert (r.base_dir / "llm_prompt.txt").exists()
assert (r.base_dir / "llm_response.txt").exists()
assert (r.base_dir / "run_metadata.json").exists()
print("✓ Step 1 verification passed.")