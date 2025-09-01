from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.pipeline.runner import PipelineRunner  # adjust import if your module path differs

project = ProjectManager(Path("data/projects/demo"))  # pick a small/valid project
runner = PipelineRunner(project, config={})

runner.clear_steps()
runner.add_step("retrieve", query="what’s new?", strategy="late_fusion", top_k=3)
runner.add_step("ask", query="what’s new?", model_name="gpt-4o-mini")  # or your default
for msg in runner.run_steps():
    print(msg)