from types import SimpleNamespace
from scripts.utils.logger import LoggerManager
from scripts.pipeline.runner import PipelineRunner  # <-- replace with the real import path

class DummyProject: pass

pr = PipelineRunner(DummyProject(), config={}, run_id="demo_run_002")
pr.logger.info("pipeline.per_run_test", extra={"extra_data": {"check": "ok", "run": "demo_run_002"}})
print("WROTE_RUN:", "logs/runs/demo_run_002/app.log")