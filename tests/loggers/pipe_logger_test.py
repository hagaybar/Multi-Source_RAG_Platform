from types import SimpleNamespace
from scripts.utils.task_paths import TaskPaths
from scripts.utils.logger import LoggerManager
# Import your real PipelineRunner (update the import path below):
from scripts.pipeline.runner import PipelineRunner  

pr = PipelineRunner(SimpleNamespace(), config={}, run_id=None)
pr.logger.info("pipeline.app_level_test")
print("WROTE_APP: logs/app/pipeline.log")
