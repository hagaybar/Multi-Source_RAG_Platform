from scripts.utils.logger import LoggerManager
from scripts.utils.task_paths import TaskPaths
from scripts.utils.logger_context import with_context

# Base logger (JSON file formatter), app-level
base = LoggerManager.get_logger(
    name="pipeline", task_paths=TaskPaths(), run_id=None, use_json=True
)

# Wrap with context
log = with_context(base, component="smoke", run_id="ctx_demo_001")

log.info("context.smoke.start")
log.warning("context.smoke.warning", extra={"extra_data": {"note": "hello"}})
log.info("context.smoke.end")
print("Wrote logs to logs/app/app.log")