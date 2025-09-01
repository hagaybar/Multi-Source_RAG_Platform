import yaml
from pathlib import Path
from typing import Any

from scripts.utils.logger import LoggerManager
from scripts.utils.task_paths import TaskPaths

class ConfigLoader:
    """
    Loads and provides access to a YAML configuration file.
    Supports nested keys via dot notation.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.config = self._load()

    def _get_logger(self):
        # app-level project/config logs → logs/app/project.log
        return LoggerManager.get_logger(
            name="project", task_paths=TaskPaths(), run_id=None, use_json=True
        )

    def _load(self):
        log = self._get_logger()
        path = self.path
        if not path.exists():
            # LOG, then raise
            log.error("config.missing", extra={"extra_data": {"path": str(path)}})
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            # your existing file reading / yaml load
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)  # or whatever you use
            if not isinstance(data, dict):
                log.error("config.invalid_type", extra={"extra_data": {"path": str(path)}})
                raise ValueError(f"Invalid config (expected mapping) at {path}")
            log.info("config.loaded", extra={"extra_data": {"path": str(path)}})
            return data
        except Exception as e:
            # Ensure parse errors are captured
            log.error("config.load.fail", extra={"extra_data": {"path": str(path), "error": str(e)}}, exc_info=True)
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Supports dot notation for nested access."""
        parts = key.split(".")
        val = self.config
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val

    def as_dict(self) -> dict:
        return self.config
