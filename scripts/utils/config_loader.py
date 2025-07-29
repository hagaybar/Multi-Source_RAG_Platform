import yaml
from pathlib import Path
from typing import Any


class ConfigLoader:
    """
    Loads and provides access to a YAML configuration file.
    Supports nested keys via dot notation.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.config = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

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
