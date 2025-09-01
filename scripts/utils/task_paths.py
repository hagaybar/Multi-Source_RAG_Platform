from pathlib import Path

class TaskPaths:
    def __init__(self, logs_root: str = "logs"):
        self.logs_root = Path(logs_root)

    def get_log_path(self, run_id: str | None = None, name: str = "app") -> str:
        if run_id:
            p = self.logs_root / "runs" / run_id / "app.log"
        else:
            p = self.logs_root / "app" / f"{name}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
