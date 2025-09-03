from pathlib import Path
from typing import Optional

class TaskPaths:
    """
    Unified logging path manager supporting both global and project-specific logs.
    
    Standardized structure:
    - Global logs: logs/app/<name>.log
    - Per-run logs: logs/runs/<run_id>/<name>.log  
    - Project logs: <project_root>/logs/<name>.log
    - Project runs: <project_root>/logs/runs/<run_id>/<name>.log
    """
    
    def __init__(self, logs_root: str = "logs", project_root: Optional[Path] = None):
        """
        Initialize TaskPaths with unified logging structure.
        
        Args:
            logs_root: Base logs directory name (default: "logs")
            project_root: If provided, creates project-specific log paths
        """
        if project_root:
            # Project-specific logging: <project_root>/logs/
            self.logs_root = Path(project_root) / logs_root
        else:
            # Global logging: logs/
            self.logs_root = Path(logs_root)

    def get_log_path(self, run_id: str | None = None, name: str = "app") -> str:
        """
        Get standardized log file path.
        
        Args:
            run_id: Optional run identifier for per-run logging
            name: Log file name (without .log extension)
            
        Returns:
            Full path to log file as string
        """
        if run_id:
            p = self.logs_root / "runs" / run_id / f"{name}.log"
        else:
            p = self.logs_root / f"{name}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
        
    def get_module_log_path(self, module_name: str, run_id: str | None = None) -> str:
        """
        Get log path for a specific module (replaces ProjectManager.get_log_path).
        
        Args:
            module_name: Module name (e.g., "ingestion", "chunker", "pipeline")
            run_id: Optional run identifier
            
        Returns:
            Full path to module log file
        """
        return self.get_log_path(run_id=run_id, name=module_name)
