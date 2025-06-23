from pathlib import Path
from scripts.utils.config_loader import ConfigLoader

class ProjectManager:
    """
    Represents a RAG project workspace with its own config, input, and output directories.
    """
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).resolve()
        self.config_path = self.root_dir / "config.yml"
        
        raw_config = ConfigLoader(self.config_path)
        self.config = raw_config.as_dict()  

        self.input_dir = self.root_dir / self.config.get("paths.input_dir", "input")
        self.output_dir = self.root_dir / self.config.get("paths.output_dir", "output")
        self.logs_dir = self.root_dir / self.config.get("paths.logs_dir", "output/logs")
        self.faiss_dir = self.root_dir / self.config.get("paths.faiss_dir", "output/faiss")
        self.metadata_dir = self.root_dir / self.config.get("paths.metadata_dir", "output/metadata")

        self._ensure_directories()

    def _ensure_directories(self):
        for path in [self.input_dir, self.output_dir, self.logs_dir, self.faiss_dir, self.metadata_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_input_dir(self) -> Path:
        return self.input_dir

    def get_faiss_path(self, doc_type: str) -> Path:
        return self.faiss_dir / f"{doc_type}.faiss"

    def get_metadata_path(self, doc_type: str) -> Path:
        return self.metadata_dir / f"{doc_type}_metadata.jsonl"

    def get_log_path(self, module: str, run_id: str | None = None) -> Path:
        """
        Returns the path for a log file under the project-specific output/logs/ directory.
        If run_id is provided, it appends it to the filename.
        """
        name = f"{module}.log" if not run_id else f"{module}_{run_id}.log"
        return self.logs_dir / name
            
    def get_chunks_path(self) -> Path:
        return self.root_dir / "input" / "chunks.tsv"
