from pathlib import Path
from scripts.utils.config_loader import ConfigLoader
from werkzeug.utils import secure_filename

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

    @staticmethod
    def create_project(project_name: str, project_description: str, language: str, image_enrichment: bool, embedding_model: str, projects_base_dir: Path):
        """
        Creates a new project directory and a default config.yml file.
        """
        project_name = secure_filename(project_name)
        project_root = projects_base_dir / project_name
        if project_root.exists():
            raise FileExistsError(f"Project '{project_name}' already exists.")

        # Create project directories
        input_raw_dir = project_root / "input" / "raw"
        output_dir = project_root / "output"
        logs_dir = output_dir / "logs"
        faiss_dir = output_dir / "faiss"
        metadata_dir = output_dir / "metadata"

        for path in [input_raw_dir, logs_dir, faiss_dir, metadata_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Create a default config.yml
        default_config = {
            "project": {
                "name": project_name,
                "description": project_description,
                "language": language,
            },
            "paths": {
                "input_dir": "input",
                "output_dir": "output",
                "logs_dir": "output/logs",
                "faiss_dir": "output/faiss",
                "metadata_dir": "output/metadata",
            },
            "embedding": {
                "skip_duplicates": True,
                "provider": "litellm",
                "endpoint": "https://api.openai.com/v1/embeddings",
                "mode": "batch",
                "model": embedding_model,
                "embed_batch_size": 64,
                "use_async_batch": True,
                "image_enrichment": image_enrichment,
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.4,
                "max_tokens": 400,
                "prompt_strategy": "default",
            },
            "agents": {
                "enable_image_insight": image_enrichment,
                "image_agent_model": "gpt-4o",
                "output_mode": "append_to_chunk",
                "image_prompt": (
                    "This is a screenshot from a tutorial document.\n\n"
                    "Surrounding Text:\n{{ context }}\n\n"
                    "Based on the screenshot and the surrounding text, describe what this image shows, "
                    "what step it illustrates, and why it is helpful in the tutorial."
                ),
            },
        }

        config_path = project_root / "config.yml"
        with config_path.open("w", encoding="utf-8") as f:
            import yaml
            yaml.dump(default_config, f, default_flow_style=False)

        return project_root
