from pathlib import Path
from scripts.utils.config_loader import ConfigLoader
from werkzeug.utils import secure_filename
from typing import List, Tuple, Dict, Any


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
        for path in [
            self.input_dir,
            self.output_dir,
            self.logs_dir,
            self.faiss_dir,
            self.metadata_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def get_input_dir(self) -> Path:
        return self.input_dir

    def raw_docs_dir(self) -> Path:
        """
        Directory where raw input files live (used by validation_helpers).
        """
        return self.input_dir / "raw"

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
    def create_project(
        project_name: str,
        project_description: str,
        language: str,
        image_enrichment: bool,
        embedding_model: str,
        projects_base_dir: Path,
    ):
        """
        Creates a new project directory and a default config.yml file.
        """
        project_name = secure_filename(project_name)
        project_root = projects_base_dir / project_name
        if project_root.exists():
            raise FileExistsError(f"Project '{project_name}' already exists.")

        # Define the default config template FIRST
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
                "raw_dir": "raw",
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

        # Now we can safely use the config to get paths
        paths_cfg = default_config["paths"]

        input_dir = project_root / paths_cfg.get("input_dir", "input")
        raw_dir = input_dir / "raw"  # hardcoded since it's not in paths config
        output_dir = project_root / paths_cfg.get("output_dir", "output")
        logs_dir = project_root / paths_cfg.get("logs_dir", "output/logs")
        faiss_dir = project_root / paths_cfg.get("faiss_dir", "output/faiss")
        metadata_dir = project_root / paths_cfg.get("metadata_dir", "output/metadata")

        # Create necessary directories (including input_dir and output_dir as parents)
        for path in [input_dir, raw_dir, output_dir, logs_dir, faiss_dir, metadata_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Write the config file
        config_path = project_root / "config.yml"
        with config_path.open("w", encoding="utf-8") as f:
            import yaml

            yaml.dump(default_config, f, default_flow_style=False)

        return project_root

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        """
        Returns the expected configuration schema.
        This defines what a valid config should look like.
        """
        return {
            "project": {
                "name": str,
                "description": str,
                "language": str,
            },
            "paths": {
                "input_dir": str,
                "output_dir": str,
                "logs_dir": str,
                "faiss_dir": str,
                "metadata_dir": str,
            },
            "embedding": {
                "skip_duplicates": bool,
                "provider": str,
                "endpoint": str,
                "mode": str,
                "model": str,
                "embed_batch_size": int,
                "use_async_batch": bool,
                "image_enrichment": bool,
            },
            "llm": {
                "provider": str,
                "model": str,
                "temperature": float,
                "max_tokens": int,
                "prompt_strategy": str,
            },
            "agents": {
                "enable_image_insight": bool,
                "image_agent_model": str,
                "output_mode": str,
                "image_prompt": str,
            },
        }

    @staticmethod
    def validate_config(config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Basic validation of configuration structure.
        Returns (is_valid, list_of_errors)
        """
        errors = []

        # Check required sections exist
        required_sections = ["project", "paths", "embedding", "llm", "agents"]
        for section in required_sections:
            if section not in config_data:
                errors.append(f"Missing required section: '{section}'")
            elif not isinstance(config_data[section], dict):
                errors.append(f"Section '{section}' must be a dictionary")

        # Check critical fields exist
        critical_fields = [
            ("project", "name"),
            ("project", "language"),
            ("embedding", "model"),
        ]

        for section, field in critical_fields:
            if section in config_data and field not in config_data[section]:
                errors.append(f"Missing critical field: '{section}.{field}'")

        return len(errors) == 0, errors

    @staticmethod
    def validate_config_file(config_path: Path) -> Tuple[bool, List[str]]:
        """
        Validates a config.yml file.
        Returns (is_valid, list_of_errors)
        """
        import yaml

        try:
            if not config_path.exists():
                return False, [f"Config file not found: {config_path}"]

            with config_path.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                return False, ["Config file is empty"]

            return ProjectManager.validate_config(config_data)

        except yaml.YAMLError as e:
            return False, [f"Invalid YAML: {e}"]
        except Exception as e:
            return False, [f"Error reading config: {e}"]
