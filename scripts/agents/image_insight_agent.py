from pathlib import Path
import base64
import uuid

from scripts.agents.base import AgentProtocol
from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.api_clients.openai.completer import OpenAICompleter
from scripts.utils.logger import LoggerManager


class ImageInsightAgent(AgentProtocol):
    
    def __init__(self, project: ProjectManager):
        self.project = project
        agent_cfg = project.config.get("agents", {})

        self.model_name = agent_cfg.get("image_agent_model", "gpt-4o")
        self.prompt_template = agent_cfg.get("image_prompt", self.default_prompt())
        self.output_mode = agent_cfg.get("output_mode", "append_to_chunk").lower()

        print(f"[DEBUG] Using image prompt:\n{self.prompt_template}")
        if not self.prompt_template:
            raise ValueError("ImageInsightAgent requires a valid prompt template.")
        print(f"[DEBUG] Using model: {self.model_name}")
        print(f"[DEBUG] Output mode: {self.output_mode}")
        


        self.logger = LoggerManager.get_logger(__name__)


    def run(self, chunk: Chunk, project: ProjectManager) -> list[Chunk]:
        image_path = chunk.meta.get("image_path")
        if not image_path:
            return [chunk]

        full_path = Path(project.root_dir) / image_path
        if not full_path.exists():
            self.logger.warning(f"ImageInsightAgent: file not found {full_path}")
            return [chunk]

        context = chunk.text[:500]
        encoded_image = self.encode_image(full_path)
        prompt = self.prompt_template.replace("{{ context }}", context)

        try:
            completer = OpenAICompleter(model_name=self.model_name)
            insight = completer.get_multimodal_completion(
                prompt=prompt, image_b64=encoded_image
            )
        except Exception as e:
            self.logger.error(f"Image insight generation failed: {e}")
            chunk.meta["image_summary_error"] = str(e)
            return [chunk]

        # Determine output behavior based on project config
        # cfg = project.config.get("agents", {}).get("image_insight", {})
        # mode = cfg.get("output_mode", "append_to_chunk").lower()

        if self.output_mode == "separate_chunk":
            image_chunk = Chunk(
                id=str(uuid.uuid4()),
                doc_id=chunk.doc_id,
                text=insight,
                token_count=len(insight.split()),
                meta={
                    "chunk_type": "image_insight",
                    "source_filepath": chunk.meta.get("source_filepath"),
                    "image_path": image_path,
                    "parent_chunk_id": chunk.id,
                }
            )
            return [chunk, image_chunk]

        # Default behavior: append to meta
        chunk.meta["image_summary"] = insight
        return [chunk]

    def encode_image(self, path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def default_prompt(self) -> str:
        return (
            "This is a screenshot extracted from a tutorial document.\n\n"
            "Surrounding Text:\n{{ context }}\n\n"
            "Based on the screenshot and the text, describe what this image shows, "
            "what step it illustrates, and why it is helpful."
        )
