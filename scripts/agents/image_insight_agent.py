from pathlib import Path
import base64

from scripts.agents.base import AgentProtocol
from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from scripts.api_clients.openai.completer import OpenAICompleter
from scripts.utils.logger import LoggerManager


class ImageInsightAgent(AgentProtocol):
    def __init__(self, model_name: str = "gpt-4o", prompt_template: str | None = None):
        self.model_name = model_name
        self.prompt_template = prompt_template or self.default_prompt()
        self.logger = LoggerManager.get_logger(__name__)

    def run(self, chunk: Chunk, project: ProjectManager) -> Chunk:
        image_path = chunk.meta.get("image_path")
        if not image_path:
            return chunk

        context = chunk.text[:500]
        full_path = Path(project.root_dir) / image_path
        if not full_path.exists():
            self.logger.warning(f"ImageInsightAgent: file not found {full_path}")
            return chunk

        encoded_image = self.encode_image(full_path)
        prompt = self.prompt_template.replace("{{ context }}", context)

        try:
            completer = OpenAICompleter(model_name=self.model_name)
            insight = completer.get_multimodal_completion(
                prompt=prompt, image_b64=encoded_image
            )
            if insight:
                chunk.meta["image_summary"] = insight
        except Exception as e:
            self.logger.error(f"Image insight generation failed: {e}")
            chunk.meta["image_summary_error"] = str(e)

        return chunk

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
