from pathlib import Path
import base64
import uuid

from scripts.agents.base import AgentProtocol
from scripts.chunking.models import Chunk, ImageChunk
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


    def run(self, chunk: Chunk, project: ProjectManager) -> list:
        image_paths = chunk.meta.get("image_paths", [])
        if not image_paths:
            return [chunk]

        image_chunks = []
        context = chunk.text[:500]
        prompt = self.prompt_template.replace("{{ context }}", context)

        for image_path in image_paths:
            # full_path = Path(project.root_dir) / image_path
            full_path = project.input_dir / image_path

            if not full_path.exists():
                self.logger.warning(f"Image file not found: {full_path}")
                continue

            try:
                encoded_image = self.encode_image(full_path)
                completer = OpenAICompleter(model_name=self.model_name)
                insight = completer.get_multimodal_completion(prompt=prompt, image_b64=encoded_image)
            except Exception as e:
                self.logger.error(f"Failed to enrich {image_path}: {e}")
                continue

            image_meta = {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "source_chunk_id": chunk.id,
                "doc_type": chunk.meta.get("doc_type"),
                "page_number": chunk.meta.get("page_number"),
                "source_filepath": chunk.meta.get("source_filepath"),
            }

            image_chunk = ImageChunk(
                id=str(uuid.uuid4()),
                description=insight,
                meta=image_meta
            )
            image_chunks.append(image_chunk)

        if self.output_mode == "separate_chunk":
            return [chunk] + image_chunks

        # Default: append summaries to chunk.meta
        chunk.meta["image_summaries"] = [
            {
                "image_path": ic.meta["image_path"],
                "description": ic.description
            }
            for ic in image_chunks
        ]
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
