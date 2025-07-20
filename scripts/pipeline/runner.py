from typing import Callable, Iterator
from pathlib import Path

from scripts.core.project_manager import ProjectManager
from scripts.utils.logger import LoggerManager


class PipelineRunner:
    """
    Orchestrates sequential execution of modular pipeline steps (ingest, chunk, enrich, embed, index).
    """

    def __init__(self, project: ProjectManager, config: dict):
        self.project = project
        self.config = config
        self.steps: list[tuple[str, dict]] = []
        self.logger = LoggerManager.get_logger("PipelineRunner", log_file=project.get_log_path("pipeline"))

    def add_step(self, name: str, **kwargs) -> None:
        """
        Adds a step by name, with optional keyword arguments.
        Steps must match a method named `step_<name>`.
        """
        if not hasattr(self, f"step_{name}"):
            raise ValueError(f"Step '{name}' not implemented.")
        self.steps.append((name, kwargs))
        self.logger.info(f"Step added: {name} {kwargs}")

    def clear_steps(self) -> None:
        self.steps.clear()
        self.logger.info("All steps cleared from pipeline.")

    def run_steps(self) -> Iterator[str]:
        """
        Runs all configured steps in order. Yields status messages for UI or CLI.
        """
        self.logger.info("Running pipeline steps...")
        yield "🚀 Starting pipeline execution..."

        for name, kwargs in self.steps:
            step_fn: Callable = getattr(self, f"step_{name}", None)
            yield f"▶️ Running step: {name}"
            self.logger.info(f"Running step: {name} with args: {kwargs}")

            try:
                result = step_fn(**kwargs)
                if isinstance(result, Iterator):
                    for msg in result:
                        yield msg
                else:
                    yield f"✅ Step '{name}' completed."
                self.logger.info(f"Step '{name}' completed.")
            except Exception as e:
                self.logger.error(f"Step '{name}' failed: {e}", exc_info=True)
                yield f"❌ Step '{name}' failed: {e}"
                raise

        yield "🏁 Pipeline finished."

    # ----------------------------
    # Step stubs (to be implemented next)
    # ----------------------------
    def step_ingest(self, **kwargs) -> Iterator[str]:
        yield "[ingest] not implemented yet."

    def step_chunk(self, **kwargs) -> Iterator[str]:
        yield "[chunk] not implemented yet."

    def step_enrich(self, **kwargs) -> Iterator[str]:
        yield "[enrich] not implemented yet."

    def step_embed(self, **kwargs) -> Iterator[str]:
        yield "[embed] not implemented yet."

    def step_index(self, **kwargs) -> Iterator[str]:
        yield "[index] not implemented yet."

    def step_retrieve(self, **kwargs) -> Iterator[str]:
        yield "[retrieve] not implemented yet."

    def step_ask(self, **kwargs) -> Iterator[str]:
        yield "[ask] not implemented yet."
