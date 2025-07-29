from scripts.chunking.models import Chunk
from scripts.core.project_manager import ProjectManager
from abc import ABC, abstractmethod


class AgentProtocol(ABC):
    @abstractmethod
    def run(self, chunk: Chunk, project: ProjectManager) -> Chunk:
        """Enrich or modify a single chunk."""
        pass
