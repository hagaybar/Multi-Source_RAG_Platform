from .base import BaseEmbedder
from scripts.core.project_manager import ProjectManager
from scripts.utils.config_loader import ConfigLoader

_embedder_instance = None

def get_embedder(project: ProjectManager) -> BaseEmbedder:
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    cfg = project.config
    provider = cfg.get("embedding.provider", "local")

    if provider == "litellm":
        from .litellm_embedder import LiteLLMEmbedder
        _embedder_instance = LiteLLMEmbedder(
            endpoint=cfg.get("embedding.endpoint"),
            model=cfg.get("embedding.model"),
            api_key=cfg.get("embedding.api_key", None)
        )
    elif provider == "local":
        from .bge_embedder import BGEEmbedder
        _embedder_instance = BGEEmbedder(cfg.get("embedding.model_name", "BAAI/bge-large-en"))
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    return _embedder_instance
