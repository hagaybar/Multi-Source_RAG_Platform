from .base import BaseEmbedder
from scripts.core.project_manager import ProjectManager
from scripts.utils.config_loader import ConfigLoader

_embedder_instance = None


def get_embedder(project: ProjectManager) -> BaseEmbedder:
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    cfg = project.config
    embedding_cfg = cfg.get("embedding", {})  # <-- proper nested access
    provider = embedding_cfg.get("provider", "local")

    if provider == "litellm":
        from .litellm_embedder import LiteLLMEmbedder

        _embedder_instance = LiteLLMEmbedder(
            endpoint=embedding_cfg.get("endpoint"),
            model=embedding_cfg.get("model"),
            api_key=embedding_cfg.get("api_key", None),
        )

    elif provider == "local":
        from .bge_embedder import BGEEmbedder

        _embedder_instance = BGEEmbedder(
            embedding_cfg.get("model_name", "BAAI/bge-large-en")
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
    return _embedder_instance
