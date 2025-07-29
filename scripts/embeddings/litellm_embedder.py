import os
import requests
import numpy as np
from typing import List
from scripts.embeddings.base import BaseEmbedder


class LiteLLMEmbedder(BaseEmbedder):
    """
    Embedding client using LiteLLM-compatible API (e.g., OpenAI, Ollama, Together.ai).
    """

    def __init__(self, endpoint: str, model: str, api_key: str = None):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key or os.getenv("OPEN_AI")  # fallback to env var

    def encode(self, texts: List[str]) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {"model": self.model, "input": texts}

        response = requests.post(self.endpoint, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings, dtype=np.float32)
