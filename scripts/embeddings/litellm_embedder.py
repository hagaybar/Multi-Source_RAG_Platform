import numpy as np
import requests
from .base import BaseEmbedder

class LiteLLMEmbedder(BaseEmbedder):
    def __init__(self, endpoint, model, api_key=None):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key

    def encode(self, texts):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None
        }
        body = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(self.endpoint, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings, dtype=np.float32)
