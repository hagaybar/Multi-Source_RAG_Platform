from sentence_transformers import SentenceTransformer
import numpy as np
from .base import BaseEmbedder


class BGEEmbedder(BaseEmbedder):
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
