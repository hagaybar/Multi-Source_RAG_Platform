from abc import ABC, abstractmethod
import numpy as np
from typing import List

class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Returns float32 numpy array of shape (n, dim)"""
        pass
