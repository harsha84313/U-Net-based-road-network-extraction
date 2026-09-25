"""
Model wrapper layer to abstract model loading and inference.

This file provides a thin abstraction over existing model code so the rest
of the application can call `ModelManager.predict(...)` without depending
directly on PyTorch or a particular class.

Scalability notes / TODOs:
- Add device selection, model caching, and model versioning.
- Add an async batch inference API for high-throughput processing.
- Provide a REST/GRPC wrapper for model serving (TF-Serving, TorchServe, or custom FastAPI).
"""

import os
from typing import List
import numpy as np

# Default: reuse existing RoadExtractor for inference
try:
    from inference.predict import RoadExtractor
except Exception:
    RoadExtractor = None


class ModelManager:
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self):
        if self._model is None:
            if RoadExtractor is None:
                raise RuntimeError('RoadExtractor is not available')
            self._model = RoadExtractor(self.model_path or os.path.join('training','unet_road_extractor.pth'), device=self.device)
        return self._model

    def predict(self, image_np: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """Run a single image prediction and return binary mask (0/255).

        For production, replace this with a call to a served model endpoint.
        """
        model = self.load()
        return model.predict(image_np, threshold=threshold)

    def predict_batch(self, image_list: List[np.ndarray], batch_size: int = 8):
        """A placeholder batch predict method — currently runs sequentially.

        Replace with vectorized GPU batch processing or remote inferencing.
        """
        results = []
        for img in image_list:
            results.append(self.predict(img))
        return results
