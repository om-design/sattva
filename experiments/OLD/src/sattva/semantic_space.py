from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SemanticSpace:
    """Labeled semantic vectors in a shared space.

    For now this is a very small, hand-crafted toy space. Later this can
    be backed by real embeddings.
    """

    vectors: Dict[str, np.ndarray]

    @classmethod
    def toy_animals_and_objects(cls) -> "SemanticSpace":
        # 3D for easy inspection; later this can be real embeddings.
        vecs = {
            "cat": np.array([1.0, 0.8, 0.0]),
            "dog": np.array([0.9, 1.0, 0.0]),
            "wolf": np.array([0.8, 1.0, 0.2]),
            "car": np.array([0.0, 0.1, 1.0]),
        }
        return cls(vectors=vecs)

    def names(self):
        return list(self.vectors.keys())

    def as_matrix(self):
        """Return matrix of shape (n_items, dim) and matching name list."""
        names = self.names()
        mat = np.stack([self.vectors[n] for n in names], axis=0)
        return names, mat
