# container.py
import os
import time
import pickle
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from sattva_engine_v9 import Engine, ProgramEmbedding  # same directory

@dataclass
class SattvaContainer:
    engine: Engine
    embedding: ProgramEmbedding
    meta: Dict[str, Any]

    @classmethod
    def new_infant(cls, rng_seed: int = 42) -> "SattvaContainer":
        rng = np.random.default_rng(rng_seed)
        space = ProgramEmbedding(dim_base=4, dim_instr=4, rng=rng)
        eng = Engine(dim=space.dim, base_activation_threshold=0.1)
        meta: Dict[str, Any] = {
            "engine_version": "v9.0.0",
            "created_at": time.time(),
            "step": 0,
            "curriculum_log": [],
            "epiphany_log": [],
        }
        return cls(engine=eng, embedding=space, meta=meta)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SattvaContainer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    def log_epiphany(
        self,
        phase: str,
        input_id: str,
        epiphanies,
        tension: float,
        mean_novelty: float,
        triage: float,
    ) -> None:
        event = {
            "phase": phase,
            "input_id": input_id,
            "tension": tension,
            "mean_novelty": mean_novelty,
            "triage": triage,
            "epiphanies": [
                {
                    "ancestor": cid,
                    "contributors": contributors,
                    "vote": vote,
                    "depth": depth,
                }
                for (cid, contributors, vote, depth) in epiphanies
            ],
            "timestamp": time.time(),
        }
        self.meta.setdefault("epiphany_log", []).append(event)
