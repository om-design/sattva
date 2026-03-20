# ga_encoding.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np

@dataclass
class GAEncoding:
    """
    GA-inspired encoding: represent PROGRAM = BASE ⊗ INSTRUCTION
    as a vector in R^dim, split into (base_subspace || instr_subspace).
    """
    dim_base: int
    dim_instr: int

    def __post_init__(self) -> None:
        self.dim = self.dim_base + self.dim_instr
        self._base_cache: Dict[str, np.ndarray] = {}
        self._instr_cache: Dict[str, np.ndarray] = {}

    # --- internal helpers ---

    def _hash_to_rng(self, key: str) -> np.random.Generator:
        seed = abs(hash(key)) % (2**32)
        return np.random.default_rng(seed)

    def _unit_vector(self, dim: int, key: str) -> np.ndarray:
        rng = self._hash_to_rng(key)
        v = rng.standard_normal(dim)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # --- public API ---

    def base_vector(self, base_id: str) -> np.ndarray:
        if base_id not in self._base_cache:
            self._base_cache[base_id] = self._unit_vector(self.dim_base, f"BASE::{base_id}")
        return self._base_cache[base_id]

    def instr_vector(self, instr_id: str) -> np.ndarray:
        if instr_id not in self._instr_cache:
            self._instr_cache[instr_id] = self._unit_vector(self.dim_instr, f"INSTR::{instr_id}")
        return self._instr_cache[instr_id]

    def encode_program(
        self,
        base_ids: Iterable[str],
        instr_ids: Iterable[str] = (),
        base_weights: Optional[Iterable[float]] = None,
        instr_weights: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        base_ids = list(base_ids)
        instr_ids = list(instr_ids)
        if not base_ids:
            raise ValueError("encode_program requires at least one base id")

        if base_weights is None:
            base_weights = [1.0] * len(base_ids)
        if instr_weights is None:
            instr_weights = [1.0] * len(instr_ids) if instr_ids else []

        base_weights = list(base_weights)
        instr_weights = list(instr_weights)

        # Base part
        b_vecs = [w * self.base_vector(bid) for bid, w in zip(base_ids, base_weights)]
        base_part = np.sum(b_vecs, axis=0)
        nb = np.linalg.norm(base_part)
        if nb > 0:
            base_part = base_part / nb

        # Instruction part
        if instr_ids:
            i_vecs = [w * self.instr_vector(iid) for iid, w in zip(instr_ids, instr_weights)]
            instr_part = np.sum(i_vecs, axis=0)
            ni = np.linalg.norm(instr_part)
            if ni > 0:
                instr_part = instr_part / ni
        else:
            instr_part = np.zeros(self.dim_instr, dtype=float)

        return np.concatenate([base_part, instr_part])

    def split_embedding(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if emb.shape[0] != self.dim:
            raise ValueError("Embedding dimension mismatch")
        return emb[: self.dim_base], emb[self.dim_base :]
