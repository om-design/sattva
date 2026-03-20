"""GA-SATTVA core: patterns as geometric algebra multivectors.

This module defines a minimal substrate where:
- Each unit has a GA multivector representation and a scalar activation.
- Patterns are multivectors built from subsets of units.
- Resonance is GA overlap between current and stored patterns.

No myelination or infrastructure yet; this is the clean core for
"expert intuition" experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np


# ---- GA utilities (placeholder: Euclidean inner product in R^16) ----

MV_DIM = 16  # e.g., G(3,0,1) multivectors flattened to 16D


def normalize_mv(mv: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(mv))
    if norm < 1e-8:
        return np.zeros_like(mv)
    return mv / norm


def mv_inner(a: np.ndarray, b: np.ndarray) -> float:
    """Inner product between two multivectors.

    Placeholder: plain dot product in R^16. Later, this can be replaced
    by a proper GA library implementing the Clifford inner product.
    """
    return float(np.dot(a, b))


# ---- Core data structures ----

@dataclass
class GAUnitSet:
    """Collection of units with GA multivectors and scalar activations."""

    n_units: int
    mv_dim: int = MV_DIM

    mv: np.ndarray = field(init=False)         # (n_units, mv_dim)
    activations: np.ndarray = field(init=False)  # (n_units,)

    def __post_init__(self) -> None:
        self.mv = np.zeros((self.n_units, self.mv_dim), dtype=np.float32)
        self.activations = np.zeros(self.n_units, dtype=np.float32)

    def reset_activations(self, value: float = 0.0) -> None:
        self.activations.fill(float(value))

    def random_initialize_multivectors(self, seed: int | None = None) -> None:
        """Assign random multivectors and normalize them.

        This can later be replaced by embedding-based initialization.
        """
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(self.n_units, self.mv_dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        self.mv = (raw / norms).astype(np.float32)


@dataclass
class GAPattern:
    """A pattern represented as a GA multivector.

    - active_units: indices of units participating in the pattern
    - multivector: canonical pattern multivector in R^mv_dim
    - mean_activation: average activation of participating units
    """

    active_units: np.ndarray
    multivector: np.ndarray
    mean_activation: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "GAPattern") -> float:
        """Similarity via normalized inner product in GA space."""
        a = normalize_mv(self.multivector)
        b = normalize_mv(other.multivector)
        if a.sum() == 0.0 or b.sum() == 0.0:
            return 0.0
        sim = mv_inner(a, b)
        # Clip to [0, 1] for interpretability
        return float(max(0.0, min(1.0, sim)))

    def resonance_strength(self, other: "GAPattern") -> float:
        """Resonance = GA similarity * geometric mean of strengths."""
        sim = self.similarity(other)
        strength = (self.mean_activation * other.mean_activation) ** 0.5
        return float(sim * strength)


def pattern_from_units(units: GAUnitSet, threshold: float = 0.1) -> GAPattern:
    """Build a GA pattern from currently active units.

    Args:
        units: GAUnitSet
        threshold: minimum activation to include a unit
    """
    active = np.where(units.activations > threshold)[0]
    if len(active) == 0:
        return GAPattern(
            active_units=np.array([], dtype=int),
            multivector=np.zeros(units.mv_dim, dtype=np.float32),
            mean_activation=0.0,
            meta={"empty": True},
        )

    acts = units.activations[active]
    mvs = units.mv[active]  # (n_active, mv_dim)
    weighted = (acts[:, None] * mvs).sum(axis=0)
    mv = weighted / (acts.sum() + 1e-8)

    return GAPattern(
        active_units=active,
        multivector=mv.astype(np.float32),
        mean_activation=float(acts.mean()),
        meta={"empty": False},
    )


# ---- Minimal GA-SATTVA dynamics ----

@dataclass
class GASATTVADynamics:
    """Minimal GA-SATTVA dynamics.

    - units: GAUnitSet
    - stored_patterns: expert library (GA patterns)
    - gamma: restoration rate
    - u_rest: resting activation
    - ga_coupling: strength of GA resonance drive
    """

    units: GAUnitSet
    stored_patterns: List[GAPattern] = field(default_factory=list)
    gamma: float = 1.5
    u_rest: float = 0.1
    ga_coupling: float = 0.2

    # Logs
    resonance_history: List[float] = field(default_factory=list)
    mean_history: List[float] = field(default_factory=list)
    max_history: List[float] = field(default_factory=list)

    def step(self, dt: float = 0.1) -> Dict[str, float]:
        """Single Euler step of dynamics, driven by GA resonance only.

        This is intentionally minimal: no local connectivity or field,
        just restoration plus a global GA-driven term.
        """
        a = self.units.activations

        # Basic restoration toward resting level
        da_rest = -self.gamma * (a - self.u_rest)

        # GA resonance term: aggregate overlap with stored patterns
        current_pattern = pattern_from_units(self.units, threshold=0.1)
        total_resonance = 0.0
        if self.stored_patterns:
            for p in self.stored_patterns:
                if p.meta.get("empty"):
                    continue
                total_resonance += current_pattern.resonance_strength(p)

        # For now, apply GA drive uniformly to all units proportional to
        # total resonance. Later, this can be shaped by which units
        # participate most in resonant patterns.
        da_ga = self.ga_coupling * total_resonance

        # Update activations
        a_new = a + dt * (da_rest + da_ga)
        self.units.activations = a_new.astype(np.float32)

        # Logging
        mean_val = float(self.units.activations.mean())
        max_val = float(self.units.activations.max())
        self.mean_history.append(mean_val)
        self.max_history.append(max_val)
        self.resonance_history.append(float(total_resonance))

        return {
            "mean": mean_val,
            "max": max_val,
            "resonance": float(total_resonance),
        }


# ---- Primitive creation for experiments ----

def create_ga_primitive(units: GAUnitSet, center_indices: np.ndarray) -> GAPattern:
    """Create a GA primitive from a chosen subset of units.

    This is the GA-first version of a "primitive": explicitly select
    which units define the concept, set a clean activation plateau, and
    compute a canonical multivector pattern.
    """
    if len(center_indices) == 0:
        return GAPattern(
            active_units=np.array([], dtype=int),
            multivector=np.zeros(units.mv_dim, dtype=np.float32),
            mean_activation=0.0,
            meta={"empty": True},
        )

    units.reset_activations(0.0)
    units.activations[center_indices] = 0.5

    acts = units.activations[center_indices]
    mvs = units.mv[center_indices]
    weighted = (acts[:, None] * mvs).sum(axis=0)
    mv = weighted / (acts.sum() + 1e-8)

    return GAPattern(
        active_units=center_indices.astype(int),
        multivector=mv.astype(np.float32),
        mean_activation=float(acts.mean()),
        meta={"empty": False},
    )
