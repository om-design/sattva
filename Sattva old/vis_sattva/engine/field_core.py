# vis_sattva/engine/field_core.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .config import FieldConfig


@dataclass
class FieldState:
    """
    Minimal latent field for vis_sattva.

    - a: activations of N units
    - W: synaptic weights (N x N)
    - M: myelination levels (N x N, in [0,1])
    - U: usage EMA for each synapse (N x N, in [0,1])
    """

    a: np.ndarray               # shape (N,)
    W: np.ndarray               # shape (N, N)
    M: np.ndarray               # shape (N, N)
    U: np.ndarray               # shape (N, N)
    energy_history: list[float] = field(default_factory=list)
    tension_history: list[float] = field(default_factory=list)

    @classmethod
    def init_random(cls, n_units: int, rng: Optional[np.random.Generator] = None) -> "FieldState":
        if rng is None:
            rng = np.random.default_rng()
        a = np.zeros(n_units, dtype=np.float32)
        # Small random initial weights
        W = rng.normal(loc=0.0, scale=0.01, size=(n_units, n_units)).astype(np.float32)
        M = np.zeros((n_units, n_units), dtype=np.float32)
        U = np.zeros((n_units, n_units), dtype=np.float32)
        return cls(a=a, W=W, M=M, U=U)

    def step(self, input_vec: np.ndarray, cfg: FieldConfig) -> Tuple[float, float]:
        """
        Single update step given an external input vector of shape (N,).

        Returns:
            (energy, tension) after the update.
        """
        assert input_vec.shape == self.a.shape

        # --- Core dynamics ---
        # Raw update: recurrent + input - damping
        h = self.W @ self.a + input_vec - cfg.damping * self.a

        # Nonlinearity
        a_new = np.tanh(h)

        # Energy constraint
        energy = float(np.sum(a_new * a_new))
        if energy > cfg.energy_max and energy > 0.0:
            scale = (cfg.energy_max / energy) ** 0.5
            a_new *= scale
            energy = float(np.sum(a_new * a_new))  # recompute after scaling

        # Compute "tension" as L1 norm of input (can refine later)
        tension = float(np.sum(np.abs(input_vec)))

        # Commit activations
        self.a = a_new

        # --- Synapse plasticity ---
        self._update_synapses(cfg=cfg, tension=tension)

        # --- Logging ---
        if cfg.log_energy:
            self.energy_history.append(energy)
        if cfg.log_tension:
            self.tension_history.append(tension)

        return energy, tension

    def _update_synapses(self, cfg: FieldConfig, tension: float) -> None:
        """
        Update W, U, M given current activations self.a.

        - Usage EMA U_ij tracks |a_i * a_j|
        - Hebbian: W_ij += lr * a_i * a_j
        - Myelination M_ij moves slowly toward U_ij
        - Decay: W_ij *= (1 - decay_eps * (1 - M_ij))
        - Trauma: if enabled and tension high, temporarily boost Hebbian
                  and enforce a minimum myelin value on active synapses.
        """
        a = self.a
        # Outer product of activations gives co-activation for all pairs
        co_use = np.abs(np.outer(a, a)).astype(np.float32)

        # Usage EMA
        self.U = (1.0 - cfg.ema_alpha) * self.U + cfg.ema_alpha * co_use

        # Determine Hebbian learning rate (trauma-modulated)
        lr = cfg.lr_hebb
        trauma_active = False
        if cfg.trauma_mode and tension > cfg.trauma_threshold:
            trauma_active = True
            lr = cfg.lr_hebb * cfg.trauma_lr_boost

        # Hebbian update
        self.W += lr * np.outer(a, a).astype(np.float32)

        # Myelination drifts toward usage
        self.M += cfg.myelin_beta * (self.U - self.M)
        np.clip(self.M, 0.0, 1.0, out=self.M)

        # Gentle decay of weak, unmyelinated synapses
        if cfg.decay_eps > 0.0:
            decay_factor = 1.0 - cfg.decay_eps * (1.0 - self.M)
            self.W *= decay_factor.astype(np.float32)

        # Trauma-specific myelination floor on currently active synapses
        if trauma_active and cfg.trauma_min_myelin > 0.0:
            # Apply minimum myelin only where co_use is non-trivial
            mask = co_use > 0.0
            self.M[mask] = np.maximum(self.M[mask], cfg.trauma_min_myelin)
