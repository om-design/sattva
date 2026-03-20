"""Myelination substrate for GA-SATTVA.

Implements slow, activity-dependent conductance changes on a fixed
connectivity graph, gated by GA resonance.

- Fast dynamics are handled by GASATTVADynamics in ga_sattva_core.
- This module tracks co-activation and GA resonance over long periods
  and updates edge conductance accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


@dataclass
class MyelinationSubstrate:
    """Activity-dependent myelination layer.

    - Maintains a sparse connectivity graph between units.
    - Tracks per-edge conductance and usage.
    - Updates conductance slowly when units are co-active AND
      GA resonance is high.

    This is conceptually downstream of GA resonance: it does not define
    patterns, only reinforces highways that support them.
    """

    n_units: int
    local_radius: float = 0.15

    # Myelination parameters
    base_conductance: float = 1.0
    max_accretion: float = 2.0
    alpha_accrete: float = 2e-4    # very slow accretion
    alpha_decay: float = 1e-4      # slower decay
    alpha_usage: float = 1e-3      # usage EMA rate

    # GA resonance gating
    resonance_threshold: float = 0.3

    # Internal state
    connections: csr_matrix = field(init=False)
    conductance: csr_matrix = field(init=False)
    usage: csr_matrix = field(init=False)

    def __post_init__(self) -> None:
        self._build_local_connections()

    def _build_local_connections(self) -> None:
        """Build symmetric local connectivity based on geometric proximity.

        This uses unit indices only; the caller must supply positions
        if geometric distances are desired. For now, we assume indices
        are ordered so that nearby indices are roughly neighbors.
        """
        # Simple ring-like connectivity as a placeholder; can be
        # replaced by a call that uses actual positions.
        m = lil_matrix((self.n_units, self.n_units))
        for i in range(self.n_units):
            for j in range(max(0, i - 3), min(self.n_units, i + 4)):
                if i == j:
                    continue
                m[i, j] = 0.5
        self.connections = m.tocsr()
        self.conductance = csr_matrix((self.n_units, self.n_units))
        self.usage = csr_matrix((self.n_units, self.n_units))

    def effective_weights(self) -> csr_matrix:
        """Compute effective weights (base + myelination) per edge."""
        base = self.connections * self.base_conductance
        if self.conductance.nnz == 0:
            return base
        return base + self.connections.multiply(self.conductance)

    def step_myelination(
        self,
        activations: np.ndarray,
        ga_resonance: float,
        dt: float = 0.1,
    ) -> None:
        """Update myelination based on activity and GA resonance.

        Args:
            activations: current unit activations (shape: (n_units,))
            ga_resonance: scalar GA resonance signal from dynamics
            dt: timestep
        """
        from scipy.sparse import lil_matrix

        if ga_resonance < self.resonance_threshold:
            # Not enough coherent GA structure; only apply decay
            if self.conductance.nnz > 0:
                self.conductance = self.conductance.multiply(1.0 - self.alpha_decay * dt)
            return

        # Units considered active for myelination
        active_indices = np.where(activations > 0.25)[0]
        if len(active_indices) < 2:
            # No co-activation to drive myelination
            if self.conductance.nnz > 0:
                self.conductance = self.conductance.multiply(1.0 - self.alpha_decay * dt)
            return

        if not isinstance(self.conductance, lil_matrix):
            self.conductance = self.conductance.tolil()
            self.usage = self.usage.tolil()

        # Scale accretion by GA resonance (stronger resonance → faster myelination)
        alpha = self.alpha_accrete * float(np.clip(ga_resonance, 0.0, 1.0))

        for i in active_indices:
            row_start = max(0, i - 10)
            row_end = min(self.n_units, i + 11)
            for j in range(row_start, row_end):
                if i == j or self.connections[i, j] == 0:
                    continue

                co_signal = float(activations[i] * activations[j])
                if co_signal <= 0.0:
                    continue

                # Accrete conductance toward max_accretion
                current = float(self.conductance[i, j])
                target = float(self.max_accretion * np.tanh(100.0 * co_signal))
                new_val = current + dt * alpha * (target - current)
                if new_val > 0.0:
                    self.conductance[i, j] = float(min(new_val, self.max_accretion))

                # Update usage EMA
                u_current = float(self.usage[i, j])
                self.usage[i, j] = float(
                    (1.0 - self.alpha_usage) * u_current + self.alpha_usage * co_signal
                )

        # Passive decay
        self.conductance = self.conductance.tocsr().multiply(1.0 - self.alpha_decay * dt)
        self.usage = self.usage.tocsr()

    def snapshot(self) -> Dict[str, Any]:
        """Return summary statistics for analysis."""
        total_conductance = float(self.conductance.sum())
        n_myelinated = int(self.conductance.nnz)
        return {
            "total_conductance": total_conductance,
            "n_myelinated_edges": n_myelinated,
        }
