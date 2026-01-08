from dataclasses import dataclass

import numpy as np


@dataclass
class HopfieldCore:
    """Simple continuous Hopfield-style attractor core.

    This is a first prototype for SATTVA's dynamical core: a small system
    whose states evolve under fixed weights and can settle into attractors.
    """

    W: np.ndarray  # weight matrix (dim x dim)

    @classmethod
    def from_patterns(cls, patterns: np.ndarray) -> "HopfieldCore":
        """Hebbian-style construction from a set of pattern vectors."""
        dim = patterns.shape[1]
        W = np.zeros((dim, dim))
        for p in patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0.0)
        W /= patterns.shape[0]
        return cls(W=W)

    def step(self, x: np.ndarray, beta: float = 1.5) -> np.ndarray:
        """One update step of the dynamics."""
        return np.tanh(beta * (self.W @ x))

    def run(self, x0: np.ndarray, steps: int = 15, beta: float = 1.5):
        """Run the dynamics for a fixed number of steps, returning the trajectory."""
        traj = [x0.copy()]
        x = x0.copy()
        for _ in range(steps):
            x = self.step(x, beta=beta)
            traj.append(x.copy())
        return traj
