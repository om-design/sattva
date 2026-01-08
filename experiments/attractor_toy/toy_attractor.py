"""Toy semantic attractor experiment.

This is a small end-to-end wiring of the current SATTVA primitives:
- SemanticSpace: labeled semantic vectors
- HopfieldCore: a simple continuous attractor dynamic over vectors

The goal is not "good performance" yet, but a clear place where information
flows:

    semantic cue -> dynamical core -> settled state -> interpretation

Run:
    python experiments/attractor_toy/toy_attractor.py

Notes:
- This uses hand-crafted vectors for now.
- The continuous Hopfield-style core can collapse to near-zero depending on
  hyperparameters; that's a useful diagnostic, not a failure.
"""

import numpy as np

from sattva.attractor_core import HopfieldCore
from sattva.semantic_space import SemanticSpace


def closest_pattern(final_state: np.ndarray, names: list[str], patterns: np.ndarray):
    dists = np.linalg.norm(patterns - final_state[None, :], axis=1)
    idx = int(np.argmin(dists))
    return names[idx], float(dists[idx])


def run_demo(seed: int = 0, noise: float = 0.3, steps: int = 15, beta: float = 1.5):
    space = SemanticSpace.toy_animals_and_objects()
    names, patterns = space.as_matrix()

    core = HopfieldCore.from_patterns(patterns)

    rng = np.random.default_rng(seed)
    start = space.vectors["cat"] + noise * rng.normal(size=patterns.shape[1])

    traj = core.run(start, steps=steps, beta=beta)
    final = traj[-1]

    winner, dist = closest_pattern(final, names, patterns)

    print("Start:", start)
    print("Final state:", final)
    print("Winner:", winner, "distance", f"{dist:.3f}")
    print("Known patterns:")
    for name, vec in space.vectors.items():
        d = np.linalg.norm(final - vec)
        print(f"  {name:>4}: distance {d:.3f}")


if __name__ == "__main__":
    run_demo()
