#!/usr/bin/env python3
"""Scenario 1: GA-based learning (habituation to a pattern).

Goal: show that repeated exposure to one GA primitive changes the
expert library so that, after training, weak cues preferentially
activate that primitive (higher GA resonance) compared to untrained
ones.

Phases:
- Phase 0: Build expert library (GA primitives).
- Phase 1: Baseline test with weak cues to all primitives.
- Phase 2: Training: repeatedly expose system to one "training" primitive.
- Phase 3: Post-training test with the same weak cues.

All dynamics use the GA-SATTVA core.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Repo root
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from sattva.ga_sattva_core import (
    GAUnitSet,
    GASATTVADynamics,
    create_ga_primitive,
    GAPattern,
    pattern_from_units,
)


def build_expert_library(units: GAUnitSet,
                         n_primitives: int = 5,
                         primitive_size: int = 24,
                         seed: int = 123) -> list[GAPattern]:
    rng = np.random.default_rng(seed)
    indices = np.arange(units.n_units)
    rng.shuffle(indices)

    primitive_indices = [
        indices[i * primitive_size:(i + 1) * primitive_size]
        for i in range(n_primitives)
    ]

    stored_patterns: list[GAPattern] = []
    for idx in primitive_indices:
        pattern = create_ga_primitive(units, idx)
        stored_patterns.append(pattern)

    return stored_patterns


def run_episode(units: GAUnitSet,
                dynamics: GASATTVADynamics,
                cue_indices: np.ndarray,
                cue_level: float,
                n_steps: int,
                dt: float = 0.1) -> float:
    """Run one episode with a given cue and return peak GA resonance."""
    units.reset_activations(0.0)
    units.activations[cue_indices] = cue_level

    peak_resonance = 0.0
    for _ in range(n_steps):
        info = dynamics.step(dt=dt)
        peak_resonance = max(peak_resonance, info["resonance"])
    return peak_resonance


def update_pattern(pattern: GAPattern,
                   observed: GAPattern,
                   eta: float = 5e-4,
                   sim_threshold: float = 0.9) -> GAPattern:
    """Update a stored GA primitive toward an observed pattern.

    Only update when GA similarity is high enough, mimicking slow,
    trust-based consolidation of strongly resonant experiences.
    """
    if observed.meta.get("empty"):
        return pattern

    sim = pattern.similarity(observed)
    if sim < sim_threshold:
        return pattern

    mv_old = pattern.multivector
    mv_new = (1.0 - eta) * mv_old + eta * observed.multivector
    # Renormalize
    norm = float(np.linalg.norm(mv_new)) + 1e-8
    mv_new = mv_new / norm

    return GAPattern(
        active_units=pattern.active_units,
        multivector=mv_new.astype(np.float32),
        mean_activation=pattern.mean_activation,
        meta=pattern.meta,
    )


def main() -> None:
    print("=" * 70)
    print("GA-SATTVA: Learning via GA primitive adaptation")
    print("=" * 70)

    # Initialize units and GA core
    n_units = 512
    units = GAUnitSet(n_units=n_units)
    units.random_initialize_multivectors(seed=42)

    print("Initialized GA units:")
    print(f"  n_units: {n_units}")
    print(f"  mv_dim:  {units.mv_dim}")

    # Phase 0: build expert library
    print("\n" + "=" * 70)
    print("PHASE 0: BUILDING EXPERT LIBRARY (GA primitives)")
    print("=" * 70)

    stored_patterns = build_expert_library(units)

    for k, p in enumerate(stored_patterns):
        print(f"  Primitive {k+1}: {len(p.active_units):3d} units, "
              f"mean_activation={p.mean_activation:.3f}")

    print(f"\n✓ Expert library: {len(stored_patterns)} GA primitives ready\n")

    # Dynamics
    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=stored_patterns,
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    # Helper: weak cue test for all primitives
    def weak_cue_test(label: str) -> list[float]:
        print("=" * 70)
        print(label)
        print("=" * 70)
        results: list[float] = []
        for k, p in enumerate(stored_patterns):
            # Cue: small subset of primitive units at low activation
            cue = p.active_units[: len(p.active_units) // 2]
            peak_res = run_episode(
                units,
                dynamics,
                cue_indices=cue,
                cue_level=0.3,
                n_steps=40,
            )
            results.append(peak_res)
            print(f"  Primitive {k+1}: peak_resonance={peak_res:.6f}")
        print()
        return results

    # Phase 1: baseline weak-cue test
    baseline_results = weak_cue_test("PHASE 1: BASELINE WEAK-CUE TEST")

    # Phase 2: training on primitive 1
    print("=" * 70)
    print("PHASE 2: TRAINING ON PRIMITIVE 1")
    print("=" * 70)

    train_index = 0
    train_pattern = stored_patterns[train_index]

    n_episodes = 200
    episode_length = 40
    eta = 5e-4

    for ep in range(n_episodes):
        # Full cue of training primitive at moderate level
        units.reset_activations(0.0)
        units.activations[train_pattern.active_units] = 0.5

        # Run episode
        for _ in range(episode_length):
            dynamics.step(dt=0.1)

        # Observe pattern after episode
        observed = pattern_from_units(units, threshold=0.1)
        # Update stored primitive toward observed pattern (conditionally)
        updated = update_pattern(train_pattern, observed, eta=eta, sim_threshold=0.9)
        stored_patterns[train_index] = updated
        train_pattern = updated

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: updated primitive 1")

    print("\n✓ Training complete\n")

    # Phase 3: post-training weak-cue test
    post_results = weak_cue_test("PHASE 3: POST-TRAINING WEAK-CUE TEST")

    # Analysis: compare baseline vs post-training
    baseline = np.array(baseline_results)
    post = np.array(post_results)

    print("=" * 70)
    print("ANALYSIS: GA Learning Effect")
    print("=" * 70)

    for k in range(len(stored_patterns)):
        print(
            f"Primitive {k+1}: baseline={baseline[k]:.6f}, "
            f"post={post[k]:.6f}, "
            f"delta={post[k] - baseline[k]:+.6f}",
            ("<- TRAINED" if k == train_index else ""),
        )

    # Visualization: bar chart of resonance gains
    out_path = Path(__file__).parent / "scenario_1_ga_learning_results.png"

    x = np.arange(len(stored_patterns))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline, width, label="Baseline")
    ax.bar(x + width / 2, post, width, label="Post-training")

    ax.set_xlabel("Primitive index")
    ax.set_ylabel("Peak GA resonance (weak cue)")
    ax.set_title("GA-SATTVA Learning: Weak-Cue Resonance Before/After Training")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(stored_patterns))])
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print("\n" + "=" * 70)
    print("GA LEARNING EXPERIMENT: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
