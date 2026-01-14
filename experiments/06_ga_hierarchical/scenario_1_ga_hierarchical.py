#!/usr/bin/env python3
"""Scenario 1: GA hierarchical learning.

Goal: test whether GA-SATTVA supports hierarchical structure and the
slow emergence of a more basic primitive shared between related
concepts.

Design:
- Base primitive B: core units representing a shared subspace.
- Main primitive P_main: B plus extra units E1.
- Similar primitive P_sim: B plus different extras E2.
- Different primitive P_diff: disjoint units D.

Training only targets P_main. After training, weak cues are applied to
B, P_main, P_sim, and P_diff to see how GA resonance changes.

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


def build_hierarchical_primitives(units: GAUnitSet,
                                  base_size: int = 16,
                                  extra_size: int = 16,
                                  seed: int = 321) -> dict[str, GAPattern]:
    """Construct base, main, similar, and different GA primitives."""
    rng = np.random.default_rng(seed)
    indices = np.arange(units.n_units)
    rng.shuffle(indices)

    B = indices[:base_size]
    E1 = indices[base_size:base_size + extra_size]
    E2 = indices[base_size + extra_size:base_size + 2 * extra_size]
    D = indices[base_size + 2 * extra_size:base_size + 3 * extra_size]

    # Base primitive (core subspace)
    base_pattern = create_ga_primitive(units, B)

    # Main and similar primitives share B but have distinct extras
    main_support = np.concatenate([B, E1])
    sim_support = np.concatenate([B, E2])

    main_pattern = create_ga_primitive(units, main_support)
    sim_pattern = create_ga_primitive(units, sim_support)

    # Different primitive: disjoint support
    diff_pattern = create_ga_primitive(units, D)

    return {
        "base": base_pattern,
        "main": main_pattern,
        "similar": sim_pattern,
        "different": diff_pattern,
    }


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
    """Slow, conditional GA update toward observed pattern."""
    if observed.meta.get("empty"):
        return pattern

    sim = pattern.similarity(observed)
    if sim < sim_threshold:
        return pattern

    mv_old = pattern.multivector
    mv_new = (1.0 - eta) * mv_old + eta * observed.multivector
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
    print("GA-SATTVA: Hierarchical GA learning")
    print("=" * 70)

    # Initialize units
    n_units = 512
    units = GAUnitSet(n_units=n_units)
    units.random_initialize_multivectors(seed=42)

    print("Initialized GA units:")
    print(f"  n_units: {n_units}")
    print(f"  mv_dim:  {units.mv_dim}")

    # Build hierarchical primitives
    print("\n" + "=" * 70)
    print("PHASE 0: BUILDING HIERARCHICAL PRIMITIVES")
    print("=" * 70)

    primitives = build_hierarchical_primitives(units)
    base_pattern = primitives["base"]
    main_pattern = primitives["main"]
    sim_pattern = primitives["similar"]
    diff_pattern = primitives["different"]

    print(f"  Base:       {len(base_pattern.active_units):3d} units")
    print(f"  Main:       {len(main_pattern.active_units):3d} units (B + E1)")
    print(f"  Similar:    {len(sim_pattern.active_units):3d} units (B + E2)")
    print(f"  Different:  {len(diff_pattern.active_units):3d} units (D)\n")

    stored_patterns = [main_pattern, sim_pattern, diff_pattern]

    # Dynamics
    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=stored_patterns,
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    # Helper: weak cue test on base and three primitives
    def weak_cue_test(label: str) -> dict[str, float]:
        print("=" * 70)
        print(label)
        print("=" * 70)

        results: dict[str, float] = {}

        # Cue only the base units
        base_cue = base_pattern.active_units
        res_base = run_episode(
            units,
            dynamics,
            cue_indices=base_cue,
            cue_level=0.3,
            n_steps=40,
        )
        results["base"] = res_base
        print(f"  Base cue:      peak_resonance={res_base:.6f}")

        # Cue main, similar, different (half their units at low level)
        for name, pattern in [
            ("main", main_pattern),
            ("similar", sim_pattern),
            ("different", diff_pattern),
        ]:
            cue = pattern.active_units[: len(pattern.active_units) // 2]
            peak_res = run_episode(
                units,
                dynamics,
                cue_indices=cue,
                cue_level=0.3,
                n_steps=40,
            )
            results[name] = peak_res
            print(f"  {name.capitalize():<9} cue: peak_resonance={peak_res:.6f}")

        print()
        return results

    # Phase 1: baseline weak-cue test
    baseline = weak_cue_test("PHASE 1: BASELINE WEAK-CUE TEST")

    # Phase 2: training on main primitive
    print("=" * 70)
    print("PHASE 2: TRAINING ON MAIN PRIMITIVE")
    print("=" * 70)

    n_episodes = 200
    episode_length = 40
    eta = 5e-4

    for ep in range(n_episodes):
        # Full cue of main primitive at moderate level
        units.reset_activations(0.0)
        units.activations[main_pattern.active_units] = 0.5

        # Run episode
        for _ in range(episode_length):
            dynamics.step(dt=0.1)

        # Observe pattern after episode and update main primitive
        observed = pattern_from_units(units, threshold=0.1)
        updated_main = update_pattern(main_pattern, observed, eta=eta, sim_threshold=0.9)
        main_pattern = updated_main
        stored_patterns[0] = updated_main

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: updated main primitive")

    print("\n✓ Training complete\n")

    # Phase 3: post-training weak-cue test
    post = weak_cue_test("PHASE 3: POST-TRAINING WEAK-CUE TEST")

    # Analysis
    print("=" * 70)
    print("ANALYSIS: Hierarchical GA Learning Effect")
    print("=" * 70)

    for key in ["base", "main", "similar", "different"]:
        b = baseline[key]
        p = post[key]
        print(
            f"{key.capitalize():<9}: baseline={b:.6f}, post={p:.6f}, "
            f"delta={p - b:+.6f}",
        )

    # Visualization: bar chart
    out_path = Path(__file__).parent / "scenario_1_ga_hierarchical_results.png"

    labels = ["base", "main", "similar", "different"]
    x = np.arange(len(labels))
    width = 0.35

    baseline_vals = np.array([baseline[k] for k in labels])
    post_vals = np.array([post[k] for k in labels])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline")
    ax.bar(x + width / 2, post_vals, width, label="Post-training")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Peak GA resonance (weak cue)")
    ax.set_title("GA-SATTVA Hierarchical Learning: Resonance Before/After Training")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print("\n" + "=" * 70)
    print("GA HIERARCHICAL LEARNING EXPERIMENT: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
