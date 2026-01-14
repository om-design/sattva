#!/usr/bin/env python3
"""Scenario 1: GA-based expert intuition under stress.

Uses the GA-SATTVA core where:
- Units carry GA multivectors.
- Primitives are GA patterns built from subsets of units.
- Resonance is GA overlap between live state and expert library.

Phases:
- Phase 0: Build expert library.
- Phase 1: Baseline intuition from partial cue.
- Phase 2: Stress test (increase GA coupling).
- Phase 3: Recovery (return to baseline coupling).
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
)


def main() -> None:
    print("=" * 70)
    print("GA-SATTVA: Expert Intuition (GA resonance)")
    print("=" * 70)
    print("\nPatterns are GA multivectors; intuition is GA resonance, not search.\n")

    # Initialize units
    n_units = 512
    units = GAUnitSet(n_units=n_units)
    units.random_initialize_multivectors(seed=42)

    print("Initialized GA units:")
    print(f"  n_units: {n_units}")
    print(f"  mv_dim:  {units.mv_dim}")

    # Expert library: choose disjoint subsets as primitives
    rng = np.random.default_rng(123)
    primitive_size = 24
    indices = np.arange(n_units)
    rng.shuffle(indices)

    primitive_indices = [
        indices[i * primitive_size:(i + 1) * primitive_size]
        for i in range(5)
    ]

    stored_patterns: list[GAPattern] = []

    print("\n" + "=" * 70)
    print("PHASE 0: BUILDING EXPERT LIBRARY (GA primitives)")
    print("=" * 70)

    for k, idx in enumerate(primitive_indices):
        pattern = create_ga_primitive(units, idx)
        stored_patterns.append(pattern)
        print(f"  Primitive {k+1}: {len(pattern.active_units):3d} units, "
              f"mean_activation={pattern.mean_activation:.3f}")

    print(f"\n✓ Expert library: {len(stored_patterns)} GA primitives ready\n")

    # Dynamics
    dynamics = GASATTVADynamics(units=units, stored_patterns=stored_patterns,
                                gamma=1.5, u_rest=0.1, ga_coupling=0.2)

    # Storage for plotting
    steps = []
    means = []
    maxes = []
    resonances = []

    def record(step: int, info: dict[str, float]) -> None:
        steps.append(step)
        means.append(info["mean"])
        maxes.append(info["max"])
        resonances.append(info["resonance"])

    # Helper: run a phase and log
    def run_phase(start_step: int, n_steps: int, label: str) -> int:
        print("=" * 70)
        print(f"{label}")
        print("=" * 70)
        for s in range(n_steps):
            info = dynamics.step(dt=0.1)
            step = start_step + s
            record(step, info)
            if s % max(1, n_steps // 5) == 0:
                print(f"  Step {s:3d}: mean={info['mean']:.3f}, "
                      f"max={info['max']:.3f}, "
                      f"resonance={info['resonance']:.6f}")
        print()
        return start_step + n_steps

    # Phase 1: baseline intuition from partial cue of primitive 1
    print("=" * 70)
    print("PHASE 1: BASELINE INTUITION (partial cue)")
    print("=" * 70)

    # Cue: activate half of primitive 1's units
    units.reset_activations(0.0)
    p0 = stored_patterns[0]
    cue_indices = p0.active_units[: primitive_size // 2]
    units.activations[cue_indices] = 0.5

    step = 0
    step = run_phase(step, n_steps=60, label="PHASE 1: BASELINE (ga_coupling=0.2)")

    # Adjust GA coupling for stress
    print("Adjusting GA coupling for stress: 0.2 → 0.8\n")
    dynamics.ga_coupling = 0.8

    # Phase 2: stress test
    step = run_phase(step, n_steps=100, label="PHASE 2: STRESS (ga_coupling=0.8)")

    # Recovery: return GA coupling to baseline
    print("Returning GA coupling to baseline: 0.8 → 0.2\n")
    dynamics.ga_coupling = 0.2

    step = run_phase(step, n_steps=80, label="PHASE 3: RECOVERY (ga_coupling=0.2)")

    # Summaries
    baseline_mean = float(np.mean(means[40:60]))
    baseline_res = float(np.mean(resonances[40:60]))
    stress_res = float(np.mean(resonances[60:160]))
    recovery_mean = float(np.mean(means[-40:]))
    recovery_res = float(np.mean(resonances[-40:]))

    print("=" * 70)
    print("ANALYSIS: GA Expert Intuition Performance")
    print("=" * 70)

    print(f"\n[1] Baseline: mean={baseline_mean:.3f}, resonance={baseline_res:.6f}")
    print(f"[2] Stress:   resonance={stress_res:.6f}")
    print(f"[3] Recovery: mean={recovery_mean:.3f}, "
          f"resonance={recovery_res:.6f}\n")

    # Visualization
    out_path = Path(__file__).parent / "scenario_1_ga_results.png"
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_title("GA-SATTVA: Expert Intuition Under Stress (GA resonance)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Activation", color="tab:blue")
    ax1.plot(steps, means, label="Mean activation", color="tab:blue")
    ax1.plot(steps, maxes, label="Max activation", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("GA resonance", color="tab:green")
    ax2.plot(steps, resonances, label="GA resonance", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # Phase boundaries
    ax1.axvline(60, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(160, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print("\n" + "=" * 70)
    print("GA EXPERT INTUITION: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
