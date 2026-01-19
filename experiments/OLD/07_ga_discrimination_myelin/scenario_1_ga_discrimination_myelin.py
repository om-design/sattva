#!/usr/bin/env python3
"""Scenario 1: GA discrimination with myelination (scaled up).

Configuration:
- n_units = 4096
- 3 concept families (bases), 4 concepts per base (12 concepts total).
- Each family has:
  - Base Bi
  - 3 derived concepts Bi + Ek

Task:
- Noisy classification over 12 concepts using GA resonance.
- Compare accuracy before and after training with myelination.
- Phase 4: few-shot learning of a new concept D.
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from sattva.ga_sattva_core import (
    GAUnitSet,
    GASATTVADynamics,
    create_ga_primitive,
    GAPattern,
    pattern_from_units,
)
from sattva.myelination_substrate import MyelinationSubstrate


def build_families(units: GAUnitSet,
                   n_families: int = 3,
                   base_size: int = 64,
                   extra_size: int = 64,
                   seed: int = 321) -> dict[str, GAPattern]:
    rng = np.random.default_rng(seed)
    indices = np.arange(units.n_units)
    rng.shuffle(indices)

    ptr = 0
    primitives: dict[str, GAPattern] = {}

    for f in range(n_families):
        B = indices[ptr:ptr + base_size]
        ptr += base_size
        base_pattern = create_ga_primitive(units, B)
        primitives[f"base_{f}"] = base_pattern

        for k in range(3):  # 3 derived per base
            E = indices[ptr:ptr + extra_size]
            ptr += extra_size
            support = np.concatenate([B, E])
            primitives[f"family{f}_concept{k}"] = create_ga_primitive(units, support)

    # Add one extra disjoint concept as strong negative control
    D = indices[ptr:ptr + base_size]
    primitives["different_ctrl"] = create_ga_primitive(units, D)

    return primitives


def noisy_cue(pattern: GAPattern,
              drop_fraction: float = 0.4,
              noise_level: float = 0.15,
              rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()

    units = pattern.active_units.copy()
    rng.shuffle(units)
    n_keep = max(1, int(len(units) * (1.0 - drop_fraction)))
    cue_indices = units[:n_keep]

    base_level = 0.4
    noise = rng.normal(0.0, noise_level, size=n_keep)
    cue_levels = np.clip(base_level + noise, 0.0, 1.0)
    return cue_indices, cue_levels


def run_trial(units: GAUnitSet,
              dynamics: GASATTVADynamics,
              myelin: MyelinationSubstrate | None,
              primitives: dict[str, GAPattern],
              true_label: str,
              concept_labels: list[str],
              n_steps: int = 30,
              dt: float = 0.1,
              rng: np.random.Generator | None = None) -> str:
    if rng is None:
        rng = np.random.default_rng()

    pattern = primitives[true_label]
    cue_indices, cue_levels = noisy_cue(pattern, rng=rng)

    units.reset_activations(0.0)
    units.activations[cue_indices] = cue_levels.astype(np.float32)

    peaks = {label: 0.0 for label in concept_labels}

    for _ in range(n_steps):
        info = dynamics.step(dt=dt)
        if myelin is not None:
            myelin.step_myelination(
                activations=units.activations,
                ga_resonance=info["resonance"],
                dt=dt,
            )

        current = pattern_from_units(units, threshold=0.1)
        for label in concept_labels:
            r = current.resonance_strength(primitives[label])
            if r > peaks[label]:
                peaks[label] = r

    return max(peaks.items(), key=lambda kv: kv[1])[0]


def run_block(units, dynamics, myelin, primitives,
              concept_labels, n_trials_per_class=40,
              rng=None):
    if rng is None:
        rng = np.random.default_rng()

    correct = {label: 0 for label in concept_labels}

    for label in concept_labels:
        for _ in range(n_trials_per_class):
            pred = run_trial(units, dynamics, myelin,
                             primitives, label, concept_labels,
                             rng=rng)
            if pred == label:
                correct[label] += 1

    return {label: correct[label] / n_trials_per_class
            for label in concept_labels}


def learn_new_concept(
    units: GAUnitSet,
    dynamics: GASATTVADynamics,
    base_primitives: dict[str, GAPattern],
    known_labels: list[str],
    new_label: str,
    base_label: str,
    n_support_units: int = 64,
    n_shots: int = 10,
    n_steps: int = 30,
    dt: float = 0.1,
    ema_alpha: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, GAPattern], dict[str, float], dict[str, float]]:
    """Few-shot learning of a new concept via EMA over settled patterns.

    Parameters
    ----------
    new_label:
        Name of the new concept to add (e.g., "new_conceptD").
    base_label:
        Existing concept label whose pattern defines the ground-truth
        for examples of the new concept.
    """
    if rng is None:
        rng = np.random.default_rng()

    primitives = dict(base_primitives)

    if base_label not in primitives:
        raise ValueError(f"base_label {base_label} not in primitives")

    true_pattern = primitives[base_label]

    # Initialize new concept with a random subset of units from the base pattern
    init_units = true_pattern.active_units.copy()
    rng.shuffle(init_units)
    init_units = init_units[:n_support_units]
    primitives[new_label] = create_ga_primitive(units, init_units)

    concept_labels_with_new = sorted(known_labels + [new_label])

    pre_acc = run_block(
        units,
        dynamics,
        None,
        primitives,
        concept_labels_with_new,
        n_trials_per_class=20,
        rng=rng,
    )

    ema = np.zeros(units.n_units, dtype=np.float32)

    for _ in range(n_shots):
        cue_indices, cue_levels = noisy_cue(true_pattern, rng=rng)
        units.reset_activations(0.0)
        units.activations[cue_indices] = cue_levels.astype(np.float32)

        for _ in range(n_steps):
            dynamics.step(dt=dt)

        settled = pattern_from_units(units, threshold=0.1)
        vec = np.zeros(units.n_units, dtype=np.float32)
        vec[settled.active_units] = 1.0
        ema = (1.0 - ema_alpha) * ema + ema_alpha * vec

    top_indices = np.argsort(ema)[-n_support_units:]
    primitives[new_label] = create_ga_primitive(units, top_indices)

    post_acc = run_block(
        units,
        dynamics,
        None,
        primitives,
        concept_labels_with_new,
        n_trials_per_class=20,
        rng=rng,
    )

    return primitives, pre_acc, post_acc


def main() -> None:
    print("=" * 70)
    print("GA-SATTVA: Scaled Discrimination with Myelination")
    print("=" * 70)

    n_units = 4096
    units = GAUnitSet(n_units=n_units)
    units.random_initialize_multivectors(seed=42)

    print(f"Units: {n_units}, mv_dim: {units.mv_dim}")

    primitives = build_families(units, n_families=3,
                                base_size=64, extra_size=64)

    concept_labels = [k for k in primitives.keys() if not k.startswith("base_")]
    concept_labels.sort()

    print("Concepts:")
    for label in concept_labels:
        print(f"  {label}: {len(primitives[label].active_units)} units")

    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=[primitives[l] for l in concept_labels],
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    myelin = MyelinationSubstrate(n_units=n_units)
    rng = np.random.default_rng(123)

    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE DISCRIMINATION (no myelination)")
    print("=" * 70)

    baseline = run_block(units, dynamics, None,
                         primitives, concept_labels,
                         n_trials_per_class=30,
                         rng=rng)

    for label, acc in baseline.items():
        print(f"  {label}: accuracy={acc:.3f}")

    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING WITH MYELINATION (family0 concepts)")
    print("=" * 70)

    train_labels = [l for l in concept_labels if l.startswith("family0_")]
    n_episodes = 400
    episode_length = 30

    for ep in range(n_episodes):
        for label in train_labels:
            pattern = primitives[label]
            units.reset_activations(0.0)
            units.activations[pattern.active_units] = 0.5

            for _ in range(episode_length):
                info = dynamics.step(dt=0.1)
                myelin.step_myelination(
                    activations=units.activations,
                    ga_resonance=info["resonance"],
                    dt=0.1,
                )

        if (ep + 1) % 50 == 0:
            snap = myelin.snapshot()
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"total_conductance={snap['total_conductance']:.2f}, "
                  f"n_myelinated_edges={snap['n_myelinated_edges']}")

    print("\n✓ Training with myelination complete\n")

    print("=" * 70)
    print("PHASE 3: DISCRIMINATION AFTER MYELINATION")
    print("=" * 70)

    post = run_block(units, dynamics, myelin,
                     primitives, concept_labels,
                     n_trials_per_class=30,
                     rng=rng)

    for label, acc in post.items():
        print(f"  {label}: accuracy={acc:.3f}")

    print("\n" + "=" * 70)
    print("ANALYSIS: Accuracy Before/After Myelination")
    print("=" * 70)

    fam0 = [l for l in concept_labels if l.startswith("family0_")]
    fam1 = [l for l in concept_labels if l.startswith("family1_")]
    fam2 = [l for l in concept_labels if l.startswith("family2_")]

    for group_name, group in [("family0", fam0), ("family1", fam1), ("family2", fam2)]:
        b = np.mean([baseline[l] for l in group])
        p = np.mean([post[l] for l in group])
        print(f"  {group_name}: baseline={b:.3f}, post={p:.3f}, delta={p - b:+.3f}")

    print("\n" + "=" * 70)
    print("PHASE 4: FEW-SHOT LEARNING OF NEW CONCEPT D")
    print("=" * 70)

    primitives_with_d, pre_d, post_d = learn_new_concept(
        units=units,
        dynamics=dynamics,
        base_primitives=primitives,
        known_labels=concept_labels,
        new_label="new_conceptD",
        base_label="family1_concept2",
        n_support_units=64,
        n_shots=10,
        n_steps=30,
        dt=0.1,
        ema_alpha=0.2,
        rng=rng,
    )

    concept_labels_with_d = sorted(concept_labels + ["new_conceptD"])

    print("Pre-learning accuracies (including D):")
    for label in concept_labels_with_d:
        print(f"  {label}: accuracy={pre_d.get(label, 0.0):.3f}")

    print("\nPost-learning accuracies (including D):")
    for label in concept_labels_with_d:
        print(f"  {label}: accuracy={post_d.get(label, 0.0):.3f}")

    # Phase 5: sequentially add new concepts E and F
    print("\n" + "=" * 70)
    print("PHASE 5: SEQUENTIAL NEW CONCEPTS E AND F")
    print("=" * 70)

    primitives_current = primitives_with_d
    labels_current = concept_labels_with_d

    sequence = [
        ("new_conceptE", "family0_concept1"),
        ("new_conceptF", "family2_concept0"),
    ]

    for new_label, base_label in sequence:
        print(f"\n-- Learning {new_label} from base {base_label} --")
        primitives_current, pre_acc, post_acc = learn_new_concept(
            units=units,
            dynamics=dynamics,
            base_primitives=primitives_current,
            known_labels=labels_current,
            new_label=new_label,
            base_label=base_label,
            n_support_units=64,
            n_shots=10,
            n_steps=30,
            dt=0.1,
            ema_alpha=0.2,
            rng=rng,
        )

        labels_current = sorted(labels_current + [new_label])

        # Compute simple stability stats over all labels
        pre_vals = np.array([pre_acc[l] for l in labels_current])
        post_vals = np.array([post_acc[l] for l in labels_current])

        print("Pre-learning accuracies (all concepts):")
        for label in labels_current:
            print(f"  {label}: accuracy={pre_acc.get(label, 0.0):.3f}")

        print("Post-learning accuracies (all concepts):")
        for label in labels_current:
            print(f"  {label}: accuracy={post_acc.get(label, 0.0):.3f}")

        print("Stability summary:")
        print(f"  pre  mean={pre_vals.mean():.3f}, min={pre_vals.min():.3f}")
        print(f"  post mean={post_vals.mean():.3f}, min={post_vals.min():.3f}")


if __name__ == "__main__":
    main()
