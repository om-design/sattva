#!/usr/bin/env python3
"""Part A: Shape-quality primitives and composite categories.

Goals:
- Define ~12 GA primitives capturing intrinsic shape qualities and parts.
- Build composite category patterns: rectangle-like, chair-like, other-shape.
- Test whether GA-SATTVA dynamics with a simple resonance-based decision
  can discriminate these categories from noisy cues.
"""

import numpy as np
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from sattva.ga_sattva_core import (
    GAUnitSet,
    GASATTVADynamics,
    create_ga_primitive,
    GAPattern,
    pattern_from_units,
)


QUALITY_PART_NAMES = [
    "round_curved",
    "sharp_cornered",
    "long_extent",
    "short_extent",
    "single_segment",
    "multi_segment",
    "pointed_tip",
    "blunt_end",
    "rect_part",
    "seat_part",
    "leg_part",
    "backrest_part",
]

CATEGORY_NAMES = [
    "rectangle_like",
    "chair_like",
    "other_shape",
    "football_like",
    "irregular_shape1",
    "irregular_shape2",
]


def allocate_primitive_units(
    n_units: int,
    primitive_names: list[str],
    units_per_primitive: int = 64,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Allocate disjoint unit index subsets for each primitive.

    This keeps the initial geometry simple and interpretable. Later we can
    introduce controlled overlaps between related primitives if desired.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_units)
    rng.shuffle(indices)

    mapping: dict[str, np.ndarray] = {}
    ptr = 0
    for name in primitive_names:
        if ptr + units_per_primitive > n_units:
            raise ValueError("Not enough units to allocate all primitives")
        mapping[name] = indices[ptr : ptr + units_per_primitive]
        ptr += units_per_primitive
    return mapping


def build_quality_part_primitives(
    units: GAUnitSet,
    mapping: dict[str, np.ndarray],
) -> dict[str, GAPattern]:
    primitives: dict[str, GAPattern] = {}
    for name, idx in mapping.items():
        primitives[name] = create_ga_primitive(units, idx)
    return primitives


def build_category_patterns(
    units: GAUnitSet,
    qp_primitives: dict[str, GAPattern],
) -> dict[str, GAPattern]:
    """Construct composite category patterns from quality/part primitives.

    rectangle_like:
      - rect_part, sharp_cornered, multi_segment, long_extent

    chair_like:
      - seat_part, leg_part, backrest_part, rect_part, multi_segment

    other_shape:
      - round_curved, single_segment, blunt_end + some extra generic units

    football_like:
      - round_curved, long_extent, single_segment, pointed_tip (American football / rugby ball)

    irregular_shape1 / irregular_shape2:
      - mixtures of sharp/round, multi/single segments with some extra units,
        sized to have similar footprint to rectangle_like.
    """
    rng = np.random.default_rng(999)

    def union_patterns(names: list[str]) -> np.ndarray:
        units_list: list[int] = []
        for nm in names:
            units_list.extend(qp_primitives[nm].active_units.tolist())
        return np.unique(np.array(units_list, dtype=np.int32))

    # Rectangle-like: deterministic union of relevant primitives
    rect_support = union_patterns(
        [
            "rect_part",
            "sharp_cornered",
            "multi_segment",
            "long_extent",
        ]
    )
    rectangle_like = create_ga_primitive(units, rect_support)

    # Chair-like: union of parts, sharing rect_part & multi_segment
    chair_support = union_patterns(
        [
            "seat_part",
            "leg_part",
            "backrest_part",
            "rect_part",
            "multi_segment",
        ]
    )
    chair_like = create_ga_primitive(units, chair_support)

    # Other-shape: mostly curved / single segment / blunt and a few random extras
    other_support = union_patterns(
        [
            "round_curved",
            "single_segment",
            "blunt_end",
        ]
    )
    all_units = np.arange(units.n_units)
    rng.shuffle(all_units)
    other_extra = all_units[: 64]
    other_support = np.unique(np.concatenate([other_support, other_extra]))
    other_shape = create_ga_primitive(units, other_support)

    # Football-like: elongated, rounded ends, single main segment, pointed tips
    football_support = union_patterns(
        [
            "round_curved",
            "long_extent",
            "single_segment",
            "pointed_tip",
        ]
    )
    football_like = create_ga_primitive(units, football_support)

    # Irregular shapes: similar footprint to rectangle, but mixed qualities
    irr1_support = union_patterns(
        [
            "rect_part",
            "round_curved",
            "multi_segment",
        ]
    )
    rng.shuffle(all_units)
    irr1_extra = all_units[: 32]
    irr1_support = np.unique(np.concatenate([irr1_support, irr1_extra]))
    irregular_shape1 = create_ga_primitive(units, irr1_support)

    irr2_support = union_patterns(
        [
            "sharp_cornered",
            "short_extent",
            "multi_segment",
        ]
    )
    rng.shuffle(all_units)
    irr2_extra = all_units[: 32]
    irr2_support = np.unique(np.concatenate([irr2_support, irr2_extra]))
    irregular_shape2 = create_ga_primitive(units, irr2_support)

    return {
        "rectangle_like": rectangle_like,
        "chair_like": chair_like,
        "other_shape": other_shape,
        "football_like": football_like,
        "irregular_shape1": irregular_shape1,
        "irregular_shape2": irregular_shape2,
    }


def noisy_cue(
    pattern: GAPattern,
    drop_fraction: float = 0.4,
    noise_level: float = 0.15,
    base_level: float = 0.5,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    units = pattern.active_units.copy()
    rng.shuffle(units)
    n_keep = max(1, int(len(units) * (1.0 - drop_fraction)))
    cue_indices = units[:n_keep]

    noise = rng.normal(0.0, noise_level, size=n_keep)
    cue_levels = np.clip(base_level + noise, 0.0, 1.0)
    return cue_indices, cue_levels


def run_trial(
    units: GAUnitSet,
    dynamics: GASATTVADynamics,
    primitives: dict[str, GAPattern],
    true_label: str,
    category_labels: list[str],
    n_steps: int = 30,
    dt: float = 0.1,
    rng: np.random.Generator | None = None,
) -> str:
    if rng is None:
        rng = np.random.default_rng()

    pattern = primitives[true_label]
    cue_indices, cue_levels = noisy_cue(pattern, rng=rng)

    units.reset_activations(0.0)
    units.activations[cue_indices] = cue_levels.astype(np.float32)

    peaks = {label: 0.0 for label in category_labels}

    for _ in range(n_steps):
        dynamics.step(dt=dt)
        current = pattern_from_units(units, threshold=0.1)
        for label in category_labels:
            r = current.resonance_strength(primitives[label])
            if r > peaks[label]:
                peaks[label] = r

    # Winner-take-all on peak resonance
    return max(peaks.items(), key=lambda kv: kv[1])[0]


def probe_resonance_profile(
    units: GAUnitSet,
    dynamics: GASATTVADynamics,
    category_patterns: dict[str, GAPattern],
    qp_primitives: dict[str, GAPattern],
    target_label: str,
    n_probes: int = 50,
    n_steps: int = 30,
    dt: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Probe resonance profile for a given target category.

    Returns
    -------
    cat_mean : dict[label, mean_resonance]
        Mean resonance to each category pattern over probes.
    qp_mean : dict[name, mean_resonance]
        Mean resonance to each quality/part primitive over probes.
    """
    if rng is None:
        rng = np.random.default_rng()

    cat_labels = list(category_patterns.keys())
    qp_labels = list(qp_primitives.keys())

    cat_accum = {l: 0.0 for l in cat_labels}
    qp_accum = {l: 0.0 for l in qp_labels}

    for _ in range(n_probes):
        pattern = category_patterns[target_label]
        cue_idx, cue_levels = noisy_cue(pattern, rng=rng)

        units.reset_activations(0.0)
        units.activations[cue_idx] = cue_levels.astype(np.float32)

        for _ in range(n_steps):
            dynamics.step(dt=dt)

        current = pattern_from_units(units, threshold=0.1)

        for l in cat_labels:
            cat_accum[l] += current.resonance_strength(category_patterns[l])
        for l in qp_labels:
            qp_accum[l] += current.resonance_strength(qp_primitives[l])

    cat_mean = {l: cat_accum[l] / n_probes for l in cat_labels}
    qp_mean = {l: qp_accum[l] / n_probes for l in qp_labels}
    return cat_mean, qp_mean


def run_block(
    units: GAUnitSet,
    dynamics: GASATTVADynamics,
    primitives: dict[str, GAPattern],
    category_labels: list[str],
    n_trials_per_class: int = 80,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    if rng is None:
        rng = np.random.default_rng()

    correct = {label: 0 for label in category_labels}

    for label in category_labels:
        for _ in range(n_trials_per_class):
            pred = run_trial(
                units,
                dynamics,
                primitives,
                true_label=label,
                category_labels=category_labels,
                rng=rng,
            )
            if pred == label:
                correct[label] += 1

    return {label: correct[label] / n_trials_per_class for label in category_labels}


def probe_resonance_profile(
    units: GAUnitSet,
    dynamics: GASATTVADynamics,
    category_patterns: dict[str, GAPattern],
    qp_primitives: dict[str, GAPattern],
    target_label: str,
    n_probes: int = 50,
    n_steps: int = 30,
    dt: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Probe resonance profile for a given target category.

    Returns
    -------
    cat_mean : dict[label, mean_resonance]
        Mean resonance to each category pattern over probes.
    qp_mean : dict[name, mean_resonance]
        Mean resonance to each quality/part primitive over probes.
    """
    if rng is None:
        rng = np.random.default_rng()

    cat_labels = list(category_patterns.keys())
    qp_labels = list(qp_primitives.keys())

    cat_accum = {l: 0.0 for l in cat_labels}
    qp_accum = {l: 0.0 for l in qp_labels}

    for _ in range(n_probes):
        pattern = category_patterns[target_label]
        cue_idx, cue_levels = noisy_cue(pattern, rng=rng)

        units.reset_activations(0.0)
        units.activations[cue_idx] = cue_levels.astype(np.float32)

        for _ in range(n_steps):
            dynamics.step(dt=dt)

        current = pattern_from_units(units, threshold=0.1)

        for l in cat_labels:
            cat_accum[l] += current.resonance_strength(category_patterns[l])
        for l in qp_labels:
            qp_accum[l] += current.resonance_strength(qp_primitives[l])

    cat_mean = {l: cat_accum[l] / n_probes for l in cat_labels}
    qp_mean = {l: qp_accum[l] / n_probes for l in qp_labels}
    return cat_mean, qp_mean


def main() -> None:
    print("=" * 70)
    print("GA-SATTVA Part A: Shape-Quality Composites")
    print("=" * 70)

    n_units = 4096
    units = GAUnitSet(n_units=n_units)
    units.random_initialize_multivectors(seed=42)

    print(f"Units: {n_units}, mv_dim: {units.mv_dim}")

    # 1) Allocate GA primitives for intrinsic qualities and parts
    mapping = allocate_primitive_units(
        n_units=n_units,
        primitive_names=QUALITY_PART_NAMES,
        units_per_primitive=64,
        seed=123,
    )

    qp_primitives = build_quality_part_primitives(units, mapping)

    print("Quality/part primitives:")
    for name in QUALITY_PART_NAMES:
        print(f"  {name}: {len(qp_primitives[name].active_units)} units")

    # 2) Build composite category patterns
    category_patterns = build_category_patterns(units, qp_primitives)

    print("\nCategory patterns:")
    for name in CATEGORY_NAMES:
        print(f"  {name}: {len(category_patterns[name].active_units)} units")

    # 3) Set up dynamics using only the category patterns as stored attractors
    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=[category_patterns[name] for name in CATEGORY_NAMES],
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    rng = np.random.default_rng(987)

    print("\n" + "=" * 70)
    print("PHASE: CATEGORY DISCRIMINATION UNDER NOISE")
    print("=" * 70)

    accuracies = run_block(
        units,
        dynamics,
        primitives=category_patterns,
        category_labels=CATEGORY_NAMES,
        n_trials_per_class=80,
        rng=rng,
    )

    for label, acc in accuracies.items():
        print(f"  {label}: accuracy={acc:.3f}")

    overall = np.mean(list(accuracies.values()))
    print(f"\nOverall mean accuracy: {overall:.3f}")

    # Chair resonance probe
    print("\n" + "=" * 70)
    print("PHASE: CHAIR_LIKE RESONANCE PROFILE")
    print("=" * 70)

    cat_mean, qp_mean = probe_resonance_profile(
        units=units,
        dynamics=dynamics,
        category_patterns=category_patterns,
        qp_primitives=qp_primitives,
        target_label="chair_like",
        n_probes=50,
        n_steps=30,
        dt=0.1,
        rng=rng,
    )

    print("Category resonances (mean over probes):")
    for label, val in sorted(cat_mean.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {label}: {val:.3f}")

    print("\nQuality/part resonances (mean over probes):")
    for name, val in sorted(qp_mean.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name}: {val:.3f}")
