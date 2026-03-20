#!/usr/bin/env python3
"""Developmental GA-SATTVA: prototype discovery and like/not-like behavior.

This experiment:
- Starts from a small set of "edge-like" primitives (V1-style feature fields).
- Repeatedly presents random combinations of these primitives to the GA substrate.
- Lets dynamics settle, then clusters settled patterns into emergent prototypes.
- Probes a simple like-unit / not-like-unit statistic for one prototype.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Repo import setup
# ---------------------------------------------------------------------

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from sattva.ga_sattva_core import (  # type: ignore
    GAUnitSet,
    GASATTVADynamics,
    GAPattern,
    create_ga_primitive,
    pattern_from_units,
)

# ---------------------------------------------------------------------
# Primitive feature names for two domains
# ---------------------------------------------------------------------

VISUAL_PRIMITIVES = [
    "verticalish",
    "horizontalish",
    "obliqueish",
    "curvish",
    "long_extent",
    "short_extent",
]

SEMANTIC_PRIMITIVES = [
    "agency",
    "change",
    "proximity",
    "valence_positive",
    "valence_negative",
    "repetition",
]

# Recurring objects/concepts for each domain
VISUAL_OBJECTS = {
    "face_pattern": ["verticalish", "horizontalish", "curvish", "short_extent"],
    "hand_pattern": ["curvish", "short_extent", "obliqueish"],
    "bottle_pattern": ["verticalish", "long_extent", "curvish"],
    "block_pattern": ["verticalish", "horizontalish", "short_extent"],
    "ball_pattern": ["curvish", "short_extent"],
    "stick_pattern": ["verticalish", "long_extent"],
}

SEMANTIC_CONCEPTS = {
    "caregiver_approach": ["agency", "proximity", "change", "valence_positive"],
    "feeding_event": ["proximity", "valence_positive", "repetition"],
    "discomfort": ["valence_negative", "change"],
    "soothing": ["agency", "proximity", "valence_positive", "repetition"],
    "threat": ["agency", "proximity", "change", "valence_negative"],
    "play": ["agency", "valence_positive", "repetition"],
}


# ---------------------------------------------------------------------
# Unit allocation and primitive construction
# ---------------------------------------------------------------------


def allocate_primitive_units(
    n_units: int,
    primitive_names: list[str],
    units_per_primitive: int = 64,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Assign disjoint blocks of units to each primitive name."""
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
    """Create a GAPattern for each primitive from its allocated units."""
    return {name: create_ga_primitive(units, idx) for name, idx in mapping.items()}


# ---------------------------------------------------------------------
# Prototype store: emergent reference units
# ---------------------------------------------------------------------


class PrototypeStore:
    """Simple store for emergent prototypes."""

    def __init__(self) -> None:
        self.prototypes: list[GAPattern] = []

    def add(self, pattern: GAPattern) -> None:
        self.prototypes.append(pattern)

    def closest(self, pattern: GAPattern) -> tuple[int, float]:
        """Return (index, similarity) of the most resonant prototype."""
        best_idx = -1
        best_sim = -1.0
        for i, p in enumerate(self.prototypes):
            sim = pattern.resonance_strength(p)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, best_sim


# ---------------------------------------------------------------------
# Stimulus construction and noisy cue
# ---------------------------------------------------------------------


def make_recurring_stimulus(
    rng: np.random.Generator,
    units: GAUnitSet,
    qp_primitives: dict[str, GAPattern],
    recurring_objects: dict[str, list[str]],
    drop_frac: float = 0.3,
) -> tuple[GAPattern, str]:
    """Sample a recurring object and add noise/variation.
    
    Returns the noisy pattern and the object name it came from.
    """
    obj_name = rng.choice(list(recurring_objects.keys()))
    base_primitives = recurring_objects[obj_name]
    
    # Build base pattern
    idxs: list[int] = []
    for nm in base_primitives:
        idxs.extend(qp_primitives[nm].active_units.tolist())
    support = np.unique(np.array(idxs, dtype=np.int32))
    
    # Add variation by dropping some units
    if drop_frac > 0:
        rng.shuffle(support)
        n_keep = int((1.0 - drop_frac) * support.size)
        support = np.sort(support[:n_keep])
    
    return create_ga_primitive(units, support), obj_name


def noisy_cue(
    pattern: GAPattern,
    rng: np.random.Generator,
    drop_frac: float = 0.4,
    base_level: float = 0.6,
    noise: float = 0.2,
) -> GAPattern:
    """Produce a noisy cue from a pattern by dropping and jittering active units.

    Note: the returned GAPattern is only used for its active_units;
    multivector/mean_activation are dummies here.
    """
    active = pattern.active_units.copy()
    if active.size == 0:
        return GAPattern(
            active_units=np.array([], dtype=int),
            multivector=np.zeros(pattern.multivector.shape, dtype=np.float32),
            mean_activation=0.0,
            meta={"empty": True},
        )

    rng.shuffle(active)
    n_keep = int((1.0 - drop_frac) * active.size)
    kept = np.sort(active[:n_keep])

    mv = np.zeros_like(pattern.multivector, dtype=np.float32)
    return GAPattern(
        active_units=kept,
        multivector=mv,
        mean_activation=float(base_level),
        meta={"empty": False},
    )


# ---------------------------------------------------------------------
# Main developmental loop
# ---------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(0)

    n_units = 4096
    units = GAUnitSet(n_units=n_units, mv_dim=16)
    units.random_initialize_multivectors(seed=42)

    # Allocate primitive feature populations
    mapping = allocate_primitive_units(
        n_units=n_units,
        primitive_names=QUALITY_PART_NAMES,
        units_per_primitive=64,
        seed=123,
    )
    qp_primitives = build_quality_part_primitives(units, mapping)

    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=[],  # no pre-stored categories
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    store = PrototypeStore()

    # ------------------------------------------------------------------
    # Phase 1: Prototype discovery from primitive combinations
    # ------------------------------------------------------------------

    n_trials = 10000
    new_proto_threshold = 0.4  # similarity cutoff for "new prototype"
    reinforce_threshold = 0.6  # similarity cutoff for reinforcing existing prototype
    n_steps = 40
    dt = 0.1
    report_interval = 1000

    for t in range(n_trials):
        # Build a random stimulus
        stim_pattern = make_random_stimulus(rng, units, qp_primitives)

        # Reset unit activations
        units.reset_activations(0.0)

        # Inject noisy cue
        cue = noisy_cue(
            stim_pattern,
            rng=rng,
            drop_frac=0.4,
            base_level=0.6,
            noise=0.2,
        )
        units.activations[cue.active_units] = 1.0

        # Run dynamics
        for _ in range(n_steps):
            dynamics.step(dt=dt)
        settled = pattern_from_units(units, threshold=0.1)

        # Compare to existing prototypes
        if not store.prototypes:
            store.add(settled)
            continue

        best_idx, best_sim = store.closest(settled)
        
        # Reinforce existing prototype if very similar
        if best_sim >= reinforce_threshold:
            # Blend the settled pattern into the existing prototype
            old_proto = store.prototypes[best_idx]
            # Simple averaging of active units and multivectors
            combined_units = np.unique(np.concatenate([
                old_proto.active_units,
                settled.active_units
            ]))
            # Weight by mean_activation for blending multivectors
            old_weight = old_proto.mean_activation
            new_weight = settled.mean_activation
            total_weight = old_weight + new_weight
            blended_mv = (old_weight * old_proto.multivector + new_weight * settled.multivector) / total_weight
            
            store.prototypes[best_idx] = GAPattern(
                active_units=combined_units,
                multivector=blended_mv,
                mean_activation=(old_proto.mean_activation + settled.mean_activation) / 2,
                meta={"reinforced": True, "reinforcement_count": old_proto.meta.get("reinforcement_count", 0) + 1}
            )
        elif best_sim < new_proto_threshold:
            # Create new prototype if not similar to any existing
            store.add(settled)
        
        # Periodic reporting
        if (t + 1) % report_interval == 0:
            print(f"  Trial {t+1}/{n_trials}: {len(store.prototypes)} prototypes")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    print("======================================================================")
    print("GA-SATTVA Developmental Prototype Discovery (CLEAN)")
    print("======================================================================")
    print(f"Units: {n_units}, mv_dim: {units.mv_dim}")
    print("Quality-like primitives:")
    for name, pat in qp_primitives.items():
        print(f"  {name}: {pat.active_units.size} units")
    print("")
    print(f"Number of discovered prototypes: {len(store.prototypes)}")

    # ------------------------------------------------------------------
    # Phase 2: Like-unit / not-like-unit probe for one prototype
    # ------------------------------------------------------------------

    if len(store.prototypes) >= 2:
        ref_idx = 0
        ref_pat = store.prototypes[ref_idx]

        print("")
        print(f"PHASE: LIKE / NOT-LIKE RELATIVE TO PROTOTYPE {ref_idx}")
        like_count = 0
        not_like_count = 0
        n_probe = 100

        for _ in range(n_probe):
            # New random stimulus and settling
            stim_pattern = make_random_stimulus(rng, units, qp_primitives)

            units.reset_activations(0.0)
            cue = noisy_cue(
                stim_pattern,
                rng=rng,
                drop_frac=0.4,
                base_level=0.6,
                noise=0.2,
            )
            units.activations[cue.active_units] = 1.0

            for _ in range(n_steps):
                dynamics.step(dt=dt)
            settled = pattern_from_units(units, threshold=0.1)

            # Compare to reference vs best other
            ref_sim = settled.resonance_strength(ref_pat)
            best_idx, best_sim = store.closest(settled)
            if best_idx == ref_idx or ref_sim >= best_sim:
                like_count += 1
            else:
                not_like_count += 1

        print(f"  Like-unit: {like_count}/{n_probe}")
        print(f"  Not-like-unit: {not_like_count}/{n_probe}")

    # ------------------------------------------------------------------
    # Phase 3: Resonance of complex composite objects with prototypes
    # ------------------------------------------------------------------

    # Define a few hand-crafted composite "objects" in primitive space
    complex_objects = {
        "rect_combo": ["verticalish", "horizontalish", "long_extent"],
        "blob_combo": ["curvish", "short_extent"],
        "oblique_combo": ["obliqueish", "long_extent"],
    }

    def make_complex_object(names: list[str]) -> GAPattern:
        idxs: list[int] = []
        for nm in names:
            idxs.extend(qp_primitives[nm].active_units.tolist())
        support = np.unique(np.array(idxs, dtype=np.int32))
        return create_ga_primitive(units, support)

    complex_patterns: dict[str, GAPattern] = {
        name: make_complex_object(parts) for name, parts in complex_objects.items()
    }

    print("")
    print("PHASE: COMPLEX OBJECT RESONANCE WITH PROTOTYPES")
    for obj_name, obj_pat in complex_patterns.items():
        best_idx = -1
        best_sim = -1.0
        for i, proto in enumerate(store.prototypes):
            sim = obj_pat.resonance_strength(proto)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        print(f"  {obj_name}: best prototype {best_idx}, resonance={best_sim:.3f}")


if __name__ == "__main__":
    main()
