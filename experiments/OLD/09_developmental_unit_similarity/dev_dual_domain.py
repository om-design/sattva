#!/usr/bin/env python3
"""Developmental GA-SATTVA: Visual + Semantic domains with recurring objects.

This experiment mirrors infant development:
- Visual domain: recurring objects (face, hand, bottle, etc.)
- Semantic domain: recurring concepts (caregiver_approach, feeding, etc.)
- Both use the same prototype discovery mechanism
- Shows that repeated exposure to a small set of patterns creates stable prototypes
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from sattva.ga_sattva_core import (
    GAUnitSet,
    GASATTVADynamics,
    GAPattern,
    create_ga_primitive,
    pattern_from_units,
)

# ---------------------------------------------------------------------
# Domain definitions
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

VISUAL_OBJECTS = {
    "face": ["verticalish", "horizontalish", "curvish", "short_extent"],
    "hand": ["curvish", "short_extent", "obliqueish"],
    "bottle": ["verticalish", "long_extent", "curvish"],
    "block": ["verticalish", "horizontalish", "short_extent"],
    "ball": ["curvish", "short_extent"],
    "stick": ["verticalish", "long_extent"],
}

SEMANTIC_CONCEPTS = {
    "caregiver_approach": ["agency", "proximity", "change", "valence_positive"],
    "feeding": ["proximity", "valence_positive", "repetition"],
    "discomfort": ["valence_negative", "change"],
    "soothing": ["agency", "proximity", "valence_positive", "repetition"],
    "threat": ["agency", "proximity", "change", "valence_negative"],
    "play": ["agency", "valence_positive", "repetition"],
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def allocate_primitive_units(
    n_units: int,
    primitive_names: list[str],
    units_per_primitive: int = 64,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Assign disjoint blocks of units to each primitive."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_units)
    rng.shuffle(indices)

    mapping: dict[str, np.ndarray] = {}
    ptr = 0
    for name in primitive_names:
        if ptr + units_per_primitive > n_units:
            raise ValueError("Not enough units")
        mapping[name] = indices[ptr : ptr + units_per_primitive]
        ptr += units_per_primitive
    return mapping


def build_primitives(
    units: GAUnitSet,
    mapping: dict[str, np.ndarray],
) -> dict[str, GAPattern]:
    """Create GAPattern for each primitive."""
    return {name: create_ga_primitive(units, idx) for name, idx in mapping.items()}


class PrototypeStore:
    """Store for emergent prototypes."""

    def __init__(self) -> None:
        self.prototypes: list[GAPattern] = []

    def add(self, pattern: GAPattern) -> None:
        self.prototypes.append(pattern)

    def closest(self, pattern: GAPattern) -> tuple[int, float]:
        """Return (index, similarity) of most resonant prototype."""
        best_idx = -1
        best_sim = -1.0
        for i, p in enumerate(self.prototypes):
            sim = pattern.resonance_strength(p)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx, best_sim


def make_recurring_stimulus(
    rng: np.random.Generator,
    units: GAUnitSet,
    primitives: dict[str, GAPattern],
    recurring_objects: dict[str, list[str]],
    drop_frac: float = 0.3,
) -> tuple[GAPattern, str]:
    """Sample a recurring object and add variation.
    
    Returns (noisy pattern, source object name).
    """
    obj_name = rng.choice(list(recurring_objects.keys()))
    base_prims = recurring_objects[obj_name]
    
    idxs: list[int] = []
    for nm in base_prims:
        idxs.extend(primitives[nm].active_units.tolist())
    support = np.unique(np.array(idxs, dtype=np.int32))
    
    # Add variation
    if drop_frac > 0 and support.size > 0:
        rng.shuffle(support)
        n_keep = int((1.0 - drop_frac) * support.size)
        support = np.sort(support[:n_keep])
    
    return create_ga_primitive(units, support), obj_name


def noisy_cue(
    pattern: GAPattern,
    rng: np.random.Generator,
    drop_frac: float = 0.4,
) -> GAPattern:
    """Drop some units from pattern to create noisy cue."""
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

    return GAPattern(
        active_units=kept,
        multivector=np.zeros_like(pattern.multivector),
        mean_activation=0.6,
        meta={"empty": False},
    )


# ---------------------------------------------------------------------
# Domain discovery
# ---------------------------------------------------------------------


def run_domain_discovery(
    domain_name: str,
    primitive_names: list[str],
    recurring_objects: dict[str, list[str]],
    units: GAUnitSet,
    rng: np.random.Generator,
    n_trials: int = 5000,
    seed_offset: int = 0,
) -> PrototypeStore:
    """Run prototype discovery for one domain."""
    
    mapping = allocate_primitive_units(
        n_units=units.n_units,
        primitive_names=primitive_names,
        units_per_primitive=64,
        seed=123 + seed_offset,
    )
    primitives = build_primitives(units, mapping)

    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=[],
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    store = PrototypeStore()

    new_proto_threshold = 0.2
    reinforce_threshold = 0.3
    n_steps = 40
    dt = 0.1
    report_interval = 1000

    print(f"\n{'=' * 70}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Primitives: {', '.join(primitive_names)}")
    print(f"Recurring patterns: {', '.join(recurring_objects.keys())}")
    print(f"\nRunning {n_trials} exposures...")

    for t in range(n_trials):
        stim_pattern, obj_name = make_recurring_stimulus(
            rng, units, primitives, recurring_objects
        )

        units.reset_activations(0.0)
        cue = noisy_cue(stim_pattern, rng=rng, drop_frac=0.4)
        units.activations[cue.active_units] = 1.0

        for _ in range(n_steps):
            dynamics.step(dt=dt)
        settled = pattern_from_units(units, threshold=0.1)

        if not store.prototypes:
            settled.meta["source_object"] = obj_name
            store.add(settled)
            continue

        best_idx, best_sim = store.closest(settled)
        
        if best_sim >= reinforce_threshold:
            # Reinforce existing prototype
            old_proto = store.prototypes[best_idx]
            combined_units = np.unique(np.concatenate([
                old_proto.active_units,
                settled.active_units
            ]))
            old_weight = old_proto.mean_activation
            new_weight = settled.mean_activation
            total_weight = old_weight + new_weight
            blended_mv = (old_weight * old_proto.multivector + new_weight * settled.multivector) / total_weight
            
            store.prototypes[best_idx] = GAPattern(
                active_units=combined_units,
                multivector=blended_mv,
                mean_activation=(old_proto.mean_activation + settled.mean_activation) / 2,
                meta={
                    "source_object": old_proto.meta.get("source_object", obj_name),
                    "reinforced": True,
                    "reinforcement_count": old_proto.meta.get("reinforcement_count", 0) + 1
                }
            )
        elif best_sim < new_proto_threshold:
            # Create new prototype
            settled.meta["source_object"] = obj_name
            store.add(settled)
        
        if (t + 1) % report_interval == 0:
            print(f"  Trial {t+1}/{n_trials}: {len(store.prototypes)} prototypes")

    print(f"\nDiscovered {len(store.prototypes)} prototypes from {len(recurring_objects)} recurring patterns")
    
    # Show which prototypes were reinforced
    reinforced_count = sum(1 for p in store.prototypes if p.meta.get("reinforced", False))
    print(f"Reinforced prototypes: {reinforced_count}/{len(store.prototypes)}")
    
    return store


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(0)

    n_units = 4096
    units = GAUnitSet(n_units=n_units, mv_dim=16)
    units.random_initialize_multivectors(seed=42)

    print("=" * 70)
    print("GA-SATTVA Developmental Prototype Discovery")
    print("Visual + Semantic Domains with Recurring Objects")
    print("=" * 70)
    print(f"Units: {n_units}, mv_dim: {units.mv_dim}\n")

    visual_store = run_domain_discovery(
        domain_name="visual",
        primitive_names=VISUAL_PRIMITIVES,
        recurring_objects=VISUAL_OBJECTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=0,
    )

    semantic_store = run_domain_discovery(
        domain_name="semantic",
        primitive_names=SEMANTIC_PRIMITIVES,
        recurring_objects=SEMANTIC_CONCEPTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=100,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Visual domain: {len(visual_store.prototypes)} prototypes from {len(VISUAL_OBJECTS)} objects")
    print(f"Semantic domain: {len(semantic_store.prototypes)} prototypes from {len(SEMANTIC_CONCEPTS)} concepts")
    print("\nWith recurring exposure, the system should discover ~1-2 prototypes per recurring pattern,")
    print("showing that repeated experience creates stable, reusable representations.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Developmental GA-SATTVA: Visual + Semantic domains with recurring objects.

This experiment mirrors infant development:
- Visual domain: recurring objects (face, hand, bottle, etc.)
- Semantic domain: recurring concepts (caregiver_approach, feeding, etc.)
- Both use the same prototype discovery mechanism
- Shows that repeated exposure to a small set of patterns creates stable prototypes
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from sattva.ga_sattva_core import (
    GAUnitSet,
    GASATTVADynamics,
    GAPattern,
    create_ga_primitive,
    pattern_from_units,
)

# ---------------------------------------------------------------------
# Domain definitions
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

VISUAL_OBJECTS = {
    "face": ["verticalish", "horizontalish", "curvish", "short_extent"],
    "hand": ["curvish", "short_extent", "obliqueish"],
    "bottle": ["verticalish", "long_extent", "curvish"],
    "block": ["verticalish", "horizontalish", "short_extent"],
    "ball": ["curvish", "short_extent"],
    "stick": ["verticalish", "long_extent"],
}

SEMANTIC_CONCEPTS = {
    "caregiver_approach": ["agency", "proximity", "change", "valence_positive"],
    "feeding": ["proximity", "valence_positive", "repetition"],
    "discomfort": ["valence_negative", "change"],
    "soothing": ["agency", "proximity", "valence_positive", "repetition"],
    "threat": ["agency", "proximity", "change", "valence_negative"],
    "play": ["agency", "valence_positive", "repetition"],
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def allocate_primitive_units(
    n_units: int,
    primitive_names: list[str],
    units_per_primitive: int = 64,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Assign disjoint blocks of units to each primitive."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_units)
    rng.shuffle(indices)

    mapping: dict[str, np.ndarray] = {}
    ptr = 0
    for name in primitive_names:
        if ptr + units_per_primitive > n_units:
            raise ValueError("Not enough units")
        mapping[name] = indices[ptr : ptr + units_per_primitive]
        ptr += units_per_primitive
    return mapping


def build_primitives(
    units: GAUnitSet,
    mapping: dict[str, np.ndarray],
) -> dict[str, GAPattern]:
    """Create GAPattern for each primitive."""
    return {name: create_ga_primitive(units, idx) for name, idx in mapping.items()}


class PrototypeStore:
    """Store for emergent prototypes."""

    def __init__(self) -> None:
        self.prototypes: list[GAPattern] = []

    def add(self, pattern: GAPattern) -> None:
        self.prototypes.append(pattern)

    def closest(self, pattern: GAPattern) -> tuple[int, float]:
        """Return (index, similarity) of most resonant prototype."""
        best_idx = -1
        best_sim = -1.0
        for i, p in enumerate(self.prototypes):
            sim = pattern.resonance_strength(p)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx, best_sim


def make_recurring_stimulus(
    rng: np.random.Generator,
    units: GAUnitSet,
    primitives: dict[str, GAPattern],
    recurring_objects: dict[str, list[str]],
    drop_frac: float = 0.3,
) -> tuple[GAPattern, str]:
    """Sample a recurring object and add variation.
    
    Returns (noisy pattern, source object name).
    """
    obj_name = rng.choice(list(recurring_objects.keys()))
    base_prims = recurring_objects[obj_name]
    
    idxs: list[int] = []
    for nm in base_prims:
        idxs.extend(primitives[nm].active_units.tolist())
    support = np.unique(np.array(idxs, dtype=np.int32))
    
    # Add variation
    if drop_frac > 0 and support.size > 0:
        rng.shuffle(support)
        n_keep = int((1.0 - drop_frac) * support.size)
        support = np.sort(support[:n_keep])
    
    return create_ga_primitive(units, support), obj_name


def noisy_cue(
    pattern: GAPattern,
    rng: np.random.Generator,
    drop_frac: float = 0.4,
) -> GAPattern:
    """Drop some units from pattern to create noisy cue."""
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

    return GAPattern(
        active_units=kept,
        multivector=np.zeros_like(pattern.multivector),
        mean_activation=0.6,
        meta={"empty": False},
    )


# ---------------------------------------------------------------------
# Domain discovery
# ---------------------------------------------------------------------


def run_domain_discovery(
    domain_name: str,
    primitive_names: list[str],
    recurring_objects: dict[str, list[str]],
    units: GAUnitSet,
    rng: np.random.Generator,
    n_trials: int = 5000,
    seed_offset: int = 0,
) -> PrototypeStore:
    """Run prototype discovery for one domain."""
    
    mapping = allocate_primitive_units(
        n_units=units.n_units,
        primitive_names=primitive_names,
        units_per_primitive=64,
        seed=123 + seed_offset,
    )
    primitives = build_primitives(units, mapping)

    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=[],
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    store = PrototypeStore()

    new_proto_threshold = 0.2
    reinforce_threshold = 0.3
    n_steps = 40
    dt = 0.1
    report_interval = 1000

    print(f"\n{'=' * 70}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Primitives: {', '.join(primitive_names)}")
    print(f"Recurring patterns: {', '.join(recurring_objects.keys())}")
    print(f"\nRunning {n_trials} exposures...")

    for t in range(n_trials):
        stim_pattern, obj_name = make_recurring_stimulus(
            rng, units, primitives, recurring_objects
        )

        units.reset_activations(0.0)
        cue = noisy_cue(stim_pattern, rng=rng, drop_frac=0.4)
        units.activations[cue.active_units] = 1.0

        for _ in range(n_steps):
            dynamics.step(dt=dt)
        settled = pattern_from_units(units, threshold=0.1)

        if not store.prototypes:
            settled.meta["source_object"] = obj_name
            store.add(settled)
            continue

        best_idx, best_sim = store.closest(settled)
        
        if best_sim >= reinforce_threshold:
            # Reinforce existing prototype
            old_proto = store.prototypes[best_idx]
            combined_units = np.unique(np.concatenate([
                old_proto.active_units,
                settled.active_units
            ]))
            old_weight = old_proto.mean_activation
            new_weight = settled.mean_activation
            total_weight = old_weight + new_weight
            blended_mv = (old_weight * old_proto.multivector + new_weight * settled.multivector) / total_weight
            
            store.prototypes[best_idx] = GAPattern(
                active_units=combined_units,
                multivector=blended_mv,
                mean_activation=(old_proto.mean_activation + settled.mean_activation) / 2,
                meta={
                    "source_object": old_proto.meta.get("source_object", obj_name),
                    "reinforced": True,
                    "reinforcement_count": old_proto.meta.get("reinforcement_count", 0) + 1
                }
            )
        elif best_sim < new_proto_threshold:
            # Create new prototype
            settled.meta["source_object"] = obj_name
            store.add(settled)
        
        if (t + 1) % report_interval == 0:
            print(f"  Trial {t+1}/{n_trials}: {len(store.prototypes)} prototypes")

    print(f"\nDiscovered {len(store.prototypes)} prototypes from {len(recurring_objects)} recurring patterns")
    
    # Show which prototypes were reinforced
    reinforced_count = sum(1 for p in store.prototypes if p.meta.get("reinforced", False))
    print(f"Reinforced prototypes: {reinforced_count}/{len(store.prototypes)}")
    
    return store


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(0)

    n_units = 4096
    units = GAUnitSet(n_units=n_units, mv_dim=16)
    units.random_initialize_multivectors(seed=42)

    print("=" * 70)
    print("GA-SATTVA Developmental Prototype Discovery")
    print("Visual + Semantic Domains with Recurring Objects")
    print("=" * 70)
    print(f"Units: {n_units}, mv_dim: {units.mv_dim}\n")

    visual_store = run_domain_discovery(
        domain_name="visual",
        primitive_names=VISUAL_PRIMITIVES,
        recurring_objects=VISUAL_OBJECTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=0,
    )

    semantic_store = run_domain_discovery(
        domain_name="semantic",
        primitive_names=SEMANTIC_PRIMITIVES,
        recurring_objects=SEMANTIC_CONCEPTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=100,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Visual domain: {len(visual_store.prototypes)} prototypes from {len(VISUAL_OBJECTS)} objects")
    print(f"Semantic domain: {len(semantic_store.prototypes)} prototypes from {len(SEMANTIC_CONCEPTS)} concepts")
    print("\nWith recurring exposure, the system should discover ~1-2 prototypes per recurring pattern,")
    print("showing that repeated experience creates stable, reusable representations.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Developmental GA-SATTVA: Visual + Semantic domains with recurring objects.

This experiment mirrors infant development:
- Visual domain: recurring objects (face, hand, bottle, etc.)
- Semantic domain: recurring concepts (caregiver_approach, feeding, etc.)
- Both use the same prototype discovery mechanism
- Shows that repeated exposure to a small set of patterns creates stable prototypes
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from sattva.ga_sattva_core import (
    GAUnitSet,
    GASATTVADynamics,
    GAPattern,
    create_ga_primitive,
    pattern_from_units,
)

# ---------------------------------------------------------------------
# Domain definitions
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

VISUAL_OBJECTS = {
    "face": ["verticalish", "horizontalish", "curvish", "short_extent"],
    "hand": ["curvish", "short_extent", "obliqueish"],
    "bottle": ["verticalish", "long_extent", "curvish"],
    "block": ["verticalish", "horizontalish", "short_extent"],
    "ball": ["curvish", "short_extent"],
    "stick": ["verticalish", "long_extent"],
}

SEMANTIC_CONCEPTS = {
    "caregiver_approach": ["agency", "proximity", "change", "valence_positive"],
    "feeding": ["proximity", "valence_positive", "repetition"],
    "discomfort": ["valence_negative", "change"],
    "soothing": ["agency", "proximity", "valence_positive", "repetition"],
    "threat": ["agency", "proximity", "change", "valence_negative"],
    "play": ["agency", "valence_positive", "repetition"],
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def allocate_primitive_units(
    n_units: int,
    primitive_names: list[str],
    units_per_primitive: int = 64,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Assign disjoint blocks of units to each primitive."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_units)
    rng.shuffle(indices)

    mapping: dict[str, np.ndarray] = {}
    ptr = 0
    for name in primitive_names:
        if ptr + units_per_primitive > n_units:
            raise ValueError("Not enough units")
        mapping[name] = indices[ptr : ptr + units_per_primitive]
        ptr += units_per_primitive
    return mapping


def build_primitives(
    units: GAUnitSet,
    mapping: dict[str, np.ndarray],
) -> dict[str, GAPattern]:
    """Create GAPattern for each primitive."""
    return {name: create_ga_primitive(units, idx) for name, idx in mapping.items()}


class PrototypeStore:
    """Store for emergent prototypes."""

    def __init__(self) -> None:
        self.prototypes: list[GAPattern] = []

    def add(self, pattern: GAPattern) -> None:
        self.prototypes.append(pattern)

    def closest(self, pattern: GAPattern) -> tuple[int, float]:
        """Return (index, similarity) of most resonant prototype."""
        best_idx = -1
        best_sim = -1.0
        for i, p in enumerate(self.prototypes):
            sim = pattern.resonance_strength(p)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx, best_sim


def make_recurring_stimulus(
    rng: np.random.Generator,
    units: GAUnitSet,
    primitives: dict[str, GAPattern],
    recurring_objects: dict[str, list[str]],
    drop_frac: float = 0.3,
) -> tuple[GAPattern, str]:
    """Sample a recurring object and add variation.
    
    Returns (noisy pattern, source object name).
    """
    obj_name = rng.choice(list(recurring_objects.keys()))
    base_prims = recurring_objects[obj_name]
    
    idxs: list[int] = []
    for nm in base_prims:
        idxs.extend(primitives[nm].active_units.tolist())
    support = np.unique(np.array(idxs, dtype=np.int32))
    
    # Add variation
    if drop_frac > 0 and support.size > 0:
        rng.shuffle(support)
        n_keep = int((1.0 - drop_frac) * support.size)
        support = np.sort(support[:n_keep])
    
    return create_ga_primitive(units, support), obj_name


def noisy_cue(
    pattern: GAPattern,
    rng: np.random.Generator,
    drop_frac: float = 0.4,
) -> GAPattern:
    """Drop some units from pattern to create noisy cue."""
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

    return GAPattern(
        active_units=kept,
        multivector=np.zeros_like(pattern.multivector),
        mean_activation=0.6,
        meta={"empty": False},
    )


# ---------------------------------------------------------------------
# Domain discovery
# ---------------------------------------------------------------------


def run_domain_discovery(
    domain_name: str,
    primitive_names: list[str],
    recurring_objects: dict[str, list[str]],
    units: GAUnitSet,
    rng: np.random.Generator,
    n_trials: int = 5000,
    seed_offset: int = 0,
) -> PrototypeStore:
    """Run prototype discovery for one domain."""
    
    mapping = allocate_primitive_units(
        n_units=units.n_units,
        primitive_names=primitive_names,
        units_per_primitive=64,
        seed=123 + seed_offset,
    )
    primitives = build_primitives(units, mapping)

    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=[],
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    store = PrototypeStore()

    new_proto_threshold = 0.2
    reinforce_threshold = 0.3
    n_steps = 40
    dt = 0.1
    report_interval = 1000

    print(f"\n{'=' * 70}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Primitives: {', '.join(primitive_names)}")
    print(f"Recurring patterns: {', '.join(recurring_objects.keys())}")
    print(f"\nRunning {n_trials} exposures...")

    for t in range(n_trials):
        stim_pattern, obj_name = make_recurring_stimulus(
            rng, units, primitives, recurring_objects
        )

        units.reset_activations(0.0)
        cue = noisy_cue(stim_pattern, rng=rng, drop_frac=0.4)
        units.activations[cue.active_units] = 1.0

        for _ in range(n_steps):
            dynamics.step(dt=dt)
        settled = pattern_from_units(units, threshold=0.1)

        if not store.prototypes:
            settled.meta["source_object"] = obj_name
            store.add(settled)
            continue

        best_idx, best_sim = store.closest(settled)
        
        if best_sim >= reinforce_threshold:
            # Reinforce existing prototype
            old_proto = store.prototypes[best_idx]
            combined_units = np.unique(np.concatenate([
                old_proto.active_units,
                settled.active_units
            ]))
            old_weight = old_proto.mean_activation
            new_weight = settled.mean_activation
            total_weight = old_weight + new_weight
            blended_mv = (old_weight * old_proto.multivector + new_weight * settled.multivector) / total_weight
            
            store.prototypes[best_idx] = GAPattern(
                active_units=combined_units,
                multivector=blended_mv,
                mean_activation=(old_proto.mean_activation + settled.mean_activation) / 2,
                meta={
                    "source_object": old_proto.meta.get("source_object", obj_name),
                    "reinforced": True,
                    "reinforcement_count": old_proto.meta.get("reinforcement_count", 0) + 1
                }
            )
        elif best_sim < new_proto_threshold:
            # Create new prototype
            settled.meta["source_object"] = obj_name
            store.add(settled)
        
        if (t + 1) % report_interval == 0:
            print(f"  Trial {t+1}/{n_trials}: {len(store.prototypes)} prototypes")

    print(f"\nDiscovered {len(store.prototypes)} prototypes from {len(recurring_objects)} recurring patterns")
    
    # Show which prototypes were reinforced
    reinforced_count = sum(1 for p in store.prototypes if p.meta.get("reinforced", False))
    print(f"Reinforced prototypes: {reinforced_count}/{len(store.prototypes)}")
    
    return store


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(0)

    n_units = 4096
    units = GAUnitSet(n_units=n_units, mv_dim=16)
    units.random_initialize_multivectors(seed=42)

    print("=" * 70)
    print("GA-SATTVA Developmental Prototype Discovery")
    print("Visual + Semantic Domains with Recurring Objects")
    print("=" * 70)
    print(f"Units: {n_units}, mv_dim: {units.mv_dim}\n")

    visual_store = run_domain_discovery(
        domain_name="visual",
        primitive_names=VISUAL_PRIMITIVES,
        recurring_objects=VISUAL_OBJECTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=0,
    )

    semantic_store = run_domain_discovery(
        domain_name="semantic",
        primitive_names=SEMANTIC_PRIMITIVES,
        recurring_objects=SEMANTIC_CONCEPTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=100,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Visual domain: {len(visual_store.prototypes)} prototypes from {len(VISUAL_OBJECTS)} objects")
    print(f"Semantic domain: {len(semantic_store.prototypes)} prototypes from {len(SEMANTIC_CONCEPTS)} concepts")
    print("\nWith recurring exposure, the system should discover ~1-2 prototypes per recurring pattern,")
    print("showing that repeated experience creates stable, reusable representations.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Developmental GA-SATTVA: Visual + Semantic domains with recurring objects.

This experiment mirrors infant development:
- Visual domain: recurring objects (face, hand, bottle, etc.)
- Semantic domain: recurring concepts (caregiver_approach, feeding, etc.)
- Both use the same prototype discovery mechanism
- Shows that repeated exposure to a small set of patterns creates stable prototypes
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from sattva.ga_sattva_core import (
    GAUnitSet,
    GASATTVADynamics,
    GAPattern,
    create_ga_primitive,
    pattern_from_units,
)

# ---------------------------------------------------------------------
# Domain definitions
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

VISUAL_OBJECTS = {
    "face": ["verticalish", "horizontalish", "curvish", "short_extent"],
    "hand": ["curvish", "short_extent", "obliqueish"],
    "bottle": ["verticalish", "long_extent", "curvish"],
    "block": ["verticalish", "horizontalish", "short_extent"],
    "ball": ["curvish", "short_extent"],
    "stick": ["verticalish", "long_extent"],
}

SEMANTIC_CONCEPTS = {
    "caregiver_approach": ["agency", "proximity", "change", "valence_positive"],
    "feeding": ["proximity", "valence_positive", "repetition"],
    "discomfort": ["valence_negative", "change"],
    "soothing": ["agency", "proximity", "valence_positive", "repetition"],
    "threat": ["agency", "proximity", "change", "valence_negative"],
    "play": ["agency", "valence_positive", "repetition"],
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def allocate_primitive_units(
    n_units: int,
    primitive_names: list[str],
    units_per_primitive: int = 64,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """Assign disjoint blocks of units to each primitive."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_units)
    rng.shuffle(indices)

    mapping: dict[str, np.ndarray] = {}
    ptr = 0
    for name in primitive_names:
        if ptr + units_per_primitive > n_units:
            raise ValueError("Not enough units")
        mapping[name] = indices[ptr : ptr + units_per_primitive]
        ptr += units_per_primitive
    return mapping


def build_primitives(
    units: GAUnitSet,
    mapping: dict[str, np.ndarray],
) -> dict[str, GAPattern]:
    """Create GAPattern for each primitive."""
    return {name: create_ga_primitive(units, idx) for name, idx in mapping.items()}


class PrototypeStore:
    """Store for emergent prototypes."""

    def __init__(self) -> None:
        self.prototypes: list[GAPattern] = []

    def add(self, pattern: GAPattern) -> None:
        self.prototypes.append(pattern)

    def closest(self, pattern: GAPattern) -> tuple[int, float]:
        """Return (index, similarity) of most resonant prototype."""
        best_idx = -1
        best_sim = -1.0
        for i, p in enumerate(self.prototypes):
            sim = pattern.resonance_strength(p)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx, best_sim


def make_recurring_stimulus(
    rng: np.random.Generator,
    units: GAUnitSet,
    primitives: dict[str, GAPattern],
    recurring_objects: dict[str, list[str]],
    drop_frac: float = 0.3,
) -> tuple[GAPattern, str]:
    """Sample a recurring object and add variation.
    
    Returns (noisy pattern, source object name).
    """
    obj_name = rng.choice(list(recurring_objects.keys()))
    base_prims = recurring_objects[obj_name]
    
    idxs: list[int] = []
    for nm in base_prims:
        idxs.extend(primitives[nm].active_units.tolist())
    support = np.unique(np.array(idxs, dtype=np.int32))
    
    # Add variation
    if drop_frac > 0 and support.size > 0:
        rng.shuffle(support)
        n_keep = int((1.0 - drop_frac) * support.size)
        support = np.sort(support[:n_keep])
    
    return create_ga_primitive(units, support), obj_name


def noisy_cue(
    pattern: GAPattern,
    rng: np.random.Generator,
    drop_frac: float = 0.4,
) -> GAPattern:
    """Drop some units from pattern to create noisy cue."""
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

    return GAPattern(
        active_units=kept,
        multivector=np.zeros_like(pattern.multivector),
        mean_activation=0.6,
        meta={"empty": False},
    )


# ---------------------------------------------------------------------
# Domain discovery
# ---------------------------------------------------------------------


def run_domain_discovery(
    domain_name: str,
    primitive_names: list[str],
    recurring_objects: dict[str, list[str]],
    units: GAUnitSet,
    rng: np.random.Generator,
    n_trials: int = 5000,
    seed_offset: int = 0,
) -> PrototypeStore:
    """Run prototype discovery for one domain."""
    
    mapping = allocate_primitive_units(
        n_units=units.n_units,
        primitive_names=primitive_names,
        units_per_primitive=64,
        seed=123 + seed_offset,
    )
    primitives = build_primitives(units, mapping)

    dynamics = GASATTVADynamics(
        units=units,
        stored_patterns=[],
        gamma=1.5,
        u_rest=0.1,
        ga_coupling=0.2,
    )

    store = PrototypeStore()

    new_proto_threshold = 0.2
    reinforce_threshold = 0.3
    n_steps = 40
    dt = 0.1
    report_interval = 1000

    print(f"\n{'=' * 70}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Primitives: {', '.join(primitive_names)}")
    print(f"Recurring patterns: {', '.join(recurring_objects.keys())}")
    print(f"\nRunning {n_trials} exposures...")

    for t in range(n_trials):
        stim_pattern, obj_name = make_recurring_stimulus(
            rng, units, primitives, recurring_objects
        )

        units.reset_activations(0.0)
        cue = noisy_cue(stim_pattern, rng=rng, drop_frac=0.4)
        units.activations[cue.active_units] = 1.0

        for _ in range(n_steps):
            dynamics.step(dt=dt)
        settled = pattern_from_units(units, threshold=0.1)

        if not store.prototypes:
            settled.meta["source_object"] = obj_name
            store.add(settled)
            continue

        best_idx, best_sim = store.closest(settled)
        
        if best_sim >= reinforce_threshold:
            # Reinforce existing prototype
            old_proto = store.prototypes[best_idx]
            combined_units = np.unique(np.concatenate([
                old_proto.active_units,
                settled.active_units
            ]))
            old_weight = old_proto.mean_activation
            new_weight = settled.mean_activation
            total_weight = old_weight + new_weight
            blended_mv = (old_weight * old_proto.multivector + new_weight * settled.multivector) / total_weight
            
            store.prototypes[best_idx] = GAPattern(
                active_units=combined_units,
                multivector=blended_mv,
                mean_activation=(old_proto.mean_activation + settled.mean_activation) / 2,
                meta={
                    "source_object": old_proto.meta.get("source_object", obj_name),
                    "reinforced": True,
                    "reinforcement_count": old_proto.meta.get("reinforcement_count", 0) + 1
                }
            )
        elif best_sim < new_proto_threshold:
            # Create new prototype
            settled.meta["source_object"] = obj_name
            store.add(settled)
        
        if (t + 1) % report_interval == 0:
            print(f"  Trial {t+1}/{n_trials}: {len(store.prototypes)} prototypes")

    print(f"\nDiscovered {len(store.prototypes)} prototypes from {len(recurring_objects)} recurring patterns")
    
    # Show which prototypes were reinforced
    reinforced_count = sum(1 for p in store.prototypes if p.meta.get("reinforced", False))
    print(f"Reinforced prototypes: {reinforced_count}/{len(store.prototypes)}")
    
    return store


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(0)

    n_units = 4096
    units = GAUnitSet(n_units=n_units, mv_dim=16)
    units.random_initialize_multivectors(seed=42)

    print("=" * 70)
    print("GA-SATTVA Developmental Prototype Discovery")
    print("Visual + Semantic Domains with Recurring Objects")
    print("=" * 70)
    print(f"Units: {n_units}, mv_dim: {units.mv_dim}\n")

    visual_store = run_domain_discovery(
        domain_name="visual",
        primitive_names=VISUAL_PRIMITIVES,
        recurring_objects=VISUAL_OBJECTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=0,
    )

    semantic_store = run_domain_discovery(
        domain_name="semantic",
        primitive_names=SEMANTIC_PRIMITIVES,
        recurring_objects=SEMANTIC_CONCEPTS,
        units=units,
        rng=rng,
        n_trials=5000,
        seed_offset=100,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Visual domain: {len(visual_store.prototypes)} prototypes from {len(VISUAL_OBJECTS)} objects")
    print(f"Semantic domain: {len(semantic_store.prototypes)} prototypes from {len(SEMANTIC_CONCEPTS)} concepts")
    print("\nWith recurring exposure, the system should discover ~1-2 prototypes per recurring pattern,")
    print("showing that repeated experience creates stable, reusable representations.")


if __name__ == "__main__":
    main()
