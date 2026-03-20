# training/p03_2d_relations.py
import time
from typing import List

import numpy as np

from container import SattvaContainer
from training.p01_lines import ORIENTATIONS_DEG, POS_BUCKETS  # reuse

# We reuse the same basic 2D primitive types
PRIMITIVE_TYPES: List[str] = [
    "SHAPE::edge",
    "SHAPE::corner_L",
    "SHAPE::tee",
    "SHAPE::angle",
]

# Simple 2D relations between two primitives
REL_KINDS: List[str] = [
    "touching",
    "above",
    "inside",
]


def _ensure_relation_seeds(container: SattvaContainer) -> None:
    """
    Seed a few generic relation prototypes if we haven't done so yet.

    These are not bound to specific shapes or positions; they just
    create wells for 'REL::touching', 'REL::above', 'REL::inside'
    that later inputs can snap to.
    """
    eng = container.engine
    space = container.embedding

    if container.meta.get("seeded_relations_2d", False):
        return

    print("Seeding 2D relation prototypes (p03_2d_relations)...")

    for rel in REL_KINDS:
        # Generic "A relates-to B" prototype
        base_ids = ["ROLE::A", "ROLE::B"]
        instr_ids = [f"REL::{rel}"]

        v = space.encode_program(base_ids, instr_ids)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n

        # complexity ~ 2 roles + 1 relation
        eng.create_primitive(v, complexity=3.0)

    container.meta["seeded_relations_2d"] = True


def _sample_relation2d_stimulus(container: SattvaContainer) -> np.ndarray:
    """
    Sample a 2D relation between two primitives.

    We don't explicitly model geometry here; instead we:
      - Pick two primitive types (e.g., edge, corner).
      - Give each a coarse position.
      - Choose a relation kind (touching, above, inside).
    All of that is encoded into one program vector.
    """
    space = container.embedding
    rng = space.rng

    shape_a = str(rng.choice(PRIMITIVE_TYPES))
    shape_b = str(rng.choice(PRIMITIVE_TYPES))
    pos_a = str(rng.choice(POS_BUCKETS))
    pos_b = str(rng.choice(POS_BUCKETS))
    rel = str(rng.choice(REL_KINDS))

    base_ids = [shape_a, shape_b]
    instr_ids = [
        f"POS_A::{pos_a}",
        f"POS_B::{pos_b}",
        f"REL::{rel}",
    ]

    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def run_relations_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "relations_2d",
) -> None:
    """
    Phase 03: 2D relations between local primitives.

    This still avoids heavy 'objectness' and focuses on:
      - Two 2D primitives (edges, corners, tees, angles).
      - Coarse positions for each.
      - A simple relation tag (touching, above, inside).

    Over time, SATTVA can:
      - Build wells for REL::touching / REL::above / REL::inside.
      - Form composites that capture common shape+relation motifs.
    """
    eng = container.engine

    # Install generic relation wells if needed
    _ensure_relation_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v = _sample_relation2d_stimulus(container)
        input_id = f"{phase_name}_step_{start_step + i + 1}"

        eng.activate_input(v, magnitude=1.0)

        # Keep discovery-mode epiphany logging; can tighten later
        epiphanies = eng.epiphany_check(novelty_threshold=0.2)
        if epiphanies:
            container.log_epiphany(
                phase=phase_name,
                input_id=input_id,
                epiphanies=epiphanies,
                tension=eng.last_tension,
                mean_novelty=eng.mean_novelty(),
                triage=eng.triage_score(),
            )

        eng.step()
        container.meta["step"] = start_step + i + 1

        if (i + 1) % log_every == 0:
            print(
                f"[{phase_name}] step={container.meta['step']} "
                f"mean_novelty={eng.mean_novelty():.3f} "
                f"triage={eng.triage_score():.3f} "
                f"tension={eng.last_tension:.3f}"
            )

    dt = time.time() - t0
    container.meta.setdefault("curriculum_log", []).append(
        {
            "phase": phase_name,
            "steps": steps,
            "start_step": start_step + 1,
            "end_step": container.meta["step"],
            "duration_sec": dt,
            "timestamp": time.time(),
        }
    )
    print(
        f"Completed phase '{phase_name}' from step {start_step + 1} "
        f"to {container.meta['step']} in {dt:.1f} sec"
    )
