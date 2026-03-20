import time
from typing import List, Tuple

import numpy as np
from container import SattvaContainer

SQ_POLY_TYPES: List[str] = [
    "POLY2D::square",
    "POLY2D::rectangle",
]

# Heavily skew toward REL::next_to
REL_TYPES_WEIGHTED: List[str] = [
    "REL::next_to", "REL::next_to", "REL::next_to", "REL::next_to",
    "REL::above",
    "REL::below",
]

POS2D_BUCKETS: List[str] = [
    "POS2D::left",
    "POS2D::center",
    "POS2D::right",
]

def _ensure_square_rel_seeds(container: SattvaContainer) -> None:
    eng = container.engine
    space = container.embedding
    meta = container.meta

    if meta.get("square_rel2d_seeds_installed"):
        return

    print("Seeding 2D square/rectangle relation primitives (p11_square_relations_2d)...")

    # Use the unique relation set for seed coverage
    for rel in sorted(set(REL_TYPES_WEIGHTED)):
        for subj_shape in SQ_POLY_TYPES:
            for obj_shape in SQ_POLY_TYPES:
                for pos in POS2D_BUCKETS:
                    base_ids = ["OBJ2D::pair"]
                    instr_ids = [
                        rel,
                        "ROLE::figure", subj_shape,
                        "ROLE::ground", obj_shape,
                        pos,
                    ]
                    v = space.encode_program(base_ids, instr_ids)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    eng.create_primitive(v, complexity=5.0)

    meta["square_rel2d_seeds_installed"] = True

def _sample_square_relation_stimulus(
    container: SattvaContainer,
) -> Tuple[np.ndarray, str, str, str]:
    space = container.embedding
    rng = space.rng

    rel = str(rng.choice(REL_TYPES_WEIGHTED))
    subj_shape = str(rng.choice(SQ_POLY_TYPES))
    obj_shape = str(rng.choice(SQ_POLY_TYPES))
    pos = str(rng.choice(POS2D_BUCKETS))

    base_ids = ["OBJ2D::pair"]
    instr_ids = [
        rel,
        "ROLE::figure", subj_shape,
        "ROLE::ground", obj_shape,
        pos,
    ]

    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n

    return v, rel, subj_shape, obj_shape

def run_square_relations_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "square_relations_2d_a",
) -> None:
    eng = container.engine
    _ensure_square_rel_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v, rel, subj_shape, obj_shape = _sample_square_relation_stimulus(container)
        step_idx = start_step + i + 1

        input_id = (
            f"{phase_name}_step_{step_idx}_"
            f"{rel}_{subj_shape}_FIG_{obj_shape}_GRD"
        )

        eng.activate_input(v, magnitude=1.0)

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
        container.meta["step"] = step_idx

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
