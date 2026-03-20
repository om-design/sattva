# training/p02_2d_geometry.py
import time
from typing import List

import numpy as np

from container import SattvaContainer
from training.p01_lines import ORIENTATIONS_DEG, POS_BUCKETS  # reuse

PRIMITIVE_TYPES: List[str] = [
    "SHAPE::edge",
    "SHAPE::corner_L",
    "SHAPE::tee",
    "SHAPE::angle",
]

SCALE_BUCKETS: List[str] = ["small", "medium", "large"]


def _sample_primitive2d_stimulus(container: SattvaContainer) -> np.ndarray:
    """
    Sample a 2D primitive (edge, corner, tee, angle) with ORI, POS, SCALE.
    """
    space = container.embedding
    rng = space.rng

    shape = str(rng.choice(PRIMITIVE_TYPES))
    ori_deg = int(rng.choice(ORIENTATIONS_DEG))
    pos_bucket = str(rng.choice(POS_BUCKETS))
    scale = str(rng.choice(SCALE_BUCKETS))

    base_ids = [shape]
    instr_ids = [
        f"ORI::{ori_deg}",
        f"POS::{pos_bucket}",
        f"SCALE::{scale}",
    ]
    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def run_primitives_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "primitives_2d",
) -> None:
    """
    Phase 02: train on 2D primitive motifs (edges, corners, tees, angles)
    with orientation, position, and scale — still no explicit object relations.
    """
    eng = container.engine
    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v = _sample_primitive2d_stimulus(container)
        input_id = f"{phase_name}_step_{start_step + i + 1}"

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
