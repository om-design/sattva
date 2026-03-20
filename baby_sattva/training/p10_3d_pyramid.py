# training/p10_3d_pyramid.py

import time
from typing import List, Tuple

import numpy as np

from container import SattvaContainer

# 3D pyramid attributes
PYR_TYPES: List[str] = ["TYPE3D::pyramid"]
# Simple pose set: front (apex facing viewer), side, top (base view)
PYR_POSES: List[str] = ["POSE3D::front", "POSE3D::side", "POSE3D::top"]
PYR_SIZES: List[str] = ["SIZE3D::small", "SIZE3D::medium", "SIZE3D::large"]
PYR_POS_BUCKETS: List[str] = ["POS3D::center", "POS3D::left", "POS3D::right"]


def _ensure_pyramid_seeds(container: SattvaContainer) -> None:
    """
    Seed a small bank of canonical 3D pyramid primitives once.
    """
    eng = container.engine
    space = container.embedding
    meta = container.meta

    if meta.get("pyramid3d_seeds_installed"):
        return

    print("Seeding 3D pyramid primitives (p10_3d_pyramid)...")

    # 1 type × 3 poses × 3 sizes × 3 positions = 27 seeds
    for ptype in PYR_TYPES:
        for pose in PYR_POSES:
            for size in PYR_SIZES:
                for pos in PYR_POS_BUCKETS:
                    base_ids = ["OBJ3D::solid"]
                    instr_ids = [ptype, pose, size, pos]
                    v = space.encode_program(base_ids, instr_ids)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    eng.create_primitive(v, complexity=4.0)

    meta["pyramid3d_seeds_installed"] = True


def _sample_pyramid_stimulus(
    container: SattvaContainer,
) -> Tuple[np.ndarray, str, str]:
    """
    Sample a single 3D pyramid stimulus.
    """
    space = container.embedding
    rng = space.rng

    ptype = str(rng.choice(PYR_TYPES))      # TYPE3D::pyramid
    pose = str(rng.choice(PYR_POSES))
    size = str(rng.choice(PYR_SIZES))
    pos = str(rng.choice(PYR_POS_BUCKETS))

    base_ids = ["OBJ3D::solid"]
    instr_ids = [ptype, pose, size, pos]

    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n

    return v, ptype, pose


def run_3d_pyramid(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "pyramid3d",
) -> None:
    """
    Phase: 3D pyramid solids (symbolic, no explicit projection yet).
    """
    eng = container.engine

    _ensure_pyramid_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v, ptype, pose = _sample_pyramid_stimulus(container)
        step_idx = start_step + i + 1
        input_id = f"{phase_name}_step_{step_idx}_{ptype}_{pose}"

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
