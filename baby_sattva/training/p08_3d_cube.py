# training/p08_3d_cube.py

import time
from typing import List, Tuple

import numpy as np

from container import SattvaContainer

# 3D cube attributes
CUBE_TYPES: List[str] = ["TYPE3D::cube"]
CUBE_POSES: List[str] = ["POSE3D::front", "POSE3D::side", "POSE3D::top"]
CUBE_SIZES: List[str] = ["SIZE3D::small", "SIZE3D::medium", "SIZE3D::large"]
CUBE_POS_BUCKETS: List[str] = ["POS3D::center", "POS3D::left", "POS3D::right"]


def _ensure_cube_seeds(container: SattvaContainer) -> None:
    """
    Seed a small bank of canonical 3D cube primitives once.

    These live in the same ProgramEmbedding space as your 2D shapes but use
    OBJ3D/TYPED3/POSE3D tags so the engine can treat them as a distinct
    '3D solids' family.
    """
    eng = container.engine
    space = container.embedding
    meta = container.meta

    if meta.get("cube3d_seeds_installed"):
        return

    print("Seeding 3D cube primitives (p08_3d_cube)...")

    # 1 type × 3 poses × 3 sizes × 3 positions = 27 seeds
    for ctype in CUBE_TYPES:  # currently just cube
        for pose in CUBE_POSES:
            for size in CUBE_SIZES:
                for pos in CUBE_POS_BUCKETS:
                    base_ids = ["OBJ3D::solid"]
                    instr_ids = [ctype, pose, size, pos]
                    v = space.encode_program(base_ids, instr_ids)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    eng.create_primitive(v, complexity=4.0)

    meta["cube3d_seeds_installed"] = True


def _sample_cube_stimulus(
    container: SattvaContainer,
) -> Tuple[np.ndarray, str, str]:
    """
    Sample a single 3D cube stimulus.

    Encodes:
      - OBJ3D::solid base
      - TYPE3D::cube
      - POSE3D::front/side/top
      - SIZE3D::small/medium/large
      - POS3D::center/left/right

    Representation is symbolic via ProgramEmbedding.
    """
    space = container.embedding
    rng = space.rng

    ctype = str(rng.choice(CUBE_TYPES))     # always TYPE3D::cube for now
    pose = str(rng.choice(CUBE_POSES))
    size = str(rng.choice(CUBE_SIZES))
    pos = str(rng.choice(CUBE_POS_BUCKETS))

    base_ids = ["OBJ3D::solid"]
    instr_ids = [ctype, pose, size, pos]

    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n

    # Use (ctype, pose) or (ctype, size) in the input_id for debugging
    return v, ctype, pose


def run_3d_cube(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "cube3d",
) -> None:
    """
    Phase: 3D cube solids (symbolic, no explicit projection yet).

    Behaviour:
      - Ensure a small canonical cube seed bank exists.
      - Each step:
          * Sample a symbolic cube description.
          * Encode to an 8D vector via ProgramEmbedding.
          * Activate the engine; run epiphany_check; step the engine.
      - Logging and metrics identical to run_lines.
    """
    eng = container.engine

    _ensure_cube_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v, ctype, pose = _sample_cube_stimulus(container)
        step_idx = start_step + i + 1
        input_id = f"{phase_name}_step_{step_idx}_{ctype}_{pose}"

        eng.activate_input(v, magnitude=1.0)

        # Same discovery-mode epiphany logging pattern as lines
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
