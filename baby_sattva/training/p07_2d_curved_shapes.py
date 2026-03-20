# training/p07_2d_curved_shapes.py

import time
from typing import List, Tuple

import numpy as np

from container import SattvaContainer
from training.p01_lines import POS_BUCKETS  # reuse coarse positions


# Curved closed shape types and attributes
CURVED_TYPES: List[str] = [
    "CURVED::circle",
    "CURVED::ellipse",
    "CURVED::capsule",  # rounded rectangle / stadium
]

SIZE_BUCKETS: List[str] = ["SIZE::small", "SIZE::medium", "SIZE::large"]
# Orientation is mainly meaningful for ellipse/capsule, but we keep it general
ORI_BUCKETS: List[str] = ["ORI::0", "ORI::45", "ORI::90", "ORI::135"]


def _ensure_curved_shape_seeds(container: SattvaContainer) -> None:
    """
    Seed a small bank of canonical curved closed-shape primitives once.

    These live alongside your polygon seeds but use CURVED::* tags so the
    engine can discover their relations without confusing them with POLY::*.
    """
    eng = container.engine
    space = container.embedding
    meta = container.meta

    if meta.get("curved_shape_seeds_installed"):
        return

    print("Seeding 2D curved closed-shape primitives (p07_2d_curved_shapes)...")

    # 3 curved types × 5 positions × 3 sizes × 2 orientations = 90 seeds max.
    for ctype in CURVED_TYPES:
        for pos in POS_BUCKETS:
            for size in SIZE_BUCKETS:
                # Restrict orientation variety for now so it's not over-parameterised
                for ori in ["ORI::0", "ORI::45"]:
                    base_ids = ["SHAPE::curved"]
                    instr_ids = [ctype, pos, size, ori]
                    v = space.encode_program(base_ids, instr_ids)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    eng.create_primitive(v, complexity=4.0)

    meta["curved_shape_seeds_installed"] = True


def _sample_curved_shape_stimulus(
    container: SattvaContainer,
) -> Tuple[np.ndarray, str, str]:
    """
    Sample a single 'curved closed shape' stimulus.

    Encodes:
      - curved type: circle / ellipse / capsule
      - position bucket: left / center / right / upper / lower
      - size bucket: small / medium / large
      - coarse orientation: 0 / 45 / 90 / 135 degrees

    Representation is symbolic via ProgramEmbedding.
    """
    space = container.embedding
    rng = space.rng

    ctype = str(rng.choice(CURVED_TYPES))
    pos = str(rng.choice(POS_BUCKETS))
    size = str(rng.choice(SIZE_BUCKETS))
    ori = str(rng.choice(ORI_BUCKETS))

    base_ids = ["SHAPE::curved"]
    instr_ids = [ctype, pos, size, ori]

    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n

    return v, ctype, pos


def run_curved_shapes_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "curved_shapes_2d",
) -> None:
    """
    Phase: 2D curved closed shapes (circles, ellipses, capsules).

    Behaviour:
      - Ensure a small canonical curved-shape seed bank exists.
      - Each step:
          * Sample a symbolic curved-shape description.
          * Encode to an 8D vector via ProgramEmbedding.
          * Activate the engine; run epiphany_check; step the engine.
      - Logging and metrics identical to run_lines.
    """
    eng = container.engine

    _ensure_curved_shape_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v, ctype, pos = _sample_curved_shape_stimulus(container)
        input_id = f"{phase_name}_step_{start_step + i + 1}_{ctype}_{pos}"

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
