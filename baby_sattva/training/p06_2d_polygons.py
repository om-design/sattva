# training/p06_2d_polygons.py
import time
from typing import List, Tuple

import numpy as np

from container import SattvaContainer
from training.p01_lines import POS_BUCKETS  # reuse coarse positions

# Coarse polygon types and attributes
POLY_TYPES: List[str] = ["POLY::triangle", "POLY::quad", "POLY::pent"]
SIZE_BUCKETS: List[str] = ["SIZE::small", "SIZE::medium", "SIZE::large"]
ORI_BUCKETS: List[str] = ["ORI::0", "ORI::45", "ORI::90", "ORI::135"]


def _ensure_polygon_seeds(container: SattvaContainer) -> None:
    """
    Seed a small bank of canonical polygon primitives once.

    These are 'closed shape' primitives built symbolically via ProgramEmbedding,
    separate from the line/curve/junction seeds. [file:273]
    """
    eng = container.engine
    space = container.embedding
    meta = container.meta

    if meta.get("polygon_seeds_installed"):
        return

    print("Seeding 2D polygon primitives (p06_2d_polygons)...")

    # Simple grid over polygon type, position, and size; modest ORI variety
    for poly in POLY_TYPES:
        for pos in POS_BUCKETS:
            for size in SIZE_BUCKETS:
                # Keep orientation variety small for now
                for ori in ["ORI::0", "ORI::45"]:
                    base_ids = ["SHAPE::polygon"]
                    instr_ids = [poly, pos, size, ori]
                    v = space.encode_program(base_ids, instr_ids)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    eng.create_primitive(v, complexity=4.0)

    meta["polygon_seeds_installed"] = True


def _sample_polygon_stimulus(
    container: SattvaContainer,
) -> Tuple[np.ndarray, str, str]:
    """
    Sample a single 'polygon shape' stimulus.

    Encodes:
      - polygon type: triangle / quad / pent
      - position bucket: left / center / right / upper / lower
      - size bucket: small / medium / large
      - coarse orientation bucket: 0 / 45 / 90 / 135 degrees

    Representation is symbolic via ProgramEmbedding. [file:273]
    """
    space = container.embedding
    rng = space.rng

    poly = str(rng.choice(POLY_TYPES))
    pos = str(rng.choice(POS_BUCKETS))
    size = str(rng.choice(SIZE_BUCKETS))
    ori = str(rng.choice(ORI_BUCKETS))

    base_ids = ["SHAPE::polygon"]
    instr_ids = [poly, pos, size, ori]

    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n

    return v, poly, pos


def run_polygons_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "polygons_2d",
) -> None:
    """
    Phase: 2D polygon shapes (triangles and simple polygons), no curves.

    Behaviour:
      - Ensure a small canonical polygon seed bank exists.
      - Each step:
          * Sample a symbolic polygon description.
          * Encode to an 8D vector via ProgramEmbedding.
          * Activate the engine; run epiphany_check; step the engine.
      - Logging and metrics identical to run_lines. [file:273]
    """
    eng = container.engine

    _ensure_polygon_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v, poly, pos = _sample_polygon_stimulus(container)
        input_id = f"{phase_name}_step_{start_step + i + 1}_{poly}_{pos}"

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
