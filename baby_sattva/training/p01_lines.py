# training/p01_lines.py
import time
from typing import List

import numpy as np

from container import SattvaContainer

ORIENTATIONS_DEG: List[int] = [0, 45, 90, 135]
POS_BUCKETS: List[str] = ["left", "center", "right", "upper", "lower"]


def _ensure_line_seeds(container: SattvaContainer) -> None:
    """
    Seed one primitive per (orientation, position) combo if the engine is empty.
    """
    eng = container.engine
    space = container.embedding

    if eng.primitives:
        return  # already seeded by a previous phase

    print("Seeding line primitives (p01_lines)...")
    for ori_deg in ORIENTATIONS_DEG:
        for pos_bucket in POS_BUCKETS:
            base_ids = ["SHAPE::edge"]
            instr_ids = [f"ORI::{ori_deg}", f"POS::{pos_bucket}"]
            v = space.encode_program(base_ids, instr_ids)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            eng.create_primitive(v, complexity=3.0)


def _sample_line_stimulus(container: SattvaContainer) -> np.ndarray:
    """
    Sample one 2D "line segment" stimulus: SHAPE::edge + ORI + POS.
    """
    space = container.embedding
    rng = space.rng

    ori_deg = int(rng.choice(ORIENTATIONS_DEG))
    pos_bucket = str(rng.choice(POS_BUCKETS))

    base_ids = ["SHAPE::edge"]
    instr_ids = [f"ORI::{ori_deg}", f"POS::{pos_bucket}"]
    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def run_lines(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "lines_2d",
) -> None:
    """
    Phase 01: train on simple oriented lines with position buckets.
    """
    eng = container.engine

    _ensure_line_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v = _sample_line_stimulus(container)
        input_id = f"{phase_name}_step_{start_step + i + 1}"

        eng.activate_input(v, magnitude=1.0)

        # Discovery-mode epiphany logging
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
