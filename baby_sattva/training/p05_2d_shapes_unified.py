# training/p05_2d_shapes_unified.py
import time
from typing import Tuple

import numpy as np

from container import SattvaContainer
from training.p01_lines import _sample_line_stimulus
from training.p03_2d_curves import _sample_curve_stimulus
from training.p04_2d_shapes import _sample_shape2d_stimulus, _ensure_shape_seeds
from training.p06_2d_polygons import _sample_polygon_stimulus, _ensure_polygon_seeds
from training.p07_2d_curved_shapes import (
    _sample_curved_shape_stimulus,
    _ensure_curved_shape_seeds,
)


def _sample_unified_2d_stimulus(
    container: SattvaContainer,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    """
    Sample one unified 2D stimulus from:
      - lines
      - local curves
      - junction shapes
      - polygons (tri/quad/pent)
      - curved closed shapes (circle/ellipse/capsule)
    Returns (vector, src_tag).
    """
    # Mixture weights; tweak as desired
    p_lines = 0.15
    p_curves = 0.15
    p_junctions = 0.30
    p_polygons = 0.20
    # remaining 0.20 -> curved closed shapes

    r = rng.random()
    if r < p_lines:
        v = _sample_line_stimulus(container)
        return v, "lines"
    elif r < p_lines + p_curves:
        v = _sample_curve_stimulus(container)
        return v, "curves"
    elif r < p_lines + p_curves + p_junctions:
        v, pair, rel = _sample_shape2d_stimulus(container)
        return v, "junctions"
    elif r < p_lines + p_curves + p_junctions + p_polygons:
        v, poly, pos = _sample_polygon_stimulus(container)
        return v, "polygons"
    else:
        v, ctype, pos = _sample_curved_shape_stimulus(container)
        return v, "curved_shapes"


def run_unified_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "unified_2d",
) -> None:
    """
    Unified 2D phase: interleaved lines, local curves, junctions,
    polygons, and curved closed shapes.

    Assumes all earlier 2D phases have run so their seeds exist.
    Uses the same training, epiphany logging, and metrics pattern as run_lines.
    """
    eng = container.engine
    meta = container.meta

    # Ensure seed banks exist (no-op if already done)
    _ensure_shape_seeds(container)
    _ensure_polygon_seeds(container)
    _ensure_curved_shape_seeds(container)

    start_step = meta.get("step", 0)
    t0 = time.time()

    space = container.embedding
    rng = space.rng

    for i in range(steps):
        v, src = _sample_unified_2d_stimulus(container, rng)
        step_idx = start_step + i + 1
        input_id = f"{phase_name}_step_{step_idx}_{src}"

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
        meta["step"] = step_idx

        if (i + 1) % log_every == 0:
            print(
                f"[{phase_name}] step={meta['step']} "
                f"mean_novelty={eng.mean_novelty():.3f} "
                f"triage={eng.triage_score():.3f} "
                f"tension={eng.last_tension:.3f}"
            )

    dt = time.time() - t0
    meta.setdefault("curriculum_log", []).append(
        {
            "phase": phase_name,
            "steps": steps,
            "start_step": start_step + 1,
            "end_step": meta["step"],
            "duration_sec": dt,
            "timestamp": time.time(),
        }
    )
    print(
        f"Completed phase '{phase_name}' from step {start_step + 1} "
        f"to {meta['step']} in {dt:.1f} sec"
    )
