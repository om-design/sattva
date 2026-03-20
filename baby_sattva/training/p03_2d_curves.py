# training/p03_2d_curves.py
import math
import time
from typing import Optional

import numpy as np

from container import SattvaContainer


def sample_arc_params(rng: np.random.Generator) -> dict:
    """
    Sample a single 2D arc (circular or elliptical) in a unit-square-ish field.
    """
    cx = rng.uniform(-0.5, 0.5)
    cy = rng.uniform(-0.5, 0.5)

    # Radii: allow circles and ellipses in a reasonable range
    rx = rng.uniform(0.1, 0.5)
    ry = rx * rng.uniform(0.7, 1.3)

    # Start/end angles with a minimum span to avoid degenerate arcs
    theta_start = rng.uniform(0.0, 2.0 * math.pi)
    min_span = math.radians(20.0)
    max_span = math.radians(300.0)
    span = rng.uniform(min_span, max_span)
    theta_end = theta_start + span

    # Wrap to [0, 2π)
    theta_start = theta_start % (2.0 * math.pi)
    theta_end = theta_end % (2.0 * math.pi)

    return {
        "cx": cx,
        "cy": cy,
        "rx": rx,
        "ry": ry,
        "theta_start": theta_start,
        "theta_end": theta_end,
    }


def encode_arc_to_vec(params: dict) -> np.ndarray:
    """
    Map arc parameters to an 8D embedding compatible with Engine(dim=8).

    Layout:
      [ cx,
        cy,
        cos(theta_start),
        sin(theta_start),
        cos(theta_end),
        sin(theta_end),
        rx_norm,
        ry_norm ]
    Then L2-normalize.
    """
    cx = params["cx"]
    cy = params["cy"]
    rx = params["rx"]
    ry = params["ry"]
    ts = params["theta_start"]
    te = params["theta_end"]

    # Normalise radii to a rough [0,1] range
    r_max = 0.5
    rx_norm = np.clip(rx / r_max, 0.0, 1.0)
    ry_norm = np.clip(ry / r_max, 0.0, 1.0)

    v = np.array(
        [
            cx,
            cy,
            math.cos(ts),
            math.sin(ts),
            math.cos(te),
            math.sin(te),
            rx_norm,
            ry_norm,
        ],
        dtype=np.float32,
    )

    # L2-normalize so cosine similarity is meaningful
    norm = np.linalg.norm(v)
    if norm > 0.0:
        v = v / norm

    return v


def _ensure_curve_seeds(container: SattvaContainer) -> None:
    """
    Seed a small bank of canonical curve primitives once.

    Analogous to _ensure_line_seeds, but we do not depend on the engine
    being empty, because line seeds are already present. We guard with
    a meta flag instead.
    """
    eng = container.engine
    meta: Optional[dict] = getattr(container, "meta", None)

    if isinstance(meta, dict) and meta.get("curve_seeds_installed"):
        return

    # 4 orientations × 2 radii = 8 curve seeds (quarter-circle arcs)
    orientations_deg = [0.0, 90.0, 180.0, 270.0]
    radii = [0.2, 0.4]

    print("Seeding curve primitives (p03_2d_curves)...")
    for deg in orientations_deg:
        theta_start = math.radians(deg)
        theta_end = theta_start + math.pi / 2.0  # quarter-arc

        for r in radii:
            params = {
                "cx": 0.0,
                "cy": 0.0,
                "rx": r,
                "ry": r,  # pure circles for seeds; ellipses emerge via training
                "theta_start": theta_start,
                "theta_end": theta_end,
            }
            v = encode_arc_to_vec(params)
            eng.create_primitive(v, complexity=3.0)

    if isinstance(meta, dict):
        meta["curve_seeds_installed"] = True


def _sample_curve_stimulus(container: SattvaContainer) -> np.ndarray:
    """
    Sample one 2D 'curve segment' stimulus, using the container's RNG.
    """
    space = container.embedding
    rng = space.rng  # same RNG source as lines

    params = sample_arc_params(rng)
    v = encode_arc_to_vec(params)
    return v


def run_curves_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "curves_2d",
) -> None:
    """
    Phase: 2D curves (arcs / ellipses), mirroring run_lines.

    Behaviour:
      - Ensure a small canonical curve seed bank exists.
      - For each step:
          * Sample a random arc stimulus and encode to 8D.
          * Activate the engine on this vector.
          * Run epiphany_check and log epiphanies via the container.
          * Advance engine dynamics with eng.step().
      - Logging: same mean_novelty / triage / tension fields as lines. [file:273]
    """
    eng = container.engine

    _ensure_curve_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v = _sample_curve_stimulus(container)
        input_id = f"{phase_name}_step_{start_step + i + 1}"

        eng.activate_input(v, magnitude=1.0)

        # Discovery-mode epiphany logging (same as lines)
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
