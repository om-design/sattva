# training/p04_2d_shapes.py
import time
from typing import List, Tuple

import numpy as np

from container import SattvaContainer
from training.p01_lines import ORIENTATIONS_DEG  # reuse orientations

# Relation types at a junction
REL_TYPES: List[str] = ["REL::intersect", "REL::corner", "REL::end_on", "REL::tangent"]

# Pair types for the junction: purely conceptual tags
PAIR_TYPES: List[str] = ["PAIR::line_line", "PAIR::line_curve"]


def _ensure_shape_seeds(container: SattvaContainer) -> None:
    """
    Seed a small bank of canonical 2D junction/shape primitives once.

    We don't check 'engine empty' here (lines/curves are already present);
    instead we use a meta flag so we only seed once.
    """
    eng = container.engine
    space = container.embedding
    meta = container.meta

    if meta.get("shape_seeds_installed"):
        return

    print("Seeding 2D shape/junction primitives (p04_2d_shapes)...")

    # Simple grid: a few canonical intersections and corners
    # - pure line-line intersections
    # - pure line-line corners (L-shapes)
    for rel in ["REL::intersect", "REL::corner", "REL::end_on"]:
        for ori1 in [0, 45, 90, 135]:
            # For corners, bias ori2 to be roughly perpendicular
            if rel == "REL::corner":
                ori2 = (ori1 + 90) % 180
            else:
                ori2 = int((ori1 + 45) % 180)

            base_ids = ["SHAPE::junction"]
            instr_ids = [
                f"PAIR::line_line",
                f"REL::{rel.split('::')[-1].lower()}",
                f"ORI1::{ori1}",
                f"ORI2::{ori2}",
            ]
            v = space.encode_program(base_ids, instr_ids)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            eng.create_primitive(v, complexity=4.0)

    # A few line-curve junction seeds: intersecting, tangent, end_on
    for rel in ["REL::intersect", "REL::tangent", "REL::end_on"]:
        for ori1 in [0, 90]:
            base_ids = ["SHAPE::junction"]
            instr_ids = [
                "PAIR::line_curve",
                f"REL::{rel.split('::')[-1].lower()}",
                f"ORI1::{ori1}",
            ]
            v = space.encode_program(base_ids, instr_ids)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            eng.create_primitive(v, complexity=4.0)

    meta["shape_seeds_installed"] = True


def _sample_shape2d_stimulus(
    container: SattvaContainer,
) -> Tuple[np.ndarray, str, str]:
    """
    Sample one 2D 'shape junction' stimulus.

    Encodes:
      - pair type: line-line or line-curve
      - relation: intersect / corner / end_on / tangent
      - orientations of the primary line(s)

    Representation is symbolic via ProgramEmbedding, like lines/relations. [file:273]
    """
    space = container.embedding
    rng = space.rng

    # Pick relation and pair type
    rel = str(rng.choice(REL_TYPES))
    pair = str(rng.choice(PAIR_TYPES))

    # Sample orientations
    ori1 = int(rng.choice(ORIENTATIONS_DEG))

    if pair == "PAIR::line_line":
        # Second line orientation; for "corner" bias to perpendicular
        if rel == "REL::corner":
            ori2 = (ori1 + 90) % 180
        else:
            # Small random offset
            ori2 = int(rng.choice(ORIENTATIONS_DEG))
        ori2_tag = f"ORI2::{ori2}"
    else:
        # line-curve: only one explicit line orientation for now
        ori2_tag = "ORI2::none"

    base_ids = ["SHAPE::junction"]
    instr_ids = [
        pair,
        rel,
        f"ORI1::{ori1}",
        ori2_tag,
    ]

    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n

    return v, pair, rel


def run_shapes_2d(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "shapes_2d",
) -> None:
    """
    Phase: 2D shapes/junctions.

    Starts from your existing line and curve field and adds:
      - Line-line junctions: intersections, corners, end-on.
      - Line-curve junctions: intersecting, tangent, end-on.

    Uses the same training, epiphany logging, and metrics pattern as run_lines. [file:273]
    """
    eng = container.engine

    _ensure_shape_seeds(container)

    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v, pair, rel = _sample_shape2d_stimulus(container)
        input_id = f"{phase_name}_step_{start_step + i + 1}"

        eng.activate_input(v, magnitude=1.0)

        # Epiphany logging as in run_lines
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
