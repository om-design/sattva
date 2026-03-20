# SATTVA Training Phases

This folder contains **modular training phases** for the baby SATTVA container.

Each phase is a small Python module that:

- Lives in `training/` (this folder).
- Exposes a single entry function (e.g. `run_lines`, `run_primitives_2d`).
- Accepts a `SattvaContainer` and a few simple arguments.
- Is invoked via `run_from_yaml.py` using the module + function names in `config.yaml`.

The goal is to make it easy to add new phases over time without touching the core engine or launcher. This follows standard Python packaging practices: one package (`training`) with multiple phase modules. [web:232]

---

## 1. Phase naming convention

Files:

- `p01_lines.py` – Phase 01: oriented lines.  
- `p02_2d_geometry.py` – Phase 02: 2D primitive motifs.  
- `p03_2d_relations.py` – Phase 03: 2D relations (placeholder for now).  

Pattern:

- Use `pNN_description.py` where:
  - `NN` is a zero-padded phase number (`01`, `02`, …).
  - `description` is short and descriptive (e.g. `lines`, `3d_shapes`, `loops`).

Modules are imported by name (e.g. `training.p01_lines`), so they **must** be valid Python identifiers (cannot start with a digit).

---

## 2. Phase function template

Every training phase module should:

1. Import `SattvaContainer` from `container.py`.  
2. Define one public function with the signature:

```
python
def run_phase_name(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "phase_name",
) -> None:
    ...
```
3.	Inside, it should:
	•	Possibly seed primitives (if needed).
	•	For each step:
	•	Generate an input vector (via  ProgramEmbedding ).
	•	Call  engine.activate_input(...) .
	•	Optionally call  engine.epiphany_check(...)  and  container.log_epiphany(...) .
	•	Call  engine.step() .
	•	Update  container.meta"step" .
	•	Append an entry to  container.meta"curriculum_log"  describing what it did.
A minimal template you can copy for new phases:
```
# training/pNN_new_phase.py
import time
import numpy as np
from container import SattvaContainer

def _sample_input(container: SattvaContainer) -> np.ndarray:
    # TODO: implement a sampler that returns a normalized vector
    space = container.embedding
    rng = space.rng

    # Example: one base + one instr
    base_ids = ["SOME::base"]
    instr_ids = ["SOME::instr"]
    v = space.encode_program(base_ids, instr_ids)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def run_new_phase(
    container: SattvaContainer,
    steps: int,
    log_every: int = 1000,
    phase_name: str = "new_phase",
) -> None:
    eng = container.engine
    start_step = container.meta.get("step", 0)
    t0 = time.time()

    for i in range(steps):
        v = _sample_input(container)
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
```
## 3. Wiring a new phase into the system
To activate a new phase:
	1.	Create the module in  training/ , following the template above.
	2.	Add a stanza to  config.yaml  under  run.phases :
```
run:
  phases:
    - name: "lines_2d"
      module: "training.p01_lines"
      func: "run_lines"
      steps: 20000
      log_every: 1000

    - name: "new_phase"
      module: "training.pNN_new_phase"
      func: "run_new_phase"
      steps: 30000
      log_every: 1000
```
3. Run:
```
python run_from_yaml.py --config config.yaml
```
 run_from_yaml.py  will:
	•	Load or create the  SattvaContainer .
	•	Import  training.pNN_new_phase .
	•	Call  run_new_phase(...)  with your  steps  and  log_every .
	•	Save the updated container snapshot after the phase.
4. Notes on seeding and re-use
	•	Seeding primitives:
	•	A phase may need to seed initial primitives when the engine is empty (e.g., Phase 01 lines).
	•	Later phases usually do not reseed— they just work with the existing wells and add new ones via training.
	•	Reusing vocab:
	•	Try to reuse the same symbolic base/instruction ids across phases (e.g.,  SHAPE::edge ,  ORI::0 ,  POS::left ) so wells become shared across curricula rather than fragmented.
	•	Epiphany logging:
	•	Logging epiphanies in all phases is fine; downstream filtering (by phase name, tension, depth, ancestor id) will decide which are interesting.
5. Recommended development order
	1.	 p01_lines.py  – oriented lines with position buckets.
	2.	 p02_2d_geometry.py  – local 2D motifs (corners, T‑junctions, angles).
	3.	 p03_2d_relations.py  – simple 2D relations (touching, inside, aligned).
	4.	 p04_3d_shapes.py  – basic 3D shapes and rotations.
	5.	 p05_physics_minimal.py  – simple dynamics (fall, support, slide).
	6.	Further phases (loops, grounded language, business episodes) on top.
Each new file can clone the template above and change only the  _sample_input  logic and the  phase_name .
6. Debugging tips
	•	If a phase runs and  Total primitives  in  inspect_container.py  stays 0, you forgot to seed or to create primitives.
	•	If  mean_novelty  is always 0 after seeding, it likely means:
	•	Inputs are identical to an existing primitive (no novelty), or
	•	Effective thresholds are too high (can adjust  base_activation_threshold  in  Engine  if needed). file:1
	•	If epiphany logs explode in count, you can:
	•	Raise  epiphany_k_sigma  in  Engine  to make tension gating stricter, or
	•	Increase  novelty_threshold  in  epiphany_check  calls for that phase.
