# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project workflow constraints

This project uses **bd** (beads) for issue tracking.

```bash
bd onboard
bd ready
bd show <id>
bd update <id> --status in_progress
bd close <id>
bd sync
```

When ending a work session, the repository expects work to be landed and pushed:

```bash
git pull --rebase
bd sync
git push
git status
```

`git status` must report the branch is up to date with origin before considering the session complete.

## Runtime layout (what is actively used)

The active code is split into two independent Python tracks:

1. `baby_sattva/` (main developmental loop)
   - Stateful training around a serialized `SattvaContainer` snapshot.
   - Curriculum is driven by YAML and modular phase files in `baby_sattva/training/`.
   - Diagnostic scripts read trained snapshots from `baby_sattva/artifacts/snapshots/`.

2. `engine/` (standalone resonance prototype)
   - Self-contained `SattvaEngine` implementation plus scripted training runs (`train.py`, `train-01.py`).
   - Produces JSON artifacts like `engine/training_log.json`.

Other top-level folders (`experiments/`, `theory/`, `architecture/`, `Sattva old/`) are mostly research notes, legacy experiments, or conceptual docs rather than a single packaged app.

## Core architecture map

### Baby SATTVA execution flow

Primary control path:

1. `baby_sattva/config.yaml` defines phase sequence (`module`, `func`, `steps`, `log_every`).
2. `baby_sattva/run_from_yaml.py` loads config, loads/creates container, dynamically imports phase functions, and saves after each phase.
3. `baby_sattva/container.py`
   - `SattvaContainer` bundles:
     - `Engine` and `ProgramEmbedding` (from `sattva_engine_v9.py`)
     - `meta` (`step`, `curriculum_log`, `epiphany_log`, seed flags)
   - Persists state via pickle (`artifacts/snapshots/baby_sattva.pkl`).
4. `baby_sattva/training/pNN_*.py` modules implement curricula:
   - seed primitives if needed
   - encode symbolic programs into vectors
   - call `engine.activate_input(...)`, `engine.epiphany_check(...)`, `engine.step()`
   - append phase metadata to `curriculum_log`
5. `baby_sattva/testing/*.py` and `baby_sattva/inspect_container.py` analyze trained snapshots.

### Engine internals (baby_sattva)

`baby_sattva/sattva_engine_v9.py` contains:
- `ProgramEmbedding`: deterministic vector construction from symbolic `base_ids` and `instr_ids`.
- `Primitive`: per-node state (embedding, energy, bandwidth/myelination-like state, parent/component links, usage statistics).
- `Engine`: activation, novelty/triage/tension tracking, epiphany detection, crystallisation/decomposition, and periodic consolidation.

### Standalone engine track

`engine/sattva_engine.py` is a separate implementation with:
- primitives, pillars, and orphans
- crystallization from orphan clusters
- cross-domain epiphany detection
- pruning utilities (`prune_pillars`, `prune_primitives`)

`engine/train.py` builds synthetic domain prototypes (geometry/physics/music/biology), runs phased training, then writes run data to `training_log.json`.

## Common commands

### Environment bootstrap (Python)

No root-level package/lockfile is present. Use a local venv and install the imports used by active scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pyyaml
```

### Run baby_sattva training

```bash
cd baby_sattva
python run_from_yaml.py --config config.yaml
```

### Inspect / diagnostics (baby_sattva)

```bash
cd baby_sattva
python inspect_container.py
python testing/reSi_style_polygon_diagnostic.py
python testing/sattva_emergent_diagnostics.py
```

### Run standalone engine experiments

```bash
cd engine
python train.py
python train-01.py
```

Optional Docker path for the standalone engine:

```bash
cd engine
docker build -t sattva-engine .
docker run --rm sattva-engine
```

## Testing and linting reality in this repo

- There is currently no canonical root test runner (`pytest`, `unittest`, etc.) or lint config (`ruff`, `flake8`, etc.).
- “Single test” in practice means running a specific diagnostic/experiment script directly, e.g.:

```bash
cd baby_sattva
python testing/reSi_style_polygon_diagnostic.py
```

- For quick syntax checks while editing, use:

```bash
python -m py_compile baby_sattva/*.py baby_sattva/training/*.py engine/*.py
```

