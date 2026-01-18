# SATTVA ⚛ 
Semantic Attractor Training of Transforming Vector Associations

SATTVA is a new type of AI engine. This research project explores how to build an **Attractor-based Understanding Machine (AUM)**: a system whose core representations are stable, semantic attractors in a dynamical state space rather than just token sequences. Most existing AI models gather a huge amount of statistics from training materials and guess the next most likely word.

The working hypothesis: if semantic vectors are trained so that their associations form stable attractors, then meaning, memory, and (eventually) agency can emerge from the geometry and dynamics of those attractors. In practical terms, this means that the form of connected concepts creates a geometry of neurons. When similar geometries are present, they resonate and form association. The more resonance locally, the farther it reaches. This allows two main advantages that are new. First, is that out of many, many connections many concepts can occupy the same space with patterns emerging as needed. Second, distant resonance can give rise to true creativity and perhaps imagination.

## What SATTVA means

SATTVA is both a name and an acronym:

- **S**emantic — representations live in meaning-bearing vector spaces (embeddings).
- **A**ttractor — “concepts” are stable regions/sets in state space that trajectories converge toward.
- **T**raining — we will learn parameters so attractors become useful, not just hand-constructed.
- **T**ransforming — associations change with experience; the system’s landscape is plastic.
- **V**ector — the substrate is vectors (not symbols), at least at the lowest layer.
- **A**ssociations — learning is about shaping how vectors influence and recall one another (content-addressable behavior).

## Why this could matter

- Opens a path to AI systems that **understand** via stable semantic structures, not just pattern matching over text.
- Provides a concrete playground for studying how meaning and memory can emerge from attractor dynamics instead of hand-designed symbols.
- Could inform architectures that are more robust and interpretable because their internal states live in an explicit attractor landscape (basins, stability, transitions).

## Architecture sketch

A minimal SATTVA loop looks like:

- **Encode**: convert an input cue (word/image/features) into a semantic state vector.
- **Settle**: evolve that state through a recurrent dynamical core until it approaches an attractor (a stable semantic state).
- **Read out**: interpret the settled state as a concept, memory, prediction, or action proposal.
- **Train**: adjust the core so the right attractors form and the right transitions occur (e.g., from partial cues to full memories).

## Roadmap

- **Phase 0 — Documentation spine**: keep README + `theory/overview.md` accurate and minimal while the core concepts stabilize.
- **Phase 1 — Toy loop (in progress)**: semantic cue → attractor core → settled state → interpretation, using small hand-built vector sets.
- **Phase 2 — First training objective**: introduce a small task and learn parameters so the system reliably settles into the right semantic attractors (and avoids collapsing to trivial states).
- **Phase 3 — Scaled semantics**: swap toy vectors for real embeddings and expand tasks (association, recall, composition).
- **Phase 4 — Toward agency**: add action proposals and simple environment feedback so attractor dynamics participate in decision-making, not just recall.

## What’s in this repo

- `theory/overview.md`: a concise conceptual overview (attractors + semantic vectors).
- `src/sattva/`: early code primitives (semantic space + attractor core).
- `experiments/`: runnable toys and probes.

## Current status

- A toy semantic space and a simple continuous Hopfield-style core exist as a starting point for exploration.
- The next milestone is to add a small task + training objective so this becomes an information-processing structure, not just a demo of dynamics.

## Getting started

- Read `theory/overview.md`.
- Run the toy experiment: `python experiments/attractor_toy/toy_attractor.py`.

This is early-stage and experimental; everything here is subject to change as the theory and simulations get sharper.

