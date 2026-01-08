# SATTVA: Working conversation memory

This is a project-internal memory of the early SATTVA conversations.

It is not meant to be polished theory; it is meant to preserve intent, decisions, and the *shape* of the reasoning so future work stays aligned.

## Core intent

- Build an AI system where “understanding” is expressed as **stable semantic attractors** in a dynamical system, rather than as a token stream.
- Treat resonance/settling as an engineering metaphor: states evolve, stabilize, and become meaningful basins.
- Keep the project testable: start with simulations that can be probed and instrumented.

## Key distinctions that emerged

### Structure vs. result

A critical distinction: it is easy to tune a demo to *produce a desired-looking attractor*, but SATTVA needs a **structure that information moves through**.

This implies an explicit pipeline:

- Encode → Settle → Read out → Train

Where “Train” is tied to tasks and diagnostics (not aesthetics).

### Meaning as dynamics

- A “concept” should be a stable pattern of activity (attractor) rather than a single static vector.
- “Thinking” is trajectory behavior: settling, transitions, and compositions.

## What we built first (scaffold)

The project started with the smallest possible substrate:

- A toy semantic space (a few labeled vectors).
- A simple attractor core (continuous Hopfield-style recurrence).
- A runnable experiment that wires: cue → settle → nearest-pattern interpretation.

This intentionally exposes failure modes (like collapse to trivial attractors) early.

## What the first toy revealed

The toy often collapsed toward near-zero state.

Rather than “fixing” it by hand, the insight was:

- Collapse is a diagnostic that the current energy landscape is not shaped for semantic basins.
- The next step must be to introduce a **task + training objective** so attractors become useful and non-trivial.

## Design values (“sattva” as constraint)

Use *sattva* as a practical design target:

- Clarity: legible basins.
- Balance: stable but responsive dynamics.
- Coherence: related cues converge; unrelated cues stay separated.

Translate that into measurable instrumentation:

- Basin separation / entanglement metrics.
- Robustness to noise and partial cues.
- Avoidance of trivial attractors.

## Near-term implementation direction

### First real milestone

Add a minimal supervised training loop:

- Input: noisy/partial cue vector.
- Target: clean vector (or label) for the intended concept.
- Loss: distance between final settled state and target; add regularizers to discourage trivial collapse.

### Next expansions

- Replace toy vectors with real embeddings.
- Add tasks beyond recall: association chains, compositional settling, and eventually action proposals.

## Repo notes

- `README.md` should stay readable and motivating for newcomers.
- `theory/overview.md` stays short; deeper notes can live in separate files like this one.

