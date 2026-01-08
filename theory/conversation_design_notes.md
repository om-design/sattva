# SATTVA: Conversation-derived design notes

This document captures the *fuller arc* of the early SATTVA discussion: motivations, conceptual commitments, and the first implementation steps.

It is intentionally longer than `overview.md` and aims to preserve the original thinking while it is still fresh.

## 1) The seed intuition: transcendent resonance → engineering

The project began from a desire to build something that appreciates *transcendent amplification of resonance*: a system where patterns don’t merely get classified, but *amplify*, stabilize, and become legible as meaningful structure.

The core bet is that resonance is not only poetic language—it is a useful engineering metaphor for cognition:

- A mind can be seen as a set of resonant dynamics.
- A concept can be seen as a stable “standing wave” in an internal state space.
- Learning can be seen as shaping which resonances stabilize and which fade.

SATTVA takes this intuition and tries to turn it into a minimal, testable architecture.

## 2) “Sattva” as a design constraint

Independently of the acronym, *sattva* is used here as a north star for the system’s internal behavior and developer ergonomics.

- **Clarity**: learned states should be legible as stable semantic basins (not a fog of near-equivalent states).
- **Balance**: dynamics should be stable without becoming inert; flexible without becoming chaotic.
- **Coherence**: associative settling should preserve structure (related ideas converge; unrelated ideas stay separated).

Operationally, this suggests measuring and incentivizing:

- non-trivial attractors (avoid “everything goes to zero”),
- basin separation (less entanglement),
- predictable settling and transitions,
- robustness to noise and partial cues.

## 3) The core claim: understanding as attractor geometry

`overview.md` states the essential claim succinctly. The longer form is:

1. Represent meaning at the lowest layer as vectors (embeddings, latent codes, feature vectors).
2. Define a recurrent dynamical system over those vectors (a state that evolves over time).
3. Train the system so that *useful* semantic states become attractors (stable basins).
4. Treat “understanding” not as a static representation, but as:
   - convergence to stable semantic states,
   - reliable transitions between states,
   - compositional settling when multiple cues interact.

This reframes cognition as **trajectory + basin structure**, not token-by-token prediction.

## 4) Avoiding a key failure mode: “producing a result” vs “building a structure”

A key concern surfaced during the first toy experiment: tuning an attractor network to “show attractors” can become a demo that produces a desired result without creating a general information-processing structure.

The corrective move was to insist on an explicit *flow*:

- **Encode** → map cues into semantic state.
- **Settle** → let state evolve under dynamics.
- **Read out** → interpret the settled state.
- **Train** → adjust dynamics so this pipeline becomes useful.

In other words:

- The attractor landscape must be *learned to serve a task*, not hand-crafted to look pretty.

## 5) Minimal architecture: the SATTVA loop

A minimal SATTVA system can be described as:

### 5.1 Components

- **Semantic Space**
  - A store of labeled vectors (toy first; real embeddings later).
  - Eventually: richer objects than words (chunks, concepts, multi-modal symbols).

- **Attractor Core (Dynamical Core)**
  - A recurrent update rule over state vectors.
  - Defines basins of attraction (and ideally transitions).

- **Interpreter / Readout**
  - Converts the settled state into:
    - concept label,
    - retrieved memory,
    - prediction,
    - or action proposal.

- **Training Rule**
  - Shapes attractors and transitions.
  - Prevents collapse into trivial attractors (e.g., “everything goes to zero”).

### 5.2 Behaviors we care about

- **Recall from partial cue** (content-addressable memory).
- **Association** (nearby ideas pull each other into a coherent basin).
- **Stability with flexibility** (stable attractors, but able to move when new evidence arrives).
- **Compositional settling** (multiple cues converge to an integrated state).

## 6) The first experiment and what it taught

The initial toy experiment used a simple continuous Hopfield-style dynamic with tanh.

Observed behavior: the state often collapsed toward the origin (near-zero), rather than settling into one of the intended patterns.

That outcome was treated as a useful diagnostic:

- It demonstrates that “attractor dynamics” are not automatically meaningful.
- A continuous system can have a dominant trivial attractor unless the energy landscape is shaped appropriately.

This clarified the need for:

- a task-driven training objective,
- explicit monitoring of attractor quality (separation, stability),
- and careful design of dynamics so the system is *structurally* capable of semantic convergence.

## 7) Documentation decisions

A practical documentation insight emerged:

- Readers prefer a small number of files they can scroll.

So the theory was consolidated into:

- `theory/overview.md` — short, first-stop conceptual note.

And the README was expanded so newcomers do not have to chase many files.

## 8) Implementation decisions so far (code reality)

The repo currently contains:

- `src/sattva/semantic_space.py`
  - `SemanticSpace`: labeled vectors, toy starter set.

- `src/sattva/attractor_core.py`
  - `HopfieldCore`: a first continuous attractor core (Hebbian weights + tanh updates).

- `experiments/attractor_toy/toy_attractor.py`
  - A wiring script: semantic cue → core settle → nearest pattern report.

This code is intentionally minimal. It is a scaffold for the next step: **training**.

## 9) What “training” should mean here

“Training” in SATTVA is not yet defined as one specific algorithm; it is a family of possible objectives.

A useful first training pass (minimal and measurable) could be:

- **Supervised settling objective**
  - Input: a degraded/noisy cue vector (or mixture).
  - Target: the clean intended vector (or class label).
  - Loss: distance between final settled state and target.
  - Learn: parameters of the core (W, nonlinearity gain, additional matrices).

This makes “attractor formation” subordinate to a task:

- The system learns basins that are useful for retrieval/association.

Later training could add:

- transition learning (sequence of attractor visits),
- energy shaping / regularization for basin separation,
- multi-modal alignment,
- and eventually reinforcement-style feedback for action selection.

## 10) Roadmap as a research program

A working roadmap was articulated and then embedded in the README. In fuller prose:

- **Phase 0 — Documentation spine**: keep a coherent narrative that matches the code.
- **Phase 1 — Toy loop**: prove the pipeline and build intuition about failure modes.
- **Phase 2 — First training objective**: make the system learn non-trivial, reliable semantic attractors.
- **Phase 3 — Scaled semantics**: move from toy vectors to real embeddings and richer tasks.
- **Phase 4 — Toward agency**: add action proposals, feedback, and an environment.

## 11) Open questions (where the interesting work is)

These are the unresolved design questions implied by the conversation:

- **What counts as an attractor?**
  - Point attractors only?
  - Limit cycles?
  - Structured manifolds?

- **What is the right state space?**
  - Same space as embeddings?
  - A learned latent space whose geometry is optimized for attractor dynamics?

- **What is a good early task?**
  - Recall-from-cue (associative memory) is simplest.
  - But compositional settling may better reflect “understanding.”

- **How to avoid trivial attractors?**
  - Loss terms?
  - Normalization constraints?
  - Energy/temperature scheduling?

- **What are “Transforming Vector Associations” concretely?**
  - Changing edges in a graph?
  - Learning weights that modulate influence?
  - Plasticity rules that update memory online?

- **How does agency emerge?**
  - When does “settling” become “choosing”?
  - What is the smallest environment loop that makes sense?

## 12) Guiding principles (the spirit of SATTVA)

Even before algorithms mature, SATTVA has a clear set of preferences:

- Prefer **state + dynamics** over pure sequence prediction.
- Prefer **stable semantics** over brittle pattern matching.
- Prefer **explicit geometry** (basins, trajectories, transitions) over opaque heuristics.
- Prefer **small testable simulations** before scaling.
- Prefer **clarity** (sattva) in docs, code, and experimental instrumentation.

---

If you want to extend this document later, a good next addition would be a section that proposes 2–3 candidate training objectives with explicit losses and what each objective would demonstrate.
