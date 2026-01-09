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

## 11) Critical insight: Geometric field theory (January 2026)

A major architectural clarification emerged during discussion of the implementation approach:

**SATTVA is not primarily a system of weighted connections between semantic vectors. It is a field-based system where information is encoded in the geometric configuration of activation patterns.**

### The tangled substrate insight

The brain's neural structure appears tangled because it contains many types of connections. The activation of one "memory" creates a geometric form out of that mess. Associated ideas/concepts are activated not by semantic similarity but by **geometric compatibility**—they form shapes that can constructively interfere.

### Key architectural consequences:

1. **Individual units don't "mean" anything**—meaning emerges from distributed activation patterns
2. **Geometric shape similarity enables coupling** even when patterns are semantically distant
3. **Long-range field effects** (inspired by Levin's bioelectric work) mean activation propagates further than local connectivity suggests
4. **Critical mass dynamics**: early learning appears unproductive until pattern density enables constructive interference (phase transition)
5. **Creativity = geometric resonance**: not randomness, but coupling of geometrically similar (semantically distant) patterns producing coherent novelty
6. **Depth dimension needed**: primitive/deep patterns have long-range influence; surface patterns are more local and plastic
7. **Trauma as forced deep encoding**: high-stress experiences pushed into primitive regions with their context, creating "sewer system" lateral connections

### Mathematical framing

Field dynamics where:
- State: activation levels \(u_i(t)\) at positions \(\mathbf{r}_i\) with depth \(d_i\)
- Field propagation: \(\phi(\mathbf{r}, t) = \sum_i u_i(t) \cdot K(\|\mathbf{r} - \mathbf{r}_i\|, d_i)\)
- Dynamics: \(\frac{du_i}{dt} = -\frac{\partial V}{\partial u_i} + \alpha \phi(\mathbf{r}_i, t) + \eta_i(t)\)
- Pattern matching: based on geometric shape/topology, not semantic vector distance

### Implementation implications

This is **fundamentally different** from standard neural networks:
- Not assuming pre-organized semantic space
- Geometric pattern similarity (topology, shape, moments) rather than vector cosine
- Long-range coupling via field effects
- Multi-column from day one (threads that assemble, not monolithic knowledge)
- Explicit depth/strength modeling

This requires thinking in terms of wave mechanics, field theory, holographic principles, and geometric computing rather than standard gradient descent on weight matrices.

---

## 12) Two-timescale regulation (January 2026 - Critical insight)

**User hypothesis on biological stability mechanism:**

Human brains prevent runaway resonance through two-timescale separation:

**Physical layer (slow):**
- Synaptic changes require repetition - minutes to days
- Acts as "speed limit" during pattern formation
- Metabolic constraints (ATP, neurotransmitters) limit rate
- Only patterns that survive repeated confirmation become stable

**Conceptual layer (fast):**
- Resonance between already-validated patterns can be fast (milliseconds)
- This is SAFE because patterns have been vetted
- Fast association permitted because operating over stable primitives
- "Valid" = survived sensory confirmation and prediction testing

**Why this prevents runaway:**

1. Initial formation is rate-limited by biology
2. Fast resonance over stable substrate can't create new unstable patterns
3. Separation of timescales breaks positive feedback loops

**Implementation implications:**
- Need two update rates: fast (activation/resonance) and slow (formation/learning)
- During development: only slow updates (forming primitives)
- During mature operation: fast updates over pre-validated patterns
- Anomaly regulation as backup when timescales insufficient

**Why our test exploded:**
- Everything on one timescale (fast)
- Pattern formation and resonance at same rate
- No validated primitive substrate
- Result: positive feedback at all scales

**Solution:**
1. Bootstrap with validated primitives first
2. Separate formation (slow, rare) from resonance (fast, continuous)
3. Timescale-dependent coupling strength

This explains:
- Why infants learn slowly (forming primitives, rate-limited)
- Why adult thinking is fast (resonating over stable patterns)
- Why trauma during development is persistent (written as primitive during slow formation)
- Why BDH's local passing works (implicitly slow, no explosions)
- Why SATTVA's long-range requires regulation (fast + long-range = unstable without substrate)

See: theory/two_timescale_regulation.md for full treatment.

---

## 13) Two-timescale regulation insight (January 8, 2026)

**Critical discovery about biological stability:**

Human brains prevent runaway resonance through two-timescale separation:

**Physical layer (slow - hours/days):**
- Synaptic changes require repetition
- Acts as "speed limit" during pattern formation
- Metabolic constraints limit rate
- Only validated patterns become stable

**Conceptual layer (fast - milliseconds):**
- Resonance between already-validated patterns
- Safe because patterns have been vetted through slow formation
- Fast association permitted over stable primitive substrate

**Why this prevents runaway:**
1. Formation is rate-limited (can't create explosive feedback during learning)
2. Fast resonance over stable substrate is safe (validated patterns unlikely catastrophic)
3. Timescale separation breaks positive feedback loops

**Implementation:**
- Two update rates: fast (activation) vs slow (formation)
- Development phase: only slow updates (forming primitives)
- Mature phase: fast updates over pre-validated patterns
- Anomaly regulation as backup

**Why our test exploded:** Everything on one timescale, no validated substrate.

**Solution:** Bootstrap with primitives first, separate formation from resonance.

Full treatment in: `theory/two_timescale_regulation.md`

---

## 14) Open questions (where the interesting work is)

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
