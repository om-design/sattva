# SATTVA — A theoretical development

## Abstract

SATTVA (Semantic Attractor Training of Transforming Vector Associations) is a research program for building artificial systems whose core cognitive substrate is dynamical: internal states evolve, resonate, and settle into stable semantic attractors.

The central claim is that “understanding” can be reframed as a property of an internal landscape: (1) the shape of attractor basins (stable meanings), (2) the trajectories that lead into them (inference/recognition), and (3) the transitions between them (association, planning, creativity).

SATTVA further claims two practical requirements for scalable cognition: (1) many semi-independent “columns” must settle in parallel and reach consensus (in the spirit of Numenta’s Thousand Brains theory), and (2) the system needs continuous anomaly-based self-regulation (in the spirit of HTM anomaly detection as online prediction failure) to prevent runaway dynamics while still permitting creative divergence when desired.

Finally, SATTVA emphasizes *structural creativity*: not just generating diverse samples, but forming genuinely new, reusable semantic basins (new attractors) via controlled divergence followed by consolidation.

## Key novelties

- Multi-column semantic settling with consensus dynamics (Thousand Brains as a guiding analogy).
- Task-conditioned anomaly regulation: anomaly can mean “explore” or “stabilize,” depending on goal and context.
- Creative basin formation: novelty is validated by stability, reuse, and downstream competence, not just surface diversity.

## Comparative analysis

This section frames SATTVA’s control-and-creativity proposal in contrast to HC and mHC (Hyper-Connections and Manifold-Constrained Hyper-Connections), which address instability in very deep networks by constraining how signals mix across depth.

### What HC is solving

HC broadens the residual pathway into multiple parallel streams and uses learnable mixing to route information across depth, which can increase expressivity but can also amplify signals and destabilize training at scale.

### What mHC changes

mHC stabilizes HC by projecting residual mixing matrices onto the manifold of doubly stochastic matrices (the Birkhoff polytope), using a Sinkhorn–Knopp style projection so that residual mixing becomes a convex combination and preserves stable signal propagation through depth. [web:796][web:802][web:804]

In plain terms, HC and mHC treat “runaway amplification” as the core failure mode, and mHC fixes it by forcing the “mixing” geometry to behave conservatively (roughly: preserve magnitude) even in very deep stacks.

### How SATTVA differs

SATTVA treats divergence as sometimes desirable (creative search) and sometimes dangerous (loss of coherence), so it adds a task-conditioned regulator that can continuously trade off exploration and stabilization rather than enforcing a single always-on conservation rule.

Instead of making the network always behave as if gain must remain near 1 (as with mHC), SATTVA aims to permit controlled “excursions” (high novelty / high variance internal trajectories) and then consolidate them into stable, reusable attractors when the task demands commitment.

This is a more analog solution: the system can be intentionally permissive during a creative phase and intentionally strict during an execution/safety phase, with the switch governed by an online anomaly signal over internal dynamics.

### Where this is genuinely novel - actual machine creativity

The novelty claim is not merely “more diverse output.” It is that novelty can be *structural*: new basins become stable objects in the system’s semantic geometry, meaning the system can revisit, refine, and reliably use them later.

---

## 1. Motivation: resonance as an engineering hypothesis

SATTVA begins with an intuition: cognition is not only representation; it is resonant dynamics. A mind is not a static map of the world; it is a process that stabilizes certain patterns, suppresses others, and transitions between stable states in response to cues and goals.

“Resonance” is used here in a disciplined, engineering sense:

- Amplification: small cues can recruit large internal structure (partial pattern → full recall).
- Stability: some states persist against noise (a concept remains identifiable as context shifts).
- Selectivity: only certain patterns stabilize given the current internal landscape.
- Coupling: multiple subsystems synchronize into coherent joint states.

The promise is that these properties can be built, measured, and trained.

---

## 2. Core definitions

### 2.1 State, dynamics, and attractors

SATTVA models an agent as a dynamical system with internal state \(x\) evolving in time:

- State: a vector (or structured set of vectors) representing current internal configuration.
- Dynamics: an update rule \(x_{t+1} = f(x_t, u_t; \theta)\) where \(u_t\) is input and \(\theta\) are learnable parameters.
- Attractor: a set of states toward which trajectories converge from a basin of initial conditions.

A practical SATTVA stance:

- A concept is not a single vector; it is a stable region (or set) of state space.
- Recognition is convergence (settling) toward that region.
- Reasoning/thought is controlled transition between such regions.

### 2.2 Semantic vectors and associations

“Semantic vectors” are any learned or engineered vector representations that preserve meaning-relevant geometry (neighborhoods, directions, clusters).

“Vector associations” are the learned influences between vectors—how activation in one region shapes motion in another. In SATTVA, these associations are the central learnable substrate.

### 2.3 Transforming associations

Transforming Vector Associations emphasizes plasticity:

- Associations can be updated by experience.
- Associations can be contextual (gated by attention or state).
- Associations can be multi-scale (local within a modality; global across modalities).

This “transforming” quality is the difference between a static associative memory and a developmental system.

---

## 3. SATTVA as an Attractor-based Understanding Machine (AUM)

SATTVA can be viewed as an implementation path for an Attractor-based Understanding Machine (AUM).

In token-prediction systems, cognition is often framed as next-step selection in a sequence. In SATTVA, cognition is framed as landscape shaping:

- Competence depends on the geometry of basins and on reliable transitions.
- The unit of cognition is not the token; it is the settled state.

This is not an argument against sequence models; it is a proposal that “understanding” can be engineered at a different layer: stable semantic states and their transitions.

---

## 4. Resonant brain fields (analogy)

The phrase “resonant brain fields” is used as an organizing analogy for three broad observations about brains:

1. Brains exhibit widespread rhythmic activity across frequencies.
2. Coordination often occurs through phase relationships and coupling (synchrony/coherence).
3. Cognition involves both stable patterns and rapid transitions.

SATTVA does not claim biological equivalence, but it uses the analogy to motivate architectural preferences:

- Coordination should be expressible as coupling between subspaces/modules.
- Stable semantic states should emerge as attractors supported by internal “field-like” interactions.

A practical translation:

- Treat the recurrent core as a learnable field over state space (a rule that shapes flow).
- Allow multiple modules (vision-like, language-like, body-like) to couple via shared latent fields or controlled cross-influences.
- Measure coherence: when cues align, modules should settle into compatible attractors.

---

## Numenta: HTM and Thousand Brains

Numenta’s work is directly relevant to SATTVA because it supplies two complementary ideas: (1) a column-centric view of intelligence (Thousand Brains), and (2) a practical, deployable notion of anomaly detection as *online prediction failure* (HTM), which can act as a noninvasive “health monitor” for any dynamical process.

### Thousand Brains as multi-model consensus

The Thousand Brains Theory proposes that many cortical columns learn models of complete objects and reach a perceptual consensus through a voting-like process across columns.

SATTVA’s “concept columns” can be made more explicit in this light: instead of a single monolithic attractor space, build many semi-independent subspaces (columns/modules) that each settle toward their own attractors, then couple them so the global state converges toward a consistent joint interpretation.

AUM interpretation: “understanding” becomes the stable fixed point of a coupled multi-column system, and ambiguity corresponds to competing basins that have not yet reached consensus.

### HTM anomaly detection as online stability control

Numenta’s anomaly work frames anomalies as moments when a system’s predictions fail: the model’s predictive error is converted into an anomaly score, and then smoothed/normalized into an anomaly likelihood to reduce false positives in noisy streams.

Numenta also created the Numenta Anomaly Benchmark (NAB) to evaluate streaming anomaly detection, emphasizing online operation, continuous learning, adaptation to changing “normal,” and early detection.

SATTVA can import this not only as a time-series tool, but as a meta-control principle:

- Monitor the internal settling dynamics (or trajectory predictions) and compute an “anomaly” signal when behavior deviates from learned expectations.
- Use that signal to trigger corrective mechanisms that “reign the system back in” (increase damping, reduce exploration temperature, gate plasticity, reset to a safe prior basin, or recruit additional columns for disambiguation).

In other words, anomaly detection becomes a quantitative proxy for “loss of sattva” (loss of clarity/balance/coherence), enabling closed-loop stability control over cognition.

### Task-conditioned anomaly (surprise vs danger)

HTM anomaly detection is a general-purpose signal: it quantifies prediction failure in an online stream.

SATTVA adds an important distinction: anomaly is not always “bad.” It can be interpreted differently depending on goal state and phase of problem solving:

- Epistemic surprise: a cue is unexpected, so the system should explore (creative mode).
- Dynamical danger: behavior is unexpected *and* destabilizing, so the system should constrain itself (safety mode).

This creates an analog alternative to hard “collapse-to-one” mechanisms: anomaly becomes a continuous control signal whose meaning is task-tuned.

### Creative mode (diverge then consolidate)

A practical SATTVA creativity loop:

1. Divergence: allow higher exploration (temperature/noise), higher cross-column mixing, and broader basin accessibility.
2. Evaluation: require local coherence across columns (partial consensus) even during divergence.
3. Consolidation: gradually tighten constraints (increase damping, reduce exploration, strengthen consensus coupling) until the system commits to a stable, usable basin.

This makes novelty operational: a creative episode is permitted to be “wild” early, but must eventually stabilize into a reusable attractor to become knowledge.

---

## 5. Bioelectric pattern memory (Levin-lab bridge)

A major reinforcement for SATTVA’s framing comes from developmental bioelectricity and the study of morphogenesis as a kind of basal cognition.

### 5.1 Why this matters to SATTVA

The key conceptual bridge is this: biological systems show that stable “pattern memories” can be stored and edited in distributed networks that are not brains, and that these memories guide system-level outcomes.

This supports an important SATTVA move:

- The substrate of “memory-like attractors” does not have to be a classical neural network.
- A dynamical field can store goals, constraints, and stable setpoints that guide behavior.

### 5.2 Pattern memory as attractor landscapes

A useful reading of the bioelectric literature (especially around planaria) is that tissues can exhibit:

- long-range coordination,
- stable target morphologies,
- rewriting of those targets via transient interventions,
- persistence of the rewritten target across later regeneration events.

Interpreted computationally, this resembles:

- distributed storage,
- attractor basins corresponding to target configurations,
- and editing operations that reshape basin structure.

SATTVA’s analog is not “grow an eye,” but:

- store stable semantic targets,
- rewrite them through experience,
- and use them to guide settling and action.

### 5.3 AUM implication

If morphogenesis can be described as collective intelligence navigating a problem space (morphospace), then cognition can be described as a system navigating a semantic/action problem space. SATTVA’s claim is that attractor geometry is a shared mathematical language for both.

### 5.4 Note on “Kreinen’s discoveries”

Our discussion referenced “Fenna Kreinen’s discoveries” in this bioelectric/collective-intelligence neighborhood.

Specifically, the seed thought from Kreinen is the discovery of fractal structures within the brain rising from the brain stem through to complex formations. The similarity allows for storage and transmission between primitive (foundational) structures and complex (reasoning) structures depending on the conditions and strength of impulse.

---

## 6. Structural encoding: analogies to vision

Vision is the single most highly evolved and elegantly simplistic feature of life on Earth. From all living cells some form of recognition and interaction with light is the fundamental principle and it serves as a strong metaphor for sattva because it is simultaneously:

- geometric (edges, contours, surfaces),
- hierarchical (local features → global objects),
- constraint-based (infer structure under uncertainty),
- action-coupled (vision is for behavior).

SATTVA borrows a principle: structure emerges from constrained settling.

### 6.1 From pixels to meaning: a settling story

In vision, raw input is transformed through stages:

- local feature extraction,
- grouping/segmentation,
- stable object hypotheses.

This can be recast dynamically:

- competing hypotheses interact,
- consistent structure is amplified,
- the system stabilizes on a coherent interpretation.

SATTVA generalizes this story:

- any input (text, sensory features, internal goals) becomes a cue,
- the cue seeds internal dynamics,
- the system settles into a stable semantic hypothesis.

### 6.2 Structure–function coupling

Vision illustrates that “structure” and “function” are inseparable:

- percepts are shaped by what actions they support.

SATTVA imports this as a constraint:

- attractors are not “good” because they exist; they are good because they support inference, memory, and action.

Therefore, attractor training must be coupled to tasks, not just reconstruction.

---

## 7. The SATTVA loop (formalized)

A minimal SATTVA loop:

1. Encode: map inputs \(u\) into an initial semantic state \(x_0 = E(u)\).
2. Settle: apply recurrent dynamics \(x_{t+1} = f(x_t; \theta)\) (or \(f(x_t, u_t; \theta)\)) until convergence or stopping.
3. Read out: interpret final state \(y = R(x_T)\) (label, memory, prediction, action proposal).
4. Train: adjust \(\theta\), and possibly \(E\) and \(R\), to make attractors and transitions useful.

This loop prevents the “demo trap,” because it forces information to flow and be evaluated.

---

## 8. Sattva as a design constraint (clarity, balance, coherence)

Use *sattva* as a design target.

### 8.1 Clarity

- attractors are identifiable,
- basins are separated,
- settled states are stable and interpretable.

### 8.2 Balance

- stable without being frozen,
- responsive without being chaotic,
- able to shift basins when evidence changes.

### 8.3 Coherence

- related cues converge to compatible attractors,
- unrelated cues remain separated,
- compositions integrate rather than collapse.

### 8.4 Instrumentation (early metrics)

- Trivial collapse detection: frequency of convergence to degenerate states (e.g., near-zero everywhere).
- Basin separation: distances between attractor states; confusion under noise.
- Robustness: recall from partial/noisy cues.
- Transition predictability: cue sequences move reliably between intended basins.
- Anomaly of internal dynamics: prediction-failure signals over settling trajectories, used to trigger either stabilization or exploratory creativity depending on task.

---

## 9. Layering: chemistry as a “metaphorical function layer” above deep memories

Our discussion introduced a layered metaphor:

- Deep attractors correspond to long-lived “knowledge states” (including trauma-like basins).
- Above them exists a faster, modulatory layer analogous to “chemistry”: interaction rules that gate which basins become reachable, how quickly settling occurs, and which transitions are favored.

This is a useful metaphor because chemical interaction has properties that map well to cognitive gating:

- **Catalysis**: some conditions dramatically increase the rate of specific transitions.
- **Inhibition**: some conditions suppress transitions even if they are otherwise “energetically allowed.”
- **Mixture**: multiple modulators combine nonlinearly (not just additive).
- **State-dependence**: the same molecule can have different effects in different contexts.

In SATTVA terms, the “chemical layer” can be implemented as:

- global or local gain control,
- neuromodulator-like gating vectors,
- context-dependent association matrices,
- temperature schedules,
- or learned routing functions.

The theoretical point is not the implementation detail; it is the existence of a separable layer that can:

- tune plasticity (when associations transform),
- tune stability (how deep basins are),
- tune creativity (how readily mixed states explore new regions).

---

## 10. Trauma as an attractor metaphor (careful and useful)

This section treats trauma as a metaphor for maladaptive dynamical shaping, not a clinical model.

In attractor terms, trauma-like dynamics suggest:

- Certain patterns become too deep (over-stable): the system returns to them too easily.
- The landscape becomes narrow: fewer accessible states; reduced flexibility.
- Transitions become biased: small cues trigger rapid settling into defensive basins.

Engineering takeaway:

- Unbalanced training signals can create brittle attractors.
- Robustness includes maintaining a healthy diversity of reachable states.

This motivates:

- resilience metrics (ability to leave basins when evidence contradicts them),
- constraints preventing pathological deepening,
- and relearning rules that reshape basins without catastrophic forgetting.

---

## 11. Resonant columns and creativity (strings, harmonics, and new basins)

A central question in the discussion is: how can a system built from attractors do more than recall—how can it become creative?

SATTVA’s answer is to treat a concept not as a point, but as a *column* (or fiber) spanning multiple coupled subspaces. For example:

- sensory structure (what it looks/sounds like),
- relational structure (what it implies and associates with),
- functional structure (what it affords),
- affective structure (valence/arousal signatures),
- action structure (policies or tendencies).

A “concept column” is a structured set of coupled states that can resonate together morphologically across the entire domain, allowing association from among complex networks that are intertwined.

### 11.1 Resonant string metaphor

The resonant-string metaphor becomes useful if kept precise:

- A string supports many modes (fundamental + harmonics).
- A cue can excite a subset of modes.
- Coupled strings can entrain and exchange energy.

Translated to SATTVA:

- A cue excites part of a concept column.
- The attractor core amplifies consistent structure and dampens inconsistent structure.
- Multiple concept columns can partially synchronize.

### 11.2 Creativity as controlled new attractor formation

Creativity can be modeled as the creation (or discovery) of new stable basins produced by constructive interference:

- When two or more concept columns overlap coherently in some subspaces but not others, the system can settle into an intermediate state.
- If training and gating allow that intermediate to stabilize (without collapsing to a parent attractor), it becomes a new attractor.

This suggests a concrete research hypothesis:

- Creativity corresponds to the emergence of *novel basins* that are reachable, stable enough to be used, and still connected to their ancestors by meaningful transition paths.

### 11.3 The role of the “chemical layer” in creativity

The modulatory/chemical layer is the control knob for creativity:

- High stability + low exploration → strong recall, low novelty.
- Lower stability + higher coupling/exploration → more novelty, but risk of incoherence.

The goal is not maximal novelty; it is stable novelty: basins that are new yet usable.

### 11.4 Creative novelty (strong claim)

SATTVA’s strongest creativity claim is structural: a system is “creative” when it can reliably construct new stable basins (new attractors) that are:

- Novel: not a minor perturbation of a parent basin.
- Stable: reproducible across noise and revisitable later.
- Integrated: coherent across multiple coupled subspaces (“concept columns”).
- Useful: supports downstream competence (retrieval, reasoning, planning, action).

This definition is intentionally stricter than “diverse generation,” because it requires persistence and functional integration.

---

## 12. Candidate training objectives (research anchors)

### 12.1 Supervised settling (first milestone)

Goal: from a degraded cue, settle into the intended semantic state.

- Input: noisy/partial cue \(x_0\).
- Target: clean vector \(v\) (or label whose prototype is \(v\)).
- Loss: \(\|x_T - v\|^2\) plus regularizers.

Regularizers:

- penalize trivial collapse,
- encourage basin separation,
- encourage smooth but decisive settling.

### 12.2 Association chains (controlled transitions)

Goal: one attractor leads to another under contextual drive.

- Inputs: cue sequences.
- Targets: settled-state sequences.
- Loss: distance to target sequence; penalties for chaotic wandering.

### 12.3 Compositional settling

Goal: multiple cues settle into an integrated state.

- Input: mixture/set of cues.
- Target: composed representation (learned or defined).

This is a bridge from memory toward “understanding,” because it tests integration, not just recall.

### 12.4 Novel basin formation (creativity objective)

Goal: enable stable new basins under controlled conditions.

A first measurable version:

- Present paired (or triad) cues whose joint interpretation is not explicitly stored.
- Require the model to settle into a stable state that:
  - is not too close to either parent attractor,
  - is reproducible across noise,
  - supports correct downstream readouts (classification, retrieval, or action).

### 12.5 Anomaly-based regulation (keeping sattva)

Goal: detect when internal dynamics are becoming unstable, brittle, or “spiraling,” and apply noninvasive control to restore a healthy operating regime.

A first measurable version:

- Train a predictor over settling trajectories (or next-step internal states) and compute an anomaly score from prediction error, following the HTM framing.
- Convert the anomaly stream into a smoothed likelihood (or other false-positive control), so the regulator is not hair-trigger under noise.
- Interpret anomaly using task context:
  - In creative mode, treat elevated anomaly as permission to explore (within coherence constraints).
  - In safety mode, treat elevated anomaly as a trigger to constrain dynamics.
- When anomaly crosses a regime threshold, apply a control action such as:
  - increase damping / reduce gain,
  - lower exploration temperature,
  - temporarily freeze plasticity,
  - reset state toward a designated safe basin,
  - or request additional evidence / recruit additional columns.
- Evaluate success by whether the system returns to coherent settling without collapsing into trivial attractors.

### 12.6 Divergence and consolidation (creative loop)

Goal: formalize “wild exploration → stable commitment” as a learnable controller rather than an ad-hoc sampling trick.

A first measurable version:

- Define a creative episode with phases (diverge, evaluate, consolidate).
- Provide an explicit schedule or a learned policy that modulates:
  - exploration (temperature/noise),
  - damping/gain,
  - column coupling strength,
  - and plasticity.
- Success criteria:
  - The system visits diverse candidate basins during divergence.
  - Cross-column coherence increases over time (consensus emerges).
  - The final settled basin is reproducible and supports downstream tasks.

---

## 13. Research program and near-term experiments

1. Toy vector spaces: diagnose collapse; measure basin separation.
2. Real embeddings: test robustness and confusion.
3. Task-coupled training: supervised settling + regularizers.
4. Hierarchical attractors: coarse/fine basins.
5. Field coupling between modules: e.g., “visual” and “linguistic” spaces coupled; test coherence.
6. Agent loop: tiny environment; settled states propose actions; reward shapes the landscape.

Methodological preference:

- do not scale until failure modes are visible and measurable.

---

## 14. Falsifiability

SATTVA makes testable commitments:

- If viable, training produces non-trivial attractor landscapes improving recall, association, and composition under noise.
- If not viable, dynamics collapse to trivial attractors, remain chaotic, or fail to generalize.

This is falsifiable at small scale:

- a few dozen concepts with simple tasks should show measurable improvement if the hypothesis has merit.

---

## 15. Conclusion

SATTVA proposes that “understanding” can be engineered as the geometry of stable semantic attractors and controlled transitions between them. It uses resonance and brain-field analogies as conceptual constraints, vision as a model of structural settling and structure–function coupling, bioelectric pattern memory as a motivating bridge, chemistry as a metaphor for modulatory gating above deep attractors, and trauma as a cautionary metaphor for maladaptive basin shaping.

The immediate next step is to continue evolving the first task-driven training objective and instrumentation so the project evolves from a dynamical demo into a learnable semantic structure. This can be explored in the Experiments directory.
