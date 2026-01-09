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

## Pathway's BDH (Baby Dragon Hatchling): Post-Transformer Brain-Like Architecture

Zuzanna Stamirowska and the Pathway team have developed BDH (Baby Dragon Hatchling), a post-Transformer architecture that demonstrates many principles central to SATTVA's vision and validates that brain-inspired architectures are practically viable.

### Core BDH principles aligned with SATTVA

**Emergent neural structure during training:**
BDH doesn't start with fixed architecture. Neural structures and connection patterns literally emerge during training from local message-passing rules. The team observed "the emergence of just this kind of brain appearing" in their lab - organization arises from simple interaction rules, not pre-designed topology. This directly validates SATTVA's "tangled substrate" principle where meaning emerges from activation geometry rather than pre-structured organization.

**Message passing between locally connected neurons:**
Instead of all-to-all attention (Transformers) or fixed weight matrices, BDH uses brain-like message passing: "one neuron gets information, passes it on to its neighbors, those with whom he's connected." Only neurons that "care" about incoming information light up and propagate signals. This is much closer to SATTVA's field propagation model than standard matrix multiplication.

**Hebbian learning and memory as strengthened connections:**
"Whenever two neurons were interested by something, the connection between them becomes stronger and this is memory." Connections that co-activate are reinforced; unused connections fade over time. This is exactly SATTVA's "transforming vector associations" - associations form, strengthen, and weaken based on co-activation patterns and usage.

**Scale-free biological network structure:**
BDH's neuron interaction network is explicitly a scale-free graph with heavy-tailed degree distribution and high modularity. The architecture is mathematically characterized as uniform and scale-free, meaning the same laws hold at different scales. This supports SATTVA's fractal brain structure concept - self-similar patterns at different hierarchical levels.

**True memory and continual learning:**
Unlike Transformers which "wake up with no memory" each inference (living in "Groundhog Day"), BDH maintains persistent memory through strengthened synaptic connections. It can learn continuously, adapt to new data over time, and maintain infinite context. This aligns with SATTVA's attractor basins as reusable, evolving semantic structures.

**Interpretability through sparse activation:**
BDH exhibits monosemanticity - specific neurons fire for specific abstract concepts (currencies, countries, etc.). Activation vectors are sparse and positive. The team describes having "CCTV inside the brain" rather than needing to build "MRI machines" to scan black-box models. This relates to SATTVA's geometric pattern matching - distinct activation shapes for distinct concepts.

**Neurons "getting bored" (habituation):**
When repeatedly exposed to the same information, BDH neurons reduce their activation response - they adapt and habituate. This is analogous to biological novelty detection and connects to SATTVA's anomaly-based regulation: the system modulates its response based on whether information is surprising or routine.

**Production deployment:**
BDH is not just a research prototype - it's being deployed for Formula 1, NATO, French Postal Service, and other real-world applications. This demonstrates that post-Transformer brain-inspired architectures are practically viable at scale.

### Where BDH validates and extends SATTVA's direction

1. **Emergent structure works**: You don't need to pre-specify semantic organization; it can arise from local interaction rules
2. **Message passing + connection strengthening** is a viable alternative to Transformers
3. **Memory and reasoning** are separable from language modeling per se
4. **Scale-free networks** enable generalization across scales
5. **Biological plausibility** doesn't sacrifice performance

### Where SATTVA extends beyond BDH

While BDH is a major validation, SATTVA adds:
- **Geometric shape similarity** as explicit coupling mechanism (not just connection strength)
- **Depth dimension** for primitive vs. complex patterns with different coupling ranges
- **Long-range field effects** beyond local message passing
- **Geometric resonance-based creativity** (coupling semantically distant but geometrically similar patterns)
- **Explicit developmental hierarchy** (sensory baselining → geometric primitives → compositional meaning)
- **Task-conditioned anomaly regulation** (epistemic surprise vs. dynamical danger)

**Credit and acknowledgment:**
SATTVA's geometric field theory approach was developed independently but is strongly validated by Pathway's successful deployment of BDH. We acknowledge Zuzanna Stamirowska, Adrian Kosowski, Jan Chorowski, Przemysław Uznański, Michał Bartoszkiewicz, and the Pathway team for demonstrating that emergent brain-like architectures with message passing, Hebbian learning, and scale-free structure can rival Transformer performance while offering superior interpretability, continual learning, and memory capabilities. Their work provides crucial evidence that the principles underlying SATTVA are not merely theoretical but can be successfully implemented and deployed.

**References:**
- Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., & Bartoszkiewicz, M. (2025). The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507
- [Pathway AI](https://pathway.com/)

---

## Pathway's BDH (Baby Dragon Hatchling): Post-Transformer Brain-Like Architecture

Zuzanna Stamirowska and the Pathway team have developed BDH (Baby Dragon Hatchling), a post-Transformer architecture that demonstrates many principles central to SATTVA's vision and validates that brain-inspired architectures are practically viable.

### Core BDH principles aligned with SATTVA

**Emergent neural structure during training:**
BDH doesn't start with fixed architecture. Neural structures and connection patterns literally emerge during training from local message-passing rules. The team observed "the emergence of just this kind of brain appearing" in their lab - organization arises from simple interaction rules, not pre-designed topology. This directly validates SATTVA's "tangled substrate" principle where meaning emerges from activation geometry rather than pre-structured organization.

**Message passing between locally connected neurons:**
Instead of all-to-all attention (Transformers) or fixed weight matrices, BDH uses brain-like message passing: "one neuron gets information, passes it on to its neighbors, those with whom he's connected." Only neurons that "care" about incoming information light up and propagate signals. This is much closer to SATTVA's field propagation model than standard matrix multiplication.

**Hebbian learning and memory as strengthened connections:**
"Whenever two neurons were interested by something, the connection between them becomes stronger and this is memory." Connections that co-activate are reinforced; unused connections fade over time. This is exactly SATTVA's "transforming vector associations" - associations form, strengthen, and weaken based on co-activation patterns and usage.

**Scale-free biological network structure:**
BDH's neuron interaction network is explicitly a scale-free graph with heavy-tailed degree distribution and high modularity. The architecture is mathematically characterized as uniform and scale-free, meaning the same laws hold at different scales. This supports SATTVA's fractal brain structure concept - self-similar patterns at different hierarchical levels.

**True memory and continual learning:**
Unlike Transformers which "wake up with no memory" each inference (living in "Groundhog Day"), BDH maintains persistent memory through strengthened synaptic connections. It can learn continuously, adapt to new data over time, and maintain infinite context. This aligns with SATTVA's attractor basins as reusable, evolving semantic structures.

**Interpretability through sparse activation:**
BDH exhibits monosemanticity - specific neurons fire for specific abstract concepts (currencies, countries, etc.). Activation vectors are sparse and positive. The team describes having "CCTV inside the brain" rather than needing to build "MRI machines" to scan black-box models. This relates to SATTVA's geometric pattern matching - distinct activation shapes for distinct concepts.

**Neurons "getting bored" (habituation):**
When repeatedly exposed to the same information, BDH neurons reduce their activation response - they adapt and habituate. This is analogous to biological novelty detection and connects to SATTVA's anomaly-based regulation: the system modulates its response based on whether information is surprising or routine.

**Production deployment:**
BDH is not just a research prototype - it's being deployed for Formula 1, NATO, French Postal Service, and other real-world applications. This demonstrates that post-Transformer brain-inspired architectures are practically viable at scale.

### Where BDH validates and extends SATTVA's direction

1. **Emergent structure works**: You don't need to pre-specify semantic organization; it can arise from local interaction rules
2. **Message passing + connection strengthening** is a viable alternative to Transformers
3. **Memory and reasoning** are separable from language modeling per se
4. **Scale-free networks** enable generalization across scales
5. **Biological plausibility** doesn't sacrifice performance

### Where SATTVA extends beyond BDH

While BDH is a major validation, SATTVA adds:
- **Geometric shape similarity** as explicit coupling mechanism (not just connection strength)
- **Depth dimension** for primitive vs. complex patterns with different coupling ranges
- **Long-range field effects** beyond local message passing
- **Geometric resonance-based creativity** (coupling semantically distant but geometrically similar patterns)
- **Explicit developmental hierarchy** (sensory baselining → geometric primitives → compositional meaning)
- **Task-conditioned anomaly regulation** (epistemic surprise vs. dynamical danger)

**Credit and acknowledgment:**
SATTVA's geometric field theory approach was developed independently but is strongly validated by Pathway's successful deployment of BDH. We acknowledge Zuzanna Stamirowska, Adrian Kosowski, Jan Chorowski, Przemysław Uznański, Michał Bartoszkiewicz, and the Pathway team for demonstrating that emergent brain-like architectures with message passing, Hebbian learning, and scale-free structure can rival Transformer performance while offering superior interpretability, continual learning, and memory capabilities. Their work provides crucial evidence that the principles underlying SATTVA are not merely theoretical but can be successfully implemented and deployed.

**References:**
- Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., & Bartoszkiewicz, M. (2025). The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507
- [Pathway AI](https://pathway.com/)

---

## Pathway's BDH (Baby Dragon Hatchling): Post-Transformer Brain-Like Architecture

Zuzanna Stamirowska and the Pathway team have developed BDH (Baby Dragon Hatchling), a post-Transformer architecture that demonstrates many principles central to SATTVA's vision and validates that brain-inspired architectures are practically viable.

### Core BDH principles aligned with SATTVA

**Emergent neural structure during training:**
BDH doesn't start with fixed architecture. Neural structures and connection patterns literally emerge during training from local message-passing rules. The team observed "the emergence of just this kind of brain appearing" in their lab - organization arises from simple interaction rules, not pre-designed topology. This directly validates SATTVA's "tangled substrate" principle where meaning emerges from activation geometry rather than pre-structured organization.

**Message passing between locally connected neurons:**
Instead of all-to-all attention (Transformers) or fixed weight matrices, BDH uses brain-like message passing: "one neuron gets information, passes it on to its neighbors, those with whom he's connected." Only neurons that "care" about incoming information light up and propagate signals. This is much closer to SATTVA's field propagation model than standard matrix multiplication.

**Hebbian learning and memory as strengthened connections:**
"Whenever two neurons were interested by something, the connection between them becomes stronger and this is memory." Connections that co-activate are reinforced; unused connections fade over time. This is exactly SATTVA's "transforming vector associations" - associations form, strengthen, and weaken based on co-activation patterns and usage.

**Scale-free biological network structure:**
BDH's neuron interaction network is explicitly a scale-free graph with heavy-tailed degree distribution and high modularity. The architecture is mathematically characterized as uniform and scale-free, meaning the same laws hold at different scales. This supports SATTVA's fractal brain structure concept - self-similar patterns at different hierarchical levels.

**True memory and continual learning:**
Unlike Transformers which "wake up with no memory" each inference (living in "Groundhog Day"), BDH maintains persistent memory through strengthened synaptic connections. It can learn continuously, adapt to new data over time, and maintain infinite context. This aligns with SATTVA's attractor basins as reusable, evolving semantic structures.

**Interpretability through sparse activation:**
BDH exhibits monosemanticity - specific neurons fire for specific abstract concepts (currencies, countries, etc.). Activation vectors are sparse and positive. The team describes having "CCTV inside the brain" rather than needing to build "MRI machines" to scan black-box models. This relates to SATTVA's geometric pattern matching - distinct activation shapes for distinct concepts.

**Neurons "getting bored" (habituation):**
When repeatedly exposed to the same information, BDH neurons reduce their activation response - they adapt and habituate. This is analogous to biological novelty detection and connects to SATTVA's anomaly-based regulation: the system modulates its response based on whether information is surprising or routine.

**Production deployment:**
BDH is not just a research prototype - it's being deployed for Formula 1, NATO, French Postal Service, and other real-world applications. This demonstrates that post-Transformer brain-inspired architectures are practically viable at scale.

### Where BDH validates and extends SATTVA's direction

1. **Emergent structure works**: You don't need to pre-specify semantic organization; it can arise from local interaction rules
2. **Message passing + connection strengthening** is a viable alternative to Transformers
3. **Memory and reasoning** are separable from language modeling per se
4. **Scale-free networks** enable generalization across scales
5. **Biological plausibility** doesn't sacrifice performance

### Where SATTVA extends beyond BDH

While BDH is a major validation, SATTVA adds:
- **Geometric shape similarity** as explicit coupling mechanism (not just connection strength)
- **Depth dimension** for primitive vs. complex patterns with different coupling ranges
- **Long-range field effects** beyond local message passing
- **Geometric resonance-based creativity** (coupling semantically distant but geometrically similar patterns)
- **Explicit developmental hierarchy** (sensory baselining → geometric primitives → compositional meaning)
- **Task-conditioned anomaly regulation** (epistemic surprise vs. dynamical danger)

**Credit and acknowledgment:**
SATTVA's geometric field theory approach was developed independently but is strongly validated by Pathway's successful deployment of BDH. We acknowledge Zuzanna Stamirowska, Adrian Kosowski, Jan Chorowski, Przemysław Uznański, Michał Bartoszkiewicz, and the Pathway team for demonstrating that emergent brain-like architectures with message passing, Hebbian learning, and scale-free structure can rival Transformer performance while offering superior interpretability, continual learning, and memory capabilities. Their work provides crucial evidence that the principles underlying SATTVA are not merely theoretical but can be successfully implemented and deployed.

**References:**
- Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., & Bartoszkiewicz, M. (2025). The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507
- [Pathway AI](https://pathway.com/)

---

## Pathway's BDH (Baby Dragon Hatchling): Post-Transformer Brain-Like Architecture

Zuzanna Stamirowska and the Pathway team have developed BDH (Baby Dragon Hatchling), a post-Transformer architecture that demonstrates many principles central to SATTVA's vision and validates that brain-inspired architectures are practically viable.

### Core BDH principles aligned with SATTVA

**Emergent neural structure during training:**
BDH doesn't start with fixed architecture. Neural structures and connection patterns literally emerge during training from local message-passing rules. The team observed "the emergence of just this kind of brain appearing" in their lab - organization arises from simple interaction rules, not pre-designed topology. This directly validates SATTVA's "tangled substrate" principle where meaning emerges from activation geometry rather than pre-structured organization.

**Message passing between locally connected neurons:**
Instead of all-to-all attention (Transformers) or fixed weight matrices, BDH uses brain-like message passing: "one neuron gets information, passes it on to its neighbors, those with whom he's connected." Only neurons that "care" about incoming information light up and propagate signals. This is much closer to SATTVA's field propagation model than standard matrix multiplication.

**Hebbian learning and memory as strengthened connections:**
"Whenever two neurons were interested by something, the connection between them becomes stronger and this is memory." Connections that co-activate are reinforced; unused connections fade over time. This is exactly SATTVA's "transforming vector associations" - associations form, strengthen, and weaken based on co-activation patterns and usage.

**Scale-free biological network structure:**
BDH's neuron interaction network is explicitly a scale-free graph with heavy-tailed degree distribution and high modularity. The architecture is mathematically characterized as uniform and scale-free, meaning the same laws hold at different scales. This supports SATTVA's fractal brain structure concept - self-similar patterns at different hierarchical levels.

**True memory and continual learning:**
Unlike Transformers which "wake up with no memory" each inference (living in "Groundhog Day"), BDH maintains persistent memory through strengthened synaptic connections. It can learn continuously, adapt to new data over time, and maintain infinite context. This aligns with SATTVA's attractor basins as reusable, evolving semantic structures.

**Interpretability through sparse activation:**
BDH exhibits monosemanticity - specific neurons fire for specific abstract concepts (currencies, countries, etc.). Activation vectors are sparse and positive. The team describes having "CCTV inside the brain" rather than needing to build "MRI machines" to scan black-box models. This relates to SATTVA's geometric pattern matching - distinct activation shapes for distinct concepts.

**Neurons "getting bored" (habituation):**
When repeatedly exposed to the same information, BDH neurons reduce their activation response - they adapt and habituate. This is analogous to biological novelty detection and connects to SATTVA's anomaly-based regulation: the system modulates its response based on whether information is surprising or routine.

**Production deployment:**
BDH is not just a research prototype - it's being deployed for Formula 1, NATO, French Postal Service, and other real-world applications. This demonstrates that post-Transformer brain-inspired architectures are practically viable at scale.

### Where BDH validates and extends SATTVA's direction

1. **Emergent structure works**: You don't need to pre-specify semantic organization; it can arise from local interaction rules
2. **Message passing + connection strengthening** is a viable alternative to Transformers
3. **Memory and reasoning** are separable from language modeling per se
4. **Scale-free networks** enable generalization across scales
5. **Biological plausibility** doesn't sacrifice performance

### Where SATTVA extends beyond BDH

While BDH is a major validation, SATTVA adds:
- **Geometric shape similarity** as explicit coupling mechanism (not just connection strength)
- **Depth dimension** for primitive vs. complex patterns with different coupling ranges
- **Long-range field effects** beyond local message passing
- **Geometric resonance-based creativity** (coupling semantically distant but geometrically similar patterns)
- **Explicit developmental hierarchy** (sensory baselining → geometric primitives → compositional meaning)
- **Task-conditioned anomaly regulation** (epistemic surprise vs. dynamical danger)

**Credit and acknowledgment:**
SATTVA's geometric field theory approach was developed independently but is strongly validated by Pathway's successful deployment of BDH. We acknowledge Zuzanna Stamirowska, Adrian Kosowski, Jan Chorowski, Przemysław Uznański, Michał Bartoszkiewicz, and the Pathway team for demonstrating that emergent brain-like architectures with message passing, Hebbian learning, and scale-free structure can rival Transformer performance while offering superior interpretability, continual learning, and memory capabilities. Their work provides crucial evidence that the principles underlying SATTVA are not merely theoretical but can be successfully implemented and deployed.

**References:**
- Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., & Bartoszkiewicz, M. (2025). The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507
- [Pathway AI](https://pathway.com/)

---

## Pathway's BDH (Baby Dragon Hatchling): Post-Transformer Brain-Like Architecture

Zuzanna Stamirowska and the Pathway team have developed BDH (Baby Dragon Hatchling), a post-Transformer architecture that demonstrates many principles central to SATTVA's vision and validates that brain-inspired architectures are practically viable.

### Core BDH principles aligned with SATTVA

**Emergent neural structure during training:**
BDH doesn't start with fixed architecture. Neural structures and connection patterns literally emerge during training from local message-passing rules. The team observed "the emergence of just this kind of brain appearing" in their lab - organization arises from simple interaction rules, not pre-designed topology. This directly validates SATTVA's "tangled substrate" principle where meaning emerges from activation geometry rather than pre-structured organization.

**Message passing between locally connected neurons:**
Instead of all-to-all attention (Transformers) or fixed weight matrices, BDH uses brain-like message passing: "one neuron gets information, passes it on to its neighbors, those with whom he's connected." Only neurons that "care" about incoming information light up and propagate signals. This is much closer to SATTVA's field propagation model than standard matrix multiplication.

**Hebbian learning and memory as strengthened connections:**
"Whenever two neurons were interested by something, the connection between them becomes stronger and this is memory." Connections that co-activate are reinforced; unused connections fade over time. This is exactly SATTVA's "transforming vector associations" - associations form, strengthen, and weaken based on co-activation patterns and usage.

**Scale-free biological network structure:**
BDH's neuron interaction network is explicitly a scale-free graph with heavy-tailed degree distribution and high modularity. The architecture is mathematically characterized as uniform and scale-free, meaning the same laws hold at different scales. This supports SATTVA's fractal brain structure concept - self-similar patterns at different hierarchical levels.

**True memory and continual learning:**
Unlike Transformers which "wake up with no memory" each inference (living in "Groundhog Day"), BDH maintains persistent memory through strengthened synaptic connections. It can learn continuously, adapt to new data over time, and maintain infinite context. This aligns with SATTVA's attractor basins as reusable, evolving semantic structures.

**Interpretability through sparse activation:**
BDH exhibits monosemanticity - specific neurons fire for specific abstract concepts (currencies, countries, etc.). Activation vectors are sparse and positive. The team describes having "CCTV inside the brain" rather than needing to build "MRI machines" to scan black-box models. This relates to SATTVA's geometric pattern matching - distinct activation shapes for distinct concepts.

**Neurons "getting bored" (habituation):**
When repeatedly exposed to the same information, BDH neurons reduce their activation response - they adapt and habituate. This is analogous to biological novelty detection and connects to SATTVA's anomaly-based regulation: the system modulates its response based on whether information is surprising or routine.

**Production deployment:**
BDH is not just a research prototype - it's being deployed for Formula 1, NATO, French Postal Service, and other real-world applications. This demonstrates that post-Transformer brain-inspired architectures are practically viable at scale.

### Where BDH validates and extends SATTVA's direction

1. **Emergent structure works**: You don't need to pre-specify semantic organization; it can arise from local interaction rules
2. **Message passing + connection strengthening** is a viable alternative to Transformers
3. **Memory and reasoning** are separable from language modeling per se
4. **Scale-free networks** enable generalization across scales
5. **Biological plausibility** doesn't sacrifice performance

### Where SATTVA extends beyond BDH

While BDH is a major validation, SATTVA adds:
- **Geometric shape similarity** as explicit coupling mechanism (not just connection strength)
- **Depth dimension** for primitive vs. complex patterns with different coupling ranges
- **Long-range field effects** beyond local message passing
- **Geometric resonance-based creativity** (coupling semantically distant but geometrically similar patterns)
- **Explicit developmental hierarchy** (sensory baselining → geometric primitives → compositional meaning)
- **Task-conditioned anomaly regulation** (epistemic surprise vs. dynamical danger)

**Credit and acknowledgment:**
SATTVA's geometric field theory approach was developed independently but is strongly validated by Pathway's successful deployment of BDH. We acknowledge Zuzanna Stamirowska, Adrian Kosowski, Jan Chorowski, Przemysław Uznański, Michał Bartoszkiewicz, and the Pathway team for demonstrating that emergent brain-like architectures with message passing, Hebbian learning, and scale-free structure can rival Transformer performance while offering superior interpretability, continual learning, and memory capabilities. Their work provides crucial evidence that the principles underlying SATTVA are not merely theoretical but can be successfully implemented and deployed.

**References:**
- Kosowski, A., Uznański, P., Chorowski, J., Stamirowska, Z., & Bartoszkiewicz, M. (2025). The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain. arXiv:2509.26507
- [Pathway AI](https://pathway.com/)

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

## 5. Geometric field theory: SATTVA as interference-based memory

### 5.1 Beyond weight matrices: activation geometry as information

A critical architectural clarification emerged from considering the physical nature of neural systems and field effects in biological computation: **SATTVA is not primarily a system of weighted connections between semantic vectors. It is a field-based system where information is encoded in the geometric configuration of activation patterns.**

This distinction is fundamental:

**Traditional neural networks:**
- Individual neurons or vector positions represent specific features/concepts
- Semantic similarity = distance in embedding space
- Learning = adjusting connection weights
- Computation = weighted sum and nonlinearity

**SATTVA's geometric field approach:**
- Individual units don't "mean" anything in isolation
- A concept = distributed pattern of activation forming a specific geometric shape
- Similar geometric configurations can couple even if semantically distant
- Computation = interference and resonance of activation patterns
- The substrate can appear as a "tangled mess" because organization emerges from activation geometry, not from pre-structured semantic organization

### 5.2 The tangled substrate and emergent geometry

The brain's neural structure appears tangled and disorganized when viewed anatomically, yet produces coherent cognition. SATTVA adopts this principle: the substrate (the collection of units and their connections) can be loosely structured or even quasi-random, because **meaning emerges from which patterns of units activate together and what geometric shape those activations form.**

When a "memory" activates, it creates a geometric form out of the substrate. Associated ideas are not activated by semantic similarity of content, but by **geometric compatibility of their activation patterns**—they form shapes that can constructively interfere.

Implication: We should expect many types of connections in the substrate, reflecting different spatial scales, modalities, and historical co-activation patterns. The apparent complexity is not a bug; it is the vocabulary from which geometric patterns are composed.

### 5.3 Long-range field effects and geometric coupling

A key insight borrowed from bioelectric field research (particularly Levin's work on morphogenetic fields): electrical and field effects can act over longer distances than typically modeled in computational neuroscience. Current tools may underestimate the brain's sensitivity to field-like coupling.

**SATTVA's commitment to long-range resonance:**

SATTVA makes an explicit architectural departure from both standard neural networks and local message-passing systems: **resonance coupling operates at 10-100x the range of typical connection distances.** This is not an incremental difference - it is a fundamental design choice based on the exponential amplification potential of resonant systems.

Why long-range coupling is essential:

**Exponential power of resonance:**
Resonant systems don't just add signals linearly - they can exhibit exponential amplification when frequencies/patterns align. A weak signal from a distant but geometrically compatible pattern can trigger significant activation if the resonance conditions are met. Standard neural network assumptions (connectivity falls off exponentially with distance) miss this entirely.

**Field propagation beyond local neighborhoods:**
Activation in one region creates a field of influence that propagates far beyond immediate neighbors. While field strength may decay with distance, it remains significant over ranges 10-100x larger than local synaptic connections would suggest. This is not about having dense all-to-all connectivity - it's about modeling the physics of field propagation.

**Geometric compatibility transcends spatial distance:**
Two patterns can be spatially distant in the network but geometrically similar in their activation shapes. Long-range coupling allows these patterns to resonate and mutually excite, regardless of their position in the substrate. This is the mechanism for "action at a distance" - not through direct connection, but through field overlap.

**Deep patterns reach further:**
Primitive/deep attractors (encoded with high "depth" parameter) have even longer coupling ranges - potentially affecting the entire system. This creates the "sewer system" lateral pathways where foundational patterns can activate distant surface concepts directly, bypassing hierarchical processing.

**Creativity through distant resonance:**
This long-range coupling is critical for creativity: patterns that are semantically distant (different conceptual domains) but geometrically similar (compatible activation shapes) can couple and interfere across long distances, producing **novel but coherent combinations** rather than random noise. Without long-range coupling, the system is trapped in local semantic neighborhoods.

**Implementation commitment:**
We assume coupling kernels with power-law or slower-than-exponential decay, ensuring significant influence extends 10-100x beyond typical "receptive field" distances. This is computationally challenging but architecturally essential - SATTVA without long-range coupling loses its core differentiation.

### 5.4 Critical mass dynamics and phase transitions

An important prediction of this geometric field view: **early learning appears unproductive until a critical mass of patterns accumulates.**

During early training:
- Individual patterns are weak and isolated
- They cannot sustain themselves or produce coherent outputs
- The system appears to learn very little

At critical density:
- Patterns begin to constructively interfere
- Weak patterns can combine to produce stable, coherent outputs
- The system undergoes a phase transition to useful behavior

This explains why deep learning systems often show sudden capability emergence after reaching sufficient scale: it's not just parameter count, but the density of learned patterns reaching a threshold where geometric interference becomes constructive.

### 5.5 Creativity as geometric resonance, not randomness

In most generative AI, creativity comes from stochastic sampling (temperature, top-k, etc.)—essentially controlled randomness. SATTVA proposes a different mechanism:

**Geometric resonance-based creativity:**
- Two patterns that are semantically distant but geometrically similar can couple
- Their interference produces a new pattern that inherits geometric coherence from both
- The result is novel (not in the training set) but meaningful (geometrically stable)
- This is why creative outputs can be "surprisingly apt" rather than merely surprising

Example: A visual pattern and a linguistic pattern might form similar geometric activation shapes. When they couple, the result is a metaphor that feels "right"—not because of semantic overlap, but because of geometric compatibility.

### 5.6 Depth, strength, and the fractal brain

Incorporating the fractal nature of brain structure (particularly insights from Kreinen's work on fractal structures from brainstem to cortex): patterns can be encoded at different "depths" in the processing hierarchy.

**Deep (primitive) patterns:**
- Encoded in foundational/brainstem-like structures
- Have long-range influence (their fields propagate further)
- Can create "lateral" connections that bypass normal hierarchical processing
- Resistant to change (structural/primitive)

**Surface (complex) patterns:**
- Encoded in higher processing regions
- More local influence
- More plastic and context-dependent

**Trauma as forced deep encoding:**
Under high stress (cortisol analog), experiences get encoded deeply along with their sensory and cognitive context. Because of the fractal structure:
- These deep patterns couple to primitive systems
- They create direct pathways (the "sewer system" metaphor) that allow rapid, inflexible activation
- They can activate distant surface patterns through their long-range fields
- This explains persistent, involuntary activation of trauma-related thoughts

**Implication for architecture:** SATTVA needs a depth dimension or strength parameter. Strong/deep attractors have larger coupling radii and influence more of the system.

### 5.7 Mathematical sketch: field dynamics

A first formalization of geometric field dynamics:

**State:**
- \(u_i(t)\) = activation level of unit \(i\) at time \(t\)
- \(\mathbf{r}_i\) = position of unit \(i\) in abstract geometric space
- \(d_i\) = depth parameter (primitive vs. complex)

**Field propagation:**
The field \(\phi(\mathbf{r}, t)\) at position \(\mathbf{r}\) due to all activations:

\[
\phi(\mathbf{r}, t) = \sum_i u_i(t) \cdot K(\|\mathbf{r} - \mathbf{r}_i\|, d_i)
\]

where \(K(\cdot)\) is a kernel that decays with distance but depends on depth (deeper patterns have longer-range kernels).

**Long-range coupling kernels:**
Unlike standard neural networks where influence falls off exponentially \(K(r) \propto e^{-r/\lambda}\) with small \(\lambda\), SATTVA uses long-range kernels:

\[
K(r, d) = \frac{A(d)}{1 + (r/R(d))^\alpha}
\]

where:
- \(R(d)\) is the characteristic range that increases with depth \(d\) (deep patterns have \(R \sim 10-100\times\) larger)
- \(\alpha \in [1, 2]\) gives power-law decay (much slower than exponential)
- \(A(d)\) is amplitude scaling

This ensures significant coupling over distances 10-100x beyond local neighborhoods, enabling:
- Distant pattern resonance
- Creative cross-domain coupling
- Deep attractor "broadcast" influence
- Field-like propagation rather than just synaptic connectivity

**Dynamics:**
Unit activation evolves based on local attractor forces plus field influence:

\[
\frac{du_i}{dt} = -\frac{\partial V}{\partial u_i}(u_i) + \alpha \phi(\mathbf{r}_i, t) + \eta_i(t)
\]

where:
- \(V(u_i)\) is a local energy function (attractor landscape)
- \(\alpha\) controls field coupling strength
- \(\eta_i(t)\) is noise/exploration

**Geometric pattern matching:**
Two activation patterns \(P_1 = \{u_i^{(1)}\}\) and \(P_2 = \{u_i^{(2)}\}\) have geometric similarity based on the shape they form, not their semantic content. This could be measured via:
- Topological features (persistent homology)
- Spatial moments and symmetries
- Graph structure of co-activated units
- Correlation of field distributions

### 5.8 Developmental hierarchy: From sensory primitives to compositional meaning

A critical architectural constraint from observing human development: **geometric primitives must stabilize before compositional meaning emerges.**

Human infants take considerable time from birth to create associations because they must first:

**Sensory baselining (0-6 months):**
- Establish "ground truth" through multi-modal sensory feedback
- Visual structures that correspond to physical reality (angles of edges, softness of light)
- Reinforcement learning to secure foundational "pieces" of awareness
- These are not concepts yet - just stable geometric patterns in sensory space

**Geometric primitive formation (6-18 months):**
- Assembly of low-level shapes activating physical forms (V1-V2 processing analog)
- Edge detection, contour integration, surface segmentation
- These patterns become stable attractors through repeated sensory confirmation
- The "geometry" here is literal: spatial relationships in visual/tactile/proprioceptive space

**Progression to meaning (18+ months):**
- Geometric primitives in sensory areas couple to patterns in higher processing regions
- Meaning emerges from associations between stable sensory patterns and contexts
- Compositional understanding: combinations of primitive patterns form complex concepts
- Language labels attach to already-formed geometric-semantic structures

**Implication for SATTVA architecture:**
We cannot skip the geometric primitive phase. The system must:

1. **Start with sensory-level patterns**: Low-dimensional, geometrically structured representations corresponding to "physical" features

2. **Build stable primitive attractors first**: Through repeated exposure and multi-modal confirmation, not through abstract semantic learning

3. **Enable hierarchical composition**: Higher-level patterns emerge as combinations/assemblies of lower-level stable patterns

4. **Maintain geometric grounding**: Even abstract concepts remain connected (through depth dimension) to primitive geometric patterns

5. **Use reinforcement from prediction success**: Patterns that successfully predict sensory consequences become more stable (like infant motor learning)

This developmental constraint explains why:
- Random initialization likely fails (no stable primitives)
- Pre-trained embeddings might help (they encode some geometric structure)
- Transfer learning works (stable lower-level patterns transfer across domains)
- Catastrophic forgetting happens (disrupting primitive patterns breaks everything above)

**Connection to trauma encoding:**
Trauma during early development (when primitive patterns are forming) creates deep attractors precisely because it gets encoded at the foundational geometric level, not just at the conceptual level. This explains the persistent, involuntary, and often "irrational" nature of trauma responses - they're wired into the geometric substrate before meaning-making structures existed.

### 5.9 Implications for implementation

This geometric field perspective means SATTVA implementation must:

1. **Not assume pre-organized semantic space**: The substrate can start loosely organized; geometry emerges through learning

2. **Measure pattern similarity geometrically**: Not just cosine similarity of vectors, but shape/topology of activation patterns

3. **Implement long-range coupling**: Field effects, not just nearest-neighbor connections

4. **Track critical mass**: Monitor when pattern density enables constructive interference

5. **Enable geometric creativity**: Allow distant but geometrically similar patterns to couple

6. **Model depth/strength explicitly**: Deep patterns have different dynamics than surface patterns

7. **Respect developmental hierarchy**: Build primitive geometric patterns before compositional meaning

8. **Provide sensory baselining**: Use prediction error from multiple modalities to stabilize foundational patterns

This is a fundamentally different computational model than standard neural networks or even most attractor networks, requiring careful thought about how to implement field propagation, geometric pattern matching, interference dynamics, and developmental sequencing efficiently.

---

## 6. Bioelectric pattern memory (Levin-lab bridge)

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
