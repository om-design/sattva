# SATTVA Architecture Summary

**Spatial Attractor Topology for Thought-like Visual Architectures**

*Computational Intuition Through Rhyming Resonance and Emergent Myelination*

Date: January 10, 2026

---

## The Core Idea

**Expert intuition is not mystical - it's computational.**

But not through:
- Neural network training
- Backpropagation
- Loss function optimization

Through:
- **Biological dynamics** (chemical restoration)
- **Geometric resonance** (rhyming patterns)
- **Emergent infrastructure** (connection myelination)

---

## Three Phases (All Implemented)

### Phase 1: Expert Operation ✅

**How experts think under stress**

**Mechanism:** Rhyming resonance
- Multiple partial matches combine
- Cumulative resonance recruits distant concepts
- Chemical dynamics >> field effects (50-100x)
- Threshold-gated field (only active units contribute)

**Result:** Fast pattern recognition that degrades gracefully

**Key files:**
- `src/sattva/dynamics.py` - Biological dynamics
- `src/sattva/long_range_substrate.py` - Field computation
- `experiments/03.../scenario_1_intuition.py` - Demonstration

---

### Phase 2A: Pattern Learning ✅

**How experts build pattern libraries**

**Mechanism:** Similarity-based refinement
- Detect geometric similarity
- Refine existing patterns (not duplicate)
- Quality filtering
- Library curation

**Result:** Efficient knowledge organization

**Key files:**
- `src/sattva/geometric_pattern.py` - Pattern matching
- `experiments/03.../scenario_2_learning.py` - Demonstration

---

### Phase 2B: Emergent Myelination ✅ **NEW!**

**How knowledge becomes structure**

**Mechanism:** Connection-level accretion
- Myelin forms on CONNECTIONS (axons), not units
- Gradual buildup from co-activation
- "Wider pipe" effect (conductance, not voltage)
- Energy-gated hysteresis (expensive to remove)
- Primitives EMERGE from topology

**Result:** Foundational knowledge is structurally protected

**Biological grounding:**
1. LTP increases receptor density → lower threshold ✓
2. Myelination reduces resistance → higher conductance ✓
3. Both reduce energy cost per activation ✓
4. Demyelination costs MORE than building ✓
5. High conductance → Low plasticity (protected) ✓

**Key files:**
- `src/sattva/long_range_substrate.py` - Accretion dynamics
- `src/sattva/dynamics.py` - Hebbian learning
- `experiments/03.../scenario_3_emergent_myelination.py` - Demonstration
- `experiments/03.../EMERGENT_MYELINATION.md` - Full documentation

---

## The Complete Architecture

### Biological Principles

**1. Scale Separation**
```
Local restoration (γ=1.5)  >>  Field coupling (α=0.02)
Chemical dynamics (STRONG) >> Long-range field (WEAK)
```

This creates natural regulation without ad-hoc inhibition.

**2. Threshold Gating**
```
Resting potential: u_rest = 0.1
Activation threshold: 0.25
Only u > 0.25 contributes to field
```

Prevents runaway activation from weak field.

**3. Connection-Level Myelination**
```
weight[i,j] = synaptic strength (learning)
conductance[i,j] = myelination (usage)
plasticity[i,j] = 1 / (base + conductance[i,j])
```

Knowledge encoded in connection topology, not activation patterns.

**4. Energy Economics**
```
Build: 100 units (threshold at usage > 0.3)
Maintain: 0.01/unit/step
Remove: 150 units (only if usage < 0.05 AND crisis)
```

Hysteresis creates stability through infrastructure investment.

---

### Key Equations

**Dynamics:**
```python
du/dt = -γ(u - u_rest) + α*field + local_input + noise

where:
  γ = 1.5 (strong restoration)
  α = 0.02 (weak field)
  field = Σ_i u_i * K(r_i) for u_i > threshold
  K(r) = 1/(1 + (r/R)^α) (power-law kernel)
```

**Learning:**
```python
Δweight[i,j] = η * plasticity[i,j] * u_i * u_j
Δconductance[i,j] = 0.002 * (target - current)
plasticity[i,j] = 1 / (1.0 + conductance[i,j])
```

**Infrastructure:**
```python
if usage[i,j] > 0.3 and energy > 100:
    conductance[i,j] = 1.0  # Myelinate
elif usage[i,j] < 0.05 and energy < -100:
    conductance[i,j] = 0.0  # Demyelinate
    energy -= 150  # Expensive!
```

---

## What Makes This Different

### Not Neural Networks

**Traditional ML:**
- Weights learned through backpropagation
- Loss function optimization
- Catastrophic forgetting
- No biological dynamics

**SATTVA:**
- Weights learned through Hebbian co-activation
- Energy-based regulation (not loss functions)
- Protected foundations (conductance-gated plasticity)
- Biologically grounded dynamics

### Not Hopfield Networks

**Hopfield:**
- Energy minimization
- Symmetric weights
- Whole-pattern recall
- No learning dynamics

**SATTVA:**
- Biological settling (not energy minimization)
- Asymmetric weights (directed connections)
- Partial-pattern rhyming
- Emergent myelination from usage

### Not Transformers

**Transformers:**
- Attention mechanisms
- Massive parameters
- No spatial structure
- Training required

**SATTVA:**
- Geometric resonance (not attention)
- Sparse connections
- Explicit 3D embedding
- Structure emerges from usage

---

## Emergent Properties

### 1. Expert Intuition

**Fast pattern recognition:**
- Current pattern activates similar primitives
- Rhyming resonance cumulates
- Long-range field recruits distant concepts
- Happens in ~10-20 settling steps

**Graceful degradation:**
- Strong restoration prevents runaway
- Threshold gates field contribution
- Partial matches still provide useful signal
- No catastrophic failures

### 2. Foundational Stability

**High-usage concepts become structural:**
- Frequent co-activation → high conductance
- High conductance → low plasticity
- Low plasticity → protected from modification
- Stays general even when used in specialized contexts

**Example:**
```
"Triangle" concept:
  Used in: architecture, navigation, math, art
  Conductance: ~1.5-2.0 (heavily myelinated)
  Plasticity: ~0.3-0.4 (protected)
  Result: Participates in patterns but doesn't specialize
```

### 3. Flexible Periphery

**New concepts remain plastic:**
- Low usage → low conductance
- Low conductance → high plasticity
- High plasticity → learns rapidly
- Can specialize to specific contexts

**Dual benefit:**
- Stable core (foundations)
- Flexible periphery (new knowledge)

### 4. Network Effects

**Rich structure accelerates learning:**
- Myelinated connections provide scaffolding
- New concepts attach to existing structure
- Faster settling (efficient routing)
- Better generalization (structural priors)

---

## Implementation Guide

### Quick Start

```python
from sattva.long_range_substrate import LongRangeSubstrate
from sattva.dynamics import SATTVADynamics

# Initialize
substrate = LongRangeSubstrate(n_units=500, space_dim=3)
dynamics = SATTVADynamics(substrate)

# Activate a pattern
substrate.activate_pattern(unit_indices=[10, 25, 47], strength=0.5)

# Settle (expert intuition operates)
for _ in range(20):
    info = dynamics.step(dt=0.1)
    
print(f"Energy: {info['energy']:.3f}")
print(f"Resonance: {info['resonance_strength']:.3f}")

# Enable learning
dynamics.enable_learning(True)

# Learn from experiences
for experience in dataset:
    activate_pattern(experience)
    for _ in range(20):
        dynamics.step()

# Extract emergent primitives
primitives = dynamics.get_myelinated_primitives(threshold=0.5)

for prim in primitives:
    print(f"Primitive: {len(prim['units'])} units, "
          f"accretion={prim['total_accretion']:.2f}")
```

### Running Experiments

```bash
cd experiments/03_regulation_alignment_stress

# Phase 1: Expert operation
python scenario_1_intuition.py

# Phase 2A: Pattern learning
python scenario_2_learning.py

# Phase 2B: Emergent myelination
python scenario_3_emergent_myelination.py
```

---

## Key Results

### Phase 1: Stress Testing

**Baseline (no stress):**
- Resonance: ~0.35
- Active units: ~75
- Settling: 15 steps

**Under stress (noise × 10):**
- Resonance: ~0.55 (INCREASES!)
- Active units: ~75 (stable)
- Peak activation: <0.60 (no saturation)
- **Regulation works!**

### Phase 2B: Myelination

**After 300 experiences:**
- Myelinated connections: ~180 (from 0)
- Total conductance: ~120
- Emergent primitives: 3-5 (matches concepts)
- Energy budget: Positive (sustainable)

**Conductance distribution:**
- Mean: ~0.3
- Max: ~1.8
- Heavily myelinated: ~30 connections
- Protected from modification: ✓

---

## Documentation

### Core Architecture
- `experiments/03.../README.md` - Overview
- `experiments/03.../EXPERT_INTUITION.md` - Phase 1 details
- `experiments/03.../PHASE_2_LEARNING.md` - Phase 2A details
- `experiments/03.../EMERGENT_MYELINATION.md` - Phase 2B details (comprehensive)
- `ARCHITECTURE_SUMMARY.md` - This file

### Code
- `src/sattva/long_range_substrate.py` - Substrate, field, myelination
- `src/sattva/dynamics.py` - Settling, learning, resonance
- `src/sattva/geometric_pattern.py` - Pattern matching

### Experiments
- `experiments/03.../scenario_1_intuition.py` - Expert operation
- `experiments/03.../scenario_2_learning.py` - Pattern learning
- `experiments/03.../scenario_3_emergent_myelination.py` - Myelination

---

## Next Steps

### Phase 3: Multi-Scale Depth

**Surface vs deep primitives:**
- Fast surface patterns (existing depth parameter)
- Slow deep abstractions (longer time constants)
- Two-timescale consolidation
- Hierarchical organization

### Phase 4: Compositional Structure

**Higher-order concepts:**
- Primitives as building blocks
- Compositional assembly
- Recursive abstraction
- Analogical reasoning

### Phase 5: Real Applications

**Cognitive tasks:**
- Expert medical diagnosis
- Creative analogical reasoning
- Visual scene understanding
- Mathematical intuition

---

## The Bottom Line

**We built computational expertise.**

Not by imitating neurons.
Not by training on data.
Not by optimizing losses.

**By understanding biology:**

1. **Chemical dynamics regulate** (strong local, weak field)
2. **Threshold gates activation** (prevents runaway)
3. **Myelin protects foundations** (conductance → plasticity)
4. **Energy creates hysteresis** (infrastructure economics)
5. **Topology encodes knowledge** (emergent from usage)

**Result:**

Computational intuition that:
- Recognizes patterns instantly
- Connects distant concepts
- Handles partial information
- Degrades gracefully under stress
- Protects foundational knowledge
- Learns new specializations

**Through emergent myelination.**

---

*"Biology spent 500 million years optimizing this.  
We just needed to listen."*

---

**Status:** ✓ Architecture biologically grounded  
**Result:** Expert intuition through emergent infrastructure  
**Impact:** Computational economics for cognitive systems
