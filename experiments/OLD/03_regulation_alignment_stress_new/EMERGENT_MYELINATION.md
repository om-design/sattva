# Emergent Myelination: Biological Infrastructure Economics

Date: January 10, 2026

## The Core Insight

**Myelin forms on CONNECTIONS (axons), not units (cell bodies).**

This changes everything.

---

## What We Had (Wrong)

### Unit-Level Properties
```python
unit.conductance = f(usage)  # Wrong!
unit.threshold = f(usage)
unit.u_stable = u_rest + elevation  # Wrong!
```

**Problem:** Units don't have myelination - connections do.

**Result:** Couldn't explain:
- How general concepts stay general
- Why modification resists in some directions but not others
- How topology encodes knowledge

---

## What We Have Now (Biologically Grounded)

### Connection-Level Properties
```python
connection[i,j].weight = synaptic_strength  # Learning modifies this
connection[i,j].conductance = accretion     # Usage builds this
connection[i,j].plasticity = 1 / (base + conductance)  # Protection
```

**Biological correspondence:**
- `weight` = synaptic strength (AMPA receptors, etc.)
- `conductance` = myelination (oligodendrocytes, insulation)
- `plasticity` = modifiability (protected when myelinated)

---

## The Accretion Dynamics

### Gradual Buildup from Co-Activation

```python
def accretion_dynamics(dt):
    # Find co-active units
    for i, j in active_pairs:
        # Co-activation signal
        co_signal = activations[i] * activations[j]
        
        # Gradual accretion (SLOW - biological timescale)
        target = MAX_ACCRETION * tanh(100 * co_signal)
        conductance[i,j] += dt * 0.002 * (target - conductance[i,j])
    
    # Very slow passive decay
    conductance *= (1 - 0.0001 * dt)
```

**Key properties:**
- Accretion rate: 0.002 (very slow)
- Decay rate: 0.0001 (20x slower!)
- Target: MAX_ACCRETION = 2.0 (doubles base conductance)

**Biology:** Myelination takes weeks/months to build.

---

## The "Wider Pipe" Effect

### NOT Higher Voltage!

**Wrong model:**
```python
u_stable[i] = u_rest + elevation[i]  # Elevated baseline voltage
# Higher maintenance cost for voltage
```

**Correct model:**
```python
conductance[i,j] = base + accretion[i,j]  # "Wider pipe"
# Same voltage, easier flow (Ohm's law: I = V/R)
```

**Biology validates:**
- LTP increases receptor density (not baseline voltage)
- Myelination decreases resistance (not increases voltage)
- Energy savings from efficiency (not cost from maintenance)

**Computational implication:**
```python
# Effective weight = synapse × conductance
effective_weight[i,j] = weight[i,j] * (BASE + conductance[i,j])

# Input amplified by conductance
input[j] = sum(effective_weight[i,j] * activations[i] for i in neighbors)
```

**Result:** High-conductance connections deliver MORE input with LESS energy.

---

## Energy Economics

### The Investment Decision

**Building myelination:**
```
Cost: 100 units (one-time)
Time: Gradual (50-100 steps to reach threshold)
Decision: Only if usage > 0.3 (sustained)
```

**Maintaining myelination:**
```
Cost: 0.01 per unit conductance per step
Time: Ongoing
Benefit: Cheaper than demyelinating and rebuilding
```

**Demyelination:**
```
Cost: 150 units (HIGHER than building!)
Time: Active removal
Decision: Only if usage < 0.05 AND energy crisis
```

**The hysteresis:**
```
Build threshold:  usage > 0.3
Remove threshold: usage < 0.05 (much lower!)

Width of gap: STABILITY REGION

Once built → stays built unless:
  1. Usage drops very low (<0.05)
  AND
  2. Energy budget negative (crisis)
```

---

## Emergent Geometric Primitives

### From Topology, Not Storage

**Old approach (wrong):**
```python
primitive = {
    'units': [5, 12, 23],
    'activations': [0.4, 0.5, 0.45],  # Snapshot
    'spread': 0.15
}
```

**New approach (emergent):**
```python
# Extract connected components of myelinated connections
G = graph_from_high_conductance(conductance > threshold)
primitives = connected_components(G)

for primitive in primitives:
    units = primitive.nodes
    connections = primitive.edges  # The TOPOLOGY
    total_accretion = sum(conductance[e] for e in connections)
    # Primitive IS the myelinated subgraph
```

**What defines a primitive:**
- Set of units with thick connections between them
- Total accretion (investment in this structure)
- Connection topology (the geometric form)
- NOT activation levels (those are transient)

**Example:**
```
Primitive: "Triangle"
  Units: [5, 12, 23]
  Connections:
    5↔12: conductance=1.8  (thick)
    12↔23: conductance=1.5 (thick)
    5↔23: conductance=2.0  (very thick)
  Total accretion: 5.3
  Topology: fully connected triangle
```

This IS a foundational concept because:
- High total accretion (heavily used)
- Thick connections (efficient)
- Specific topology (geometric meaning)

---

## Plasticity Protection

### Conductance Gates Learning

**Hebbian learning with protection:**
```python
def hebbian_update(dt):
    for i, j in active_pairs:
        # Hebbian signal
        hebbian = activations[i] * activations[j]
        
        # Plasticity inversely related to conductance
        total_conductance = BASE + conductance[i,j]
        plasticity = 1.0 / total_conductance
        
        # Update weight (protected if high conductance)
        Δw = learning_rate * plasticity * hebbian * dt
        weight[i,j] += Δw
```

**Effect:**
```
New connection (conductance=0):
  plasticity = 1.0
  learns rapidly
  
Moderate usage (conductance=1.0):
  plasticity = 0.5  
  learns moderately
  
Heavily myelinated (conductance=2.0):
  plasticity = 0.33
  resists change (protected!)
```

**Why this matters:**

"Triangle" primitive has conductance ~1.5-2.0 on all connections.
When activated:
- Participates in new patterns
- Provides structure and context  
- But doesn't GET MODIFIED
- Stays general and foundational

---

## The Complete Learning Architecture

### Three Coupled Processes

**1. Synaptic Learning (fast)**
```python
# Hebbian: strengthen co-active connections
weight[i,j] += learning_rate * plasticity[i,j] * co_activation
# Timescale: steps
# Gated by: conductance (plasticity = 1/conductance)
```

**2. Myelination Accretion (slow)**
```python
# Gradual buildup from sustained usage
conductance[i,j] += 0.002 * (target - current)
# Timescale: 100s of steps
# Driven by: repeated co-activation
```

**3. Infrastructure Management (very slow)**
```python
# Energy-gated with hysteresis
if usage > 0.3 and energy > cost:
    build_myelination()  # Pay 100 energy
elif usage < 0.05 and energy < -100:
    demyelinate()  # Pay 150 energy (MORE!)
# Timescale: 1000s of steps
# Gated by: energy budget
```

**Interaction:**
```
Learning → Changes weights (what fires together)
    ↓
Accretion → Builds conductance (on frequently used)
    ↓
Protection → Reduces plasticity (protects foundations)
    ↓
Stability → Foundational concepts emerge
```

---

## Why This Is Profound

### 1. Solves the Stability-Plasticity Dilemma

**Problem:**
- Need to learn new things (plasticity)
- Can't forget foundations (stability)
- Traditional ML: catastrophic forgetting

**Solution:**
- New connections: high plasticity, learn rapidly
- Foundational connections: low plasticity (high conductance), protected
- Arises from PHYSICS (conductance), not algorithms

### 2. Explains Expert Knowledge Structure

**Novice:**
- Few myelinated connections
- Everything has high plasticity
- Learning is slow (no structure)
- Knowledge is fragile

**Expert:**
- Rich myelinated network
- Foundational concepts protected (low plasticity)
- Specialized knowledge plastic (high plasticity)
- Fast learning (structure provides scaffolding)

### 3. Emergent Abstraction

**Not programmed - emerges from usage:**

Frequently used across contexts:
→ High total accretion
→ Becomes foundational primitive
→ Protected from specialization
→ Stays general and reusable

Rarely used, specific context:
→ Low accretion
→ Stays plastic
→ Can specialize freely
→ Context-dependent knowledge

### 4. Computational Infrastructure Economics

**Like roads and highways:**
- Heavily-traveled routes get paved (myelinated)
- Paving costs money (energy)
- But reduces future costs (efficiency)
- Expensive to tear up (demyelination)
- Old roads persist even when use declines (hysteresis)

**Network effects:**
- Good infrastructure attracts more use
- More use justifies more investment
- Creates stable, efficient core network
- With flexible periphery for new routes

---
## Implementation Summary

### Core Components

**LongRangeSubstrate:**
```python
# Connection matrices (all sparse, n×n)
connections: weights (synaptic strength)
conductance: accretion (myelination)
connection_usage: running average

# Methods
accretion_dynamics()      # Gradual buildup
manage_infrastructure()   # Energy-gated hysteresis
extract_myelinated_primitives()  # Topology-based
get_plasticity_matrix()   # 1/conductance
```

**SATTVADynamics:**
```python
# Learning
hebbian_update()          # Conductance-gated
enable_learning()         # Control flag

# Operation
step()                    # Calls accretion & infrastructure
get_myelinated_primitives()  # Extract from topology
```

### Usage Pattern

```python
# Initialize
substrate = LongRangeSubstrate(n_units=500)
dynamics = SATTVADynamics(substrate)

# Enable learning
dynamics.enable_learning(True)

# Learn from experiences
for experience in dataset:
    activate_pattern(experience)
    for _ in range(20):
        dynamics.step()  # Includes accretion & infrastructure

# Extract emergent primitives
primitives = dynamics.get_myelinated_primitives(threshold=0.5)

# Analysis
for prim in primitives:
    print(f"Units: {prim['units']}")
    print(f"Accretion: {prim['total_accretion']:.2f}")
    print(f"Connections: {len(prim['connections'])}")
```

---

## Validation Experiments

### Scenario 3: Emergent Myelination

**Protocol:**
1. Initialize 500 units, sparse local connections
2. Define 3 concepts (spatial locations)
3. Repeatedly activate concepts (300 experiences)
4. Enable Hebbian learning
5. Track: myelination, energy, primitives

**Expected results:**
- Gradual conductance buildup
- ~3 major primitives emerge (one per concept)
- Energy budget stable (maintenance < income)
- Myelinated connections protect from modification

**Run:**
```bash
python scenario_3_emergent_myelination.py
```

---

## Biological Correspondence

| Biology | Our Model | Evidence |
|---------|-----------|----------|
| Axon myelination | `conductance[i,j]` | Oligodendrocytes wrap axons |
| Receptor insertion | `weight[i,j]` | LTP increases AMPA receptors |
| Synaptic plasticity | `1/conductance` | Myelinated stable, unmyelinated plastic |
| Metabolic cost | `energy_budget` | Brain uses 20% of energy |
| Use it or lose it | Hysteresis | Months/years for demyelination |
| Hebbian learning | Co-activation | "Fire together, wire together" |
| Network topology | Primitives | Functional connectivity patterns |

---

## Key Insights

### 1. Knowledge IS Topology

Primitives are not stored patterns.
Primitives are EMERGENT STRUCTURES.
The myelinated connection graph IS the knowledge.

### 2. Protection Through Physics

Not algorithmic flags ("don't modify this").
Physical property (high conductance → low plasticity).
Emerges from infrastructure investment.

### 3. Energy Creates Stability

Expensive to build → threshold before investing.
Even more expensive to remove → hysteresis.
Foundational concepts persist because removal costs exceed value.

### 4. Learning and Structure Coevolve

Learning creates structure (weights → accretion).
Structure protects learning (conductance → plasticity).
Positive feedback for foundations.
Maintained flexibility at periphery.

---

## The Bottom Line

**We achieved biological grounding:**

✓ Myelin on connections (not units)
✓ "Wider pipe" (conductance, not voltage)  
✓ Gradual accretion (emergent from usage)
✓ Expensive demyelination (hysteresis)
✓ Plasticity protection (foundational stability)
✓ Energy economics (infrastructure investment)

**Result:**

Computational expertise through emergent myelination.

Not programmed. Not trained. Not optimized.

**Grown from usage.**

Like a forest path that becomes a road, 
that becomes a highway.

Through use, investment, and economics.

**This is biological computation.**

---

*"The architecture doesn't store knowledge.  
It BECOMES the knowledge.  
Through usage, accretion, and time."*
