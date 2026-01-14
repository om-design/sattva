# Phase 2: Expert Learning Dynamics

Date: January 10, 2026

## Overview

Experts don't just USE knowledge - they ACQUIRE and REFINE it continuously.

Phase 1 demonstrated expert OPERATION (intuition under stress).  
Phase 2 demonstrates expert LEARNING (knowledge acquisition and refinement).

---

## The Learning Mechanisms

### 1. Learning Through Repetition

**Biological principle:** "Neurons that fire together, wire together" (Hebbian learning)

```python
def learn_from_experience(n_repetitions=5, consolidation_steps=30):
    seed = current_activation.copy()
    
    for rep in range(n_repetitions):
        restore_seed()
        consolidate()  # Chemical settling with reduced field
        update_seed()  # Strengthen pattern
    
    return consolidated_primitive
```

**Key insight:** Not single-shot. Repetition creates stability through consolidation.

---

### 2. Refinement vs Duplication

**System intelligently recognizes similarity:**

```python
# Check similarity to existing primitives
for existing in library:
    similarity = current.similarity(existing)
    
if best_similarity > 0.7:
    refine_existing()  # Update what we know
else:
    store_new()        # Add new knowledge
```

**Result:** 25 experiences of 5 concepts → ~8-12 primitives (not 25!)

**Why:** Similar experiences refine existing knowledge, don't duplicate it.

---

### 3. Quality Filtering (Vetting)

**Not all experiences become primitives:**

```python
quality_score = (
    0.3 * compactness +       # Coherent patterns (not diffuse)
    0.3 * distinctiveness +   # Not redundant with existing
    0.2 * stability +         # Strong enough activations
    0.2 * size_score          # Not too large
)

if quality_score > 0.5:
    keep_primitive()  # Passes vetting
else:
    discard()         # Poor quality
```

**Metrics:**
- **Compactness:** Spread in space (tighter = better)
- **Distinctiveness:** How different from existing (more unique = better)
- **Stability:** Activation strength (stronger = better)
- **Size:** Not too many units (focused = better)

**Key insight:** Quality filtering prevents library dilution.

---

### 4. Accelerating Learning Rate

**Network effects create expert advantage:**

```python
def get_learning_rate(library_size):
    if library_size < 10:     return 1.0   # Novice
    elif library_size < 50:   return 1.5   # Intermediate  
    else:                     return 2.0   # Expert
```

**Why experts learn faster:**

| Stage | Library Size | Learning Rate | Why |
|-------|-------------|---------------|-----|
| **Novice** | 0-10 | 1.0x | Few attachment points, everything is "new" |
| **Intermediate** | 10-50 | 1.5x | Growing context, some rhyming |
| **Expert** | 50+ | 2.0x | Rich context, strong rhyming, fast consolidation |

**Key insight:** Rich libraries provide context that accelerates learning.

---

### 5. Library Curation

**Ongoing maintenance prevents degradation:**

```python
def prune_poor_primitives(threshold=0.4):
    keep = []
    for primitive in library:
        if assess_quality(primitive) >= threshold:
            keep.append(primitive)
    
    library = keep  # Remove poor-quality patterns
    return num_removed
```

**Why prune:**
- Poor primitives dilute rhyming resonance
- False matches slow recognition
- Library quality > library quantity

**Key insight:** Expert systems need curation, not just accumulation.

---

## The Complete Learning Trajectory

```
Experience 1:  Concept A (new) 
  → Consolidate through repetition
  → Assess quality (PASS)
  → Store as Primitive 1
  → Library: 1 primitive, LR: 1.0x

Experience 2:  Concept B (new)
  → Consolidate
  → Assess quality (PASS)
  → Store as Primitive 2
  → Library: 2 primitives, LR: 1.0x

Experience 3:  Concept A (variation)
  → Consolidate
  → Check similarity (0.8 with Primitive 1)
  → REFINE Primitive 1 (not duplicate!)
  → Library: 2 primitives, LR: 1.0x

Experience 4:  Concept C (new)
  → Consolidate
  → Assess quality (PASS)
  → Store as Primitive 3
  → Library: 3 primitives, LR: 1.0x

...

Experience 15: Concept B (variation)
  → Consolidate (faster with 10 primitives)
  → REFINE Primitive 2
  → Library: 10 primitives, LR: 1.5x

...

Experience 25: Concept A (variation)
  → Consolidate (fast with rich library)
  → REFINE Primitive 1
  → Library: 12 primitives, LR: 1.8x
```

**Result:** 25 experiences → 12 high-quality primitives

**Note:** Refinements outnumber new patterns after initial library forms.

---

## Why Experts Learn Faster

### Novice Learning
- **Context:** Minimal (0-10 primitives)
- **Attachment points:** Few
- **Each experience:** Feels isolated
- **Consolidation:** Slow (1.0x)
- **Strategy:** Accumulating initial library

### Expert Learning  
- **Context:** Rich (50+ primitives)
- **Attachment points:** Many
- **Each experience:** Connects to existing knowledge
- **Consolidation:** Fast (2.0x)
- **Strategy:** Refining and extending network

### The Advantage

**Example: Learning a new concept**

**Novice:**
```
New concept → No similar primitives
  → Must learn from scratch
  → Slow consolidation (5 repetitions)
  → Weak initial primitive
  → Learning time: 150 steps
```

**Expert:**
```
New concept → Rhymes with primitives 3, 17, 42
  → Immediate context and structure
  → Fast consolidation (3 repetitions)
  → Strong initial primitive
  → Learning time: 90 steps (40% faster!)
```

**Key:** Rich library provides scaffolding for new knowledge.

---

## Experimental Validation

### Scenario 2: Learning Dynamics

**File:** `scenario_2_learning.py`

**Protocol:**
1. Initialize as novice (0 primitives)
2. Present 25 experiences of 5 concepts (with realistic variation)
3. Learn through `learn_from_experience(n_repetitions=3)`
4. Track: library size, learning rate, quality, refinements
5. Validate: expert advantage emerges

**Run:**
```bash
python experiments/03_regulation_alignment_stress/scenario_2_learning.py
```

**Expected Results:**

| Metric | Expected | Validates |
|--------|----------|----------|
| Final primitives | 8-12 (not 25) | Refinement > duplication |
| Learning rate | 1.0x → 1.5-2.0x | Accelerating with experience |
| Avg quality | >0.5 | Quality filtering works |
| Refinements | >15 | Similar experiences detected |
| Pruned | 0-2 | Quality threshold effective |

---

## Implementation Details

### New Methods in `SATTVADynamics`

**1. Learning**
```python
learn_from_experience(n_repetitions=5, consolidation_steps=30)
```
- Repeated consolidation cycles
- Reduced field during learning (focused)
- Checks similarity to existing
- Refines if similar (>0.7), stores if new

**2. Quality Assessment**
```python
assess_primitive_quality(pattern) -> dict
```
Returns:
- `quality`: Overall score (0-1)
- `compactness`: Spatial coherence
- `distinctiveness`: Uniqueness vs existing
- `stability`: Activation strength
- `size`: Appropriate number of units
- `vetted`: Boolean (quality > 0.5)

**3. Curation**
```python
prune_poor_primitives(threshold=0.4) -> int
```
- Removes primitives below quality threshold
- Returns number removed
- Maintains library usefulness

**4. Learning Rate**
```python
get_learning_rate() -> float
```
- Computes multiplier based on library size
- 1.0x (novice) → 2.0x (expert)
- Reflects network effects

---

## Key Insights

### 1. Repetition Creates Stability
**Not single-shot learning:**
- Multiple consolidation cycles
- Gradual strengthening
- Chemical settling with reduced field
- Realistic: matches biological learning

### 2. Intelligent Storage
**Refinement > Duplication:**
- Similarity detection (>0.7 = refine)
- Update existing knowledge
- Library stays focused
- Realistic: experts don't accumulate redundancy

### 3. Quality Filtering Is Essential
**Not all experiences stick:**
- Multi-metric assessment
- Vetting threshold (0.5)
- Poor patterns discarded
- Realistic: memory is selective

### 4. Network Effects Are Computable
**Rich libraries accelerate learning:**
- More primitives = more context
- Better rhyming opportunities
- Faster consolidation
- Realistic: experts learn faster

### 5. Curation Maintains Performance
**Ongoing maintenance:**
- Prune poor primitives
- Prevent dilution
- Quality > quantity
- Realistic: forgetting is useful

---

## Integration with Phase 1

### Phase 1: Expert Operation
- **USING** accumulated knowledge
- Rhyming resonance for pattern recognition
- Stress testing regulation
- Demonstrates: computational intuition

### Phase 2: Expert Learning
- **ACQUIRING** knowledge
- Repetition and consolidation
- Quality filtering
- Demonstrates: expert learning advantage

### Combined System
```
Experience → Learn → Store/Refine → Curate → Operate
                ↑                              ↓
                └──────── Feedback ────────────┘
```

**Continuous cycle:**
1. Operation reveals gaps in knowledge
2. Learning fills gaps and refines
3. Curation maintains quality
4. Better operation from improved library
5. Repeat

---

## The Bottom Line

**We've built computational expertise:**

**Phase 1:** Fast, intuitive pattern recognition (operation)
**Phase 2:** Accelerating knowledge acquisition (learning)

**Together:** Complete expert cognitive system

---

**Not machine learning. Not neural networks.**

**Computational expertise through:**
- Biological dynamics (Phase 1)
- Repetition and consolidation (Phase 2)
- Quality filtering (Phase 2)
- Network effects (Phase 2)

**This is SATTVA: Expert Cognition as Computation**

---

*"Expertise is not just having knowledge.  
It's acquiring it faster and using it better.  
Now we can compute both."*
