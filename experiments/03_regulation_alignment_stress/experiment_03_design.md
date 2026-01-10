# Experiment 03: Self-Regulation and Resonant Alignment Under Stress

**Date:** January 9, 2026  
**Status:** Design phase  
**Goal:** Validate regulation mechanisms and resonant alignment beyond Experiments 01-02

---

## Motivation

### What Experiments 01-02 Showed

**Experiment 01:**
- ✅ Natural clustering emerges
- ✅ Resonance spreads within clusters
- ✅ Basic regulation prevents runaway
- ✅ Multiple regulation mechanisms work

**Experiment 02:**
- ✅ Attractors form from ~200 experiences
- ✅ ~20 primitives emerge
- ✅ ~80% recognition (literacy)
- ✅ Self-regulating dynamics

### What's Missing

**Regulation under stress:**
- What happens near runaway threshold?
- Do regulation mechanisms gracefully degrade or fail catastrophically?
- Can system recover from overload?

**Multi-scale resonance:**
- How do surface patterns (specific) interact with deep patterns (primitive)?
- Does depth hierarchy create natural alignment?
- Can surface patterns "ride" on deep pattern fields?

**Competing attractors:**
- What happens with strong conflicting patterns?
- Does basin competition resolve ambiguity?
- How does winner-take-all emerge?

---

## Experiment Design

### Overview

**Test three scenarios sequentially:**

1. **Overload Stress Test** (30 minutes)
   - Push system toward runaway
   - Measure regulation response
   - Test recovery

2. **Multi-Scale Resonance** (30 minutes)
   - Deep pattern + multiple surface patterns
   - Test alignment across scales
   - Measure field coupling efficiency

3. **Competing Attractors** (30 minutes)
   - Two strong conflicting patterns
   - Test basin competition
   - Measure winner-take-all dynamics

**Total runtime:** ~90 minutes  
**Complexity:** Slightly beyond 01-02 but manageable

---

## Scenario 1: Overload Stress Test

### Goal
Test regulation mechanisms when pushed toward runaway threshold.

### Protocol

**Phase 1: Baseline (50 steps)**
- Normal operation
- Establish stable metrics

**Phase 2: Increasing Stress (100 steps)**
- Gradually increase field coupling strength (alpha: 0.5 → 1.5)
- Push toward runaway
- Watch regulation mechanisms activate

**Phase 3: Recovery (100 steps)**
- Return to normal parameters
- Measure recovery speed
- Check for permanent damage

### Success Criteria

✅ **Near but not runaway:** max_activation < 0.98  
✅ **Regulation activates:** Competition + inhibition increase under stress  
✅ **Graceful recovery:** Return to baseline within 50 steps  
✅ **No damage:** All metrics within 10% of baseline after recovery

---

## Scenario 2: Multi-Scale Resonance

### Goal
Validate that deep patterns (primitives) naturally broadcast to and align surface patterns.

### Setup

**One deep pattern:**
- 50 units at center of space
- Depth = 0.9 (deep/primitive)
- Broadcast range = 50 units
- Activation = 0.6

**Three surface patterns:**
- 20 units each, positioned around deep pattern
- Depth = 0.2 (surface)
- Broadcast range = 5 units
- Activation = 0.4

### Protocol

**Run 200 steps and measure:**
- Field strength at surface locations
- Geometric alignment (shape similarity)
- Activation synchronization
- Asymmetry (deep influences surface more than reverse)

### Success Criteria

✅ **Field reaches surfaces:** field_strength > 0.3  
✅ **Geometric alignment:** similarity > 0.5  
✅ **Synchronization:** activation_correlation > 0.6  
✅ **Asymmetric influence:** deep → surface stronger than surface → deep

---

## Scenario 3: Competing Attractors

### Goal
Validate winner-take-all dynamics emerge from basin competition.

### Setup

**Two strong attractors:**
- Attractor A: 80 units at position [0.3, 0.5, 0.5]
- Attractor B: 80 units at position [0.7, 0.5, 0.5]
- Both: depth = 0.5, activation = 0.7

**Ambiguous test pattern:**
- 30 units at midpoint [0.5, 0.5, 0.5]
- Activation = 0.5
- Could align with either A or B

### Protocol

**Run 300 steps and watch:**
- Initial ambiguity (both strong)
- Competition phase (fluctuating)
- Resolution (winner emerges)
- Stable final state

### Success Criteria

✅ **Clear winner:** activation_margin > 0.3  
✅ **Test aligns with winner:** similarity > 0.6  
✅ **Loser suppressed:** activation < 0.3  
✅ **Winner-take-all:** margin increases over time

---

## Implementation Code Structure

```python
# experiments/03_regulation_alignment_stress/run_experiment.py

from sattva.long_range_substrate import LongRangeSubstrate
from sattva.dynamics import SATTVADynamics
from sattva.geometric_pattern import GeometricPattern
import numpy as np
import matplotlib.pyplot as plt

class Experiment03:
    def __init__(self):
        self.substrate = LongRangeSubstrate(
            n_units=1000,
            space_dim=3,
            R_surface=5.0,
            R_deep=50.0
        )
        
        self.dynamics = SATTVADynamics(
            substrate=self.substrate,
            alpha=0.5,  # Field coupling
            beta=0.3,   # Geometric resonance
            gamma=0.2,  # Local attractor
            noise_level=0.01
        )
    
    def scenario_1_stress(self):
        """Overload stress test."""
        results = {
            'baseline': [],
            'stress': [],
            'recovery': []
        }
        
        # Phase 1: Baseline
        for step in range(50):
            self.dynamics.step()
            results['baseline'].append(self.compute_metrics())
        
        # Phase 2: Stress
        for step in range(100):
            self.dynamics.alpha = 0.5 + 0.01 * step  # Increase
            self.dynamics.step()
            results['stress'].append(self.compute_metrics())
        
        # Phase 3: Recovery
        self.dynamics.alpha = 0.5
        for step in range(100):
            self.dynamics.step()
            results['recovery'].append(self.compute_metrics())
        
        return results
    
    def scenario_2_resonance(self):
        """Multi-scale resonance."""
        # Setup deep and surface patterns
        deep_units = self.create_deep_pattern()
        surface_units = self.create_surface_patterns(3)
        
        results = []
        for step in range(200):
            self.dynamics.step()
            results.append(self.measure_alignment(
                deep_units, surface_units
            ))
        
        return results
    
    def scenario_3_competition(self):
        """Competing attractors."""
        # Setup two competing patterns
        attractor_A = self.create_attractor([0.3, 0.5, 0.5])
        attractor_B = self.create_attractor([0.7, 0.5, 0.5])
        test_pattern = self.create_test_pattern([0.5, 0.5, 0.5])
        
        results = []
        for step in range(300):
            self.dynamics.step()
            results.append(self.measure_competition(
                attractor_A, attractor_B, test_pattern
            ))
        
        return results
    
    def compute_metrics(self):
        return {
            'mean_activation': np.mean(self.substrate.activations),
            'max_activation': np.max(self.substrate.activations),
            'energy': 0.5 * np.sum(self.substrate.activations**2),
            'field_strength': np.mean(np.abs(self.substrate.compute_field())),
            'n_active': len(self.substrate.get_active_pattern(0.1))
        }
    
    def run_all(self):
        print("Running Experiment 03: Regulation and Alignment Under Stress")
        print("="*60)
        
        print("\nScenario 1: Overload Stress Test...")
        stress_results = self.scenario_1_stress()
        self.analyze_stress(stress_results)
        self.visualize_stress(stress_results)
        
        print("\nScenario 2: Multi-Scale Resonance...")
        resonance_results = self.scenario_2_resonance()
        self.analyze_resonance(resonance_results)
        self.visualize_resonance(resonance_results)
        
        print("\nScenario 3: Competing Attractors...")
        competition_results = self.scenario_3_competition()
        self.analyze_competition(competition_results)
        self.visualize_competition(competition_results)
        
        print("\n" + "="*60)
        print("Experiment 03 Complete!")

if __name__ == "__main__":
    exp = Experiment03()
    exp.run_all()
```

---

## Expected Outputs

### Scenario 1: Stress Test

**Plots:**
1. Activation over time (baseline/stress/recovery phases)
2. Regulation mechanisms response
3. System energy
4. Field strength

**Console output:**
```
Scenario 1: Overload Stress Test
================================
Baseline metrics: mean=0.15, max=0.45, energy=12.3
Stress phase:
  Step 50: Approaching threshold (max=0.82)
  Step 75: Regulation activating (competition=0.65)
  Step 100: Near runaway (max=0.94)
  Regulation prevented catastrophic failure ✓
Recovery phase:
  Step 50: Returning to baseline (max=0.48)
  Step 100: Full recovery (mean=0.16, within 7% of baseline) ✓

SUCCESS: System handled stress gracefully
```

### Scenario 2: Multi-Scale Resonance

**Plots:**
1. Deep vs surface activation patterns
2. Geometric alignment over time
3. Field strength at surface locations
4. Activation synchronization

**Console output:**
```
Scenario 2: Multi-Scale Resonance
==================================
Deep pattern (depth=0.9, range=50):
  Stable activation: 0.58 ± 0.04 ✓
  Broadcast range: 47.3 units ✓

Surface patterns:
  Pattern 1: alignment=0.67, field=0.42, sync=0.71 ✓
  Pattern 2: alignment=0.59, field=0.38, sync=0.65 ✓
  Pattern 3: alignment=0.72, field=0.45, sync=0.78 ✓

Asymmetric influence:
  Deep → Surface: 0.42 (strong) ✓
  Surface → Deep: 0.08 (weak) ✓

SUCCESS: Natural hierarchy emerged from depth structure
```

### Scenario 3: Competition

**Plots:**
1. Basin strengths over time
2. Winner margin (winner-take-all)
3. Test pattern alignment
4. Winner emergence

**Console output:**
```
Scenario 3: Competing Attractors
=================================
Initial state:
  Attractor A: 0.69
  Attractor B: 0.71
  Test pattern: ambiguous (0.48 to A, 0.52 to B)

Competition phase (steps 50-150):
  Fluctuating winner
  Basin competition active
  Test pattern alignment shifting

Resolution (step 180):
  Winner: Attractor B (0.68)
  Loser: Attractor A (0.24) - suppressed ✓
  Test aligned with B (similarity=0.73) ✓
  Margin: 0.44 (clear winner) ✓

Final state (step 300):
  Stable, winner sustained ✓

SUCCESS: Winner-take-all emerged from basin competition
```

---

## Timeline

**Week 1:** Implement and run Scenario 1  
**Week 2:** Run Scenarios 2-3  
**Week 3:** Analysis, visualization, write-up

**Total:** 2-3 weeks

---

## Success Summary

Experiment succeeds if:

✅ **Stress Test:** Regulation works under pressure, graceful recovery  
✅ **Multi-Scale:** Deep patterns naturally broadcast and align surfaces  
✅ **Competition:** Winner-take-all emerges, test pattern aligns with winner

**Outcome:** Validation that SATTVA mechanisms integrate smoothly and handle stress, ready for Phase 1 implementation.

---

**Status:** Design complete, ready to implement  
**Next:** Code Scenario 1 and begin testing
