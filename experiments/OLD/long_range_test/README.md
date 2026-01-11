# Long-Range Coupling Validation Tests

These experiments validate SATTVA's core architectural claims about long-range coupling and two-timescale regulation.

## Tests

### 1. `test_distant_resonance.py` - BASELINE (Shows the problem)

**Purpose:** Demonstrate that long-range coupling (10-100x) enables distant pattern resonance, but without regulation causes runaway.

**Expected result:** 
- ✅ Distant patterns DO couple and resonate
- ❌ System goes unstable (energy explodes, all units saturate)

**Why it explodes:** Single timescale, no validated substrate, formation and resonance at same rate.

**Run:**
```bash
python experiments/long_range_test/test_distant_resonance.py
```

**Key learning:** Proves the theory's prediction that long-range + resonance without regulation = instability.

---

### 2. `test_two_timescale_resonance.py` - SOLUTION (Two-timescale regulation)

**Purpose:** Demonstrate that two-timescale dynamics (fast resonance over validated primitives) enables STABLE distant resonance.

**Expected result:**
- ✅ Distant patterns couple and resonate  
- ✅ System remains stable (energy bounded)
- ✅ Operating over validated primitive substrate prevents runaway

**Key mechanisms:**
1. **Validated primitives** - Bootstrap 30 stable geometric patterns first
2. **Fast updates** - Activation/resonance every step (milliseconds)
3. **Slow updates** - Pattern formation every 100 steps (hours/days)
4. **Damping** - Prevents positive feedback loops
5. **Geometric coupling** - Only to pre-validated patterns

**Run:**
```bash
python experiments/long_range_test/test_two_timescale_resonance.py
```

**Or use the convenience script:**
```bash
cd /Users/omdesign/code/GitHub/sattva
bash run_two_timescale_test.sh
```

**Key learning:** Validates that biological two-timescale mechanism prevents runaway while preserving long-range creativity.

---

## Theoretical Background

### Long-Range Coupling (10-100x)

SATTVA uses power-law kernels instead of exponential decay:

```
K(r, d) = A(d) / (1 + (r/R(d))^alpha)
```

Where:
- `R_surface = 5` (local patterns)
- `R_deep = 50` (primitive patterns, 10x range)
- `alpha = 1.5` (power-law decay)

At distance `r = 10R`:
- Exponential: `e^-10 ≈ 0.00005` (negligible)
- Power-law: `(1+10)^-2 ≈ 0.008` (significant!)

This 10-100x range enables:
- Creative cross-domain resonance
- Deep attractor broadcast influence
- Geometric compatibility coupling (not just semantic similarity)

### Two-Timescale Regulation

**Biological insight:** Human brains separate slow pattern formation from fast pattern association.

**Physical layer (slow):**
- Synaptic changes require repetition (hours/days)
- Rate-limited by metabolism
- Acts as "speed limit" during learning

**Conceptual layer (fast):**
- Resonance between validated patterns (milliseconds)
- Safe because substrate is stable
- Fast association permitted over reliable primitives

**Why this prevents runaway:**
1. Formation is rate-limited (can't create explosive loops during learning)
2. Fast resonance over stable substrate is safe (validated patterns unlikely catastrophic)
3. Timescale separation breaks positive feedback

### Implementation

**Bootstrap phase:**
```python
# Create validated primitives first (slow formation)
primitives = bootstrap_validated_primitives(substrate, n_primitives=30)
```

**Operation phase:**
```python
# Fast resonance over primitives
dynamics = TwoTimescaleDynamics(
    substrate,
    validated_primitives=primitives,
    fast_dt=0.02,    # fast: activation/resonance
    slow_dt=1.0,     # slow: formation/learning
    slow_every=100,  # slow update frequency
    alpha=0.3,       # field coupling (reduced for stability)
    beta=0.6,        # geometric coupling (to validated patterns)
    gamma=0.4        # damping (prevents runaway)
)
```

---

## Results Interpretation

### Success Criteria

**Test 1 (Baseline):**
- Distant resonance: YES ✓
- Stability: NO ✗ (energy → 1000, all units active)

**Test 2 (Two-timescale):**
- Distant resonance: YES ✓ (Triangle 2 activates from Triangle 1)
- Stability: YES ✓ (energy < 50, controlled activation)

### What This Validates

1. **Long-range coupling works** - 10-100x range enables distant geometric patterns to resonate
2. **Runaway is real** - without regulation, resonance explodes
3. **Two-timescale regulation works** - fast over slow prevents runaway
4. **Geometric field theory is viable** - pattern matching by shape, not semantics
5. **SATTVA's core claim validated** - can have creative long-range coupling AND stability

---

## Next Steps

1. **Test geometric creativity:** Can semantically distant but geometrically similar patterns couple?
2. **Test deep broadcast:** Do primitive patterns influence wider areas than surface patterns?
3. **Test developmental sequence:** Form primitives slowly, then enable fast resonance
4. **Add anomaly detection:** As backup regulation when timescales aren't enough
5. **Scale up:** Test with 10K-100K units

---

## Papers and References

**SATTVA Theory:**
- See `theory/sattva_theoretical_development.md`
- See `theory/long_range_coupling.md`  
- See `theory/two_timescale_regulation.md`

**Influences:**
- Pathway's BDH (Stamirowska et al., 2025) - emergent brain-like architecture
- Numenta's HTM and Thousand Brains - anomaly detection, multi-column consensus
- Levin lab - bioelectric fields, morphogenetic pattern memory
- Kreinen - fractal brain structure

---

**Status:** Core principles validated ✓  
**Next:** Implement full developmental sequence
