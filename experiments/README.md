# SATTVA Experiments

## Experiment 01: Clustering, Resonance, and Self-Regulation

**File:** `01_clustering_resonance_regulation.py`

### Objective

Test the fundamental hypothesis that geometric clustering leads to resonance, and explore whether this creates runaway activation or if self-regulation mechanisms can maintain stability.

### What It Tests

1. **Natural Clustering**: Do similar patterns naturally cluster in geometric space?
2. **Resonance Spreading**: Does activation spread within clusters?
3. **Runaway Risk**: Without regulation, does the system become unstable?
4. **Self-Regulation**: Which regulation mechanisms maintain stability?

### Experimental Setup

**Substrate:**
- 500 neurons in 3D space
- Sparse connections based on spatial proximity
- Connection strength inversely proportional to distance

**Test Pattern:**
- 3 random patterns presented sequentially
- Each pattern activates neurons based on geometric similarity
- 20 time steps per pattern with activation propagation

**Regulation Mechanisms Tested:**

1. **No Regulation** (Baseline)
   - Test for runaway activation
   - Shows what happens without control

2. **Global Normalization**
   - Maintains constant total activity
   - Scales all activations proportionally

3. **Local Inhibition**
   - Nearby neurons inhibit each other
   - Models lateral inhibition in cortex

4. **Homeostatic Regulation**
   - Maintains average activity level over time
   - Adaptive scaling based on recent history

### Metrics Tracked

- **Total Activity**: Sum of all neuron activations
- **Number of Clusters**: Active neuron groups
- **Max Activation**: Highest individual neuron activation
- **Cluster Size**: Average neurons per cluster

### Expected Outcomes

**If clustering → resonance works:**
- Similar patterns should activate overlapping neurons
- Activation should spread within clusters
- Distinct clusters should form for different patterns

**If runaway is a problem:**
- No Regulation condition should show exponential growth
- Activity should exceed sustainable levels

**If self-regulation works:**
- Regulated conditions should maintain stable activity
- Clusters should form and persist
- Different patterns should activate different clusters

### How to Run

```bash
cd /Users/omdesign/code/GitHub/sattva/experiments
python 01_clustering_resonance_regulation.py
```

**Requirements:**
- numpy
- matplotlib

**Output:**
- Console logging of experiment progress
- `01_results.png` - Four-panel visualization:
  - Total network activity over time
  - Number of clusters formed
  - Peak activation levels
  - Average cluster sizes

### Interpretation Guide

**Successful Results:**
- Regulated conditions show stable, bounded activity
- Clusters form in response to patterns
- Multiple distinct clusters can coexist
- Resonance spreads within but not between unrelated clusters

**Failure Modes:**
- Runaway: Activity grows exponentially
- Over-suppression: Activity drops to near zero
- No clustering: Activation spreads uniformly
- No resonance: Activation doesn't propagate

### Connection to Theory

This experiment tests the core principle of geometric multiplexing:
- Same connections serve multiple purposes (different patterns)
- Context (input pattern) selects which clusters resonate
- Regulation prevents runaway while maintaining function

If successful, this demonstrates:
1. Natural clustering emerges from geometry
2. Resonance enables pattern recognition
3. Self-regulation maintains stability
4. Multiple patterns can coexist (multiplexing)

---

## Future Experiments

**Experiment 02**: Cross-cluster resonance (reasoning)
**Experiment 03**: Distant cluster activation (creativity)
**Experiment 04**: Multi-agent peer validation
**Experiment 05**: Physical grounding and invariant learning
**Experiment 06**: BIAS mechanism calibration
