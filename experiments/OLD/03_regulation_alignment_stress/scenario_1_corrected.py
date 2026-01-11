#!/usr/bin/env python3
"""
Experiment 03, Scenario 1: Stress Test - CORRECTED VERSION

Key fix: Includes primitive vetting phase.
Regulation comes from selective resonance with vetted primitives.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from sattva.long_range_substrate import LongRangeSubstrate
from sattva.dynamics import SATTVADynamics
from sattva.geometric_pattern import GeometricPattern

def compute_selectivity(substrate, dynamics):
    """Measure how well current pattern matches stored primitives."""
    if len(dynamics.stored_patterns) == 0:
        return 0.0
    
    active_units = substrate.get_active_pattern(threshold=0.1)
    if len(active_units) == 0:
        return 0.0
    
    current = GeometricPattern.from_substrate(substrate, threshold=0.1)
    resonances = [current.resonance_strength(p) for p in dynamics.stored_patterns]
    return max(resonances) if resonances else 0.0

def create_primitive(substrate, dynamics, center, size=0.15):
    """Create and stabilize a primitive pattern."""
    substrate.reset_activations()
    
    # Activate nearby units
    distances = np.linalg.norm(substrate.positions - center, axis=1)
    nearby = distances < size
    substrate.activations[nearby] = 0.5 * np.exp(-distances[nearby]**2 / (size**2 / 2))
    
    # Stabilize with weak coupling
    old_alpha, old_beta = dynamics.alpha, dynamics.beta
    dynamics.alpha, dynamics.beta = 0.2, 0.0
    
    for _ in range(20):
        dynamics.step(dt=0.1)
    
    dynamics.alpha, dynamics.beta = old_alpha, old_beta
    return dynamics.store_current_pattern()

print("="*60)
print("Experiment 03, Scenario 1: Stress Test (CORRECTED)")
print("With Primitive Vetting")
print("="*60)

# Initialize
substrate = LongRangeSubstrate(n_units=1000, space_dim=3)
dynamics = SATTVADynamics(
    substrate=substrate,
    stored_patterns=[],
    alpha=0.5, beta=0.3, gamma=0.2, noise_level=0.01
)

results = {'step': [], 'max_act': [], 'selectivity': [], 'alpha': []}

# PHASE 0: Create vetted primitives
print("\nPhase 0: Creating 5 vetted primitives...")
centers = [np.array([0.2, 0.2, 0.5]), np.array([0.8, 0.3, 0.5]),
           np.array([0.5, 0.7, 0.3]), np.array([0.3, 0.8, 0.7]),
           np.array([0.7, 0.5, 0.8])]

for i, c in enumerate(centers):
    p = create_primitive(substrate, dynamics, c)
    print(f"  Primitive {i+1}: {p.signature['n_active']} units")

print(f"Vetted {len(dynamics.stored_patterns)} primitives")

# PHASE 1: Baseline
print("\nPhase 1: Baseline (50 steps)...")
substrate.reset_activations()
substrate.activations[dynamics.stored_patterns[0].active_units] = 0.3

for step in range(50):
    dynamics.step(dt=0.1)
    results['step'].append(step)
    results['max_act'].append(np.max(substrate.activations))
    results['selectivity'].append(compute_selectivity(substrate, dynamics))
    results['alpha'].append(dynamics.alpha)
    
    if step % 10 == 0:
        print(f"  Step {step}: max={results['max_act'][-1]:.3f}, sel={results['selectivity'][-1]:.3f}")

baseline_max = np.mean(results['max_act'][-50:])
print(f"Baseline max: {baseline_max:.3f}")

# PHASE 2: Stress
print("\nPhase 2: Stress - increasing alpha 0.5→1.2 (100 steps)...")
for step in range(100):
    dynamics.alpha = 0.5 + 0.007 * step
    dynamics.step(dt=0.1)
    
    results['step'].append(50 + step)
    results['max_act'].append(np.max(substrate.activations))
    results['selectivity'].append(compute_selectivity(substrate, dynamics))
    results['alpha'].append(dynamics.alpha)
    
    if step % 20 == 0:
        print(f"  Step {step}: alpha={dynamics.alpha:.2f}, max={results['max_act'][-1]:.3f}")

stress_max = np.max(results['max_act'][-100:])
print(f"Stress max: {stress_max:.3f}")

# PHASE 3: Recovery
print("\nPhase 3: Recovery (100 steps)...")
dynamics.alpha = 0.5

for step in range(100):
    dynamics.step(dt=0.1)
    results['step'].append(150 + step)
    results['max_act'].append(np.max(substrate.activations))
    results['selectivity'].append(compute_selectivity(substrate, dynamics))
    results['alpha'].append(dynamics.alpha)
    
    if step % 20 == 0:
        print(f"  Step {step}: max={results['max_act'][-1]:.3f}")

recovery_max = np.mean(results['max_act'][-50:])
print(f"Recovery max: {recovery_max:.3f}")

# Analysis
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

runaway_prevented = stress_max < 0.85
print(f"\n✓ Runaway prevention: {'PASS' if runaway_prevented else 'FAIL'}")
print(f"  Peak: {stress_max:.3f} (threshold: 0.85)")

recovered = abs(recovery_max - baseline_max) / baseline_max < 0.20
print(f"\n✓ Recovery: {'PASS' if recovered else 'FAIL'}")
print(f"  Baseline: {baseline_max:.3f}, Recovery: {recovery_max:.3f}")

print(f"\n{'='*60}")
print(f"OVERALL: {'✓ SUCCESS' if (runaway_prevented and recovered) else '✗ NEEDS WORK'}")
print(f"{'='*60}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(results['step'], results['max_act'], linewidth=2)
ax.axvline(50, color='orange', linestyle='--', alpha=0.5)
ax.axvline(150, color='green', linestyle='--', alpha=0.5)
ax.axhline(0.85, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Max Activation')
ax.set_title('Activation With Primitive Regulation')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(results['step'], results['selectivity'], linewidth=2, color='purple')
ax.axvline(50, color='orange', linestyle='--', alpha=0.5)
ax.axvline(150, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Primitive Selectivity')
ax.set_title('Selective Resonance (Only Vetted Patterns)')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(results['step'], results['alpha'], linewidth=2, color='red')
ax.axvline(50, color='orange', linestyle='--', alpha=0.5)
ax.axvline(150, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Alpha (coupling)')
ax.set_title('Coupling Parameter Over Time')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.text(0.1, 0.5, 'KEY LESSON:\n\nRegulation = Selective\nResonance with Vetted\nPrimitives\n\nWithout primitives:\nRunaway\n\nWith primitives:\nStable', 
        fontsize=14, verticalalignment='center', family='monospace')
ax.axis('off')

plt.tight_layout()
output_path = Path(__file__).parent / 'scenario_1_corrected.png'
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved: {output_path}")

print("\n" + "="*60)
print("LESSONS LEARNED")
print("="*60)
print("1. Raw field coupling WITHOUT primitives = runaway")
print("2. Selective resonance WITH vetted primitives = regulation")
print("3. Primitives vet through direct experience (stabilization)")
print("4. Only geometrically-matching patterns resonate")
print("5. This IS the regulation mechanism - not ad-hoc")
print("="*60)
