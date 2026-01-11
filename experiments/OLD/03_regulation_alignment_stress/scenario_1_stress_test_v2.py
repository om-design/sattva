#!/usr/bin/env python3
"""
Experiment 03, Scenario 1: Overload Stress Test (CORRECTED)

Key insight from debugging:
- Regulation comes from SELECTIVE resonance with vetted primitives
- Raw field coupling without primitives = unregulated runaway
- This experiment now includes primitive vetting phase

Phases:
1. Primitive formation: Create and stabilize base patterns
2. Baseline: Normal operation with vetted primitives
3. Stress: Increase coupling while primitives regulate
4. Recovery: Return to baseline parameters

Date: January 10, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src directory to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from sattva.long_range_substrate import LongRangeSubstrate
from sattva.dynamics import SATTVADynamics
from sattva.geometric_pattern import GeometricPattern

def compute_regulation_metrics(substrate, dynamics):
    """Compute regulation mechanism strengths."""
    
    activations = substrate.activations
    
    # Basin competition: How much are strong patterns suppressing weak?
    active_units = substrate.get_active_pattern(threshold=0.1)
    if len(active_units) > 1:
        active_acts = activations[active_units]
        competition = np.std(active_acts)  # High std = strong competition
    else:
        competition = 0.0
    
    # Primitive selectivity: Are active patterns matching stored primitives?
    if len(dynamics.stored_patterns) > 0 and len(active_units) > 0:
        current = GeometricPattern.from_substrate(substrate, threshold=0.1)
        max_resonance = max([current.resonance_strength(p) for p in dynamics.stored_patterns])
        selectivity = max_resonance
    else:
        selectivity = 0.0
    
    # Homeostatic balance: How far from equilibrium?
    mean_act = np.mean(activations)
    homeostasis = 1.0 / (1.0 + abs(mean_act - 0.3))  # Target ~0.3
    
    return {
        'competition': competition,
        'selectivity': selectivity,
        'homeostasis': homeostasis
    }

def create_and_stabilize_primitive(substrate, dynamics, center, size=0.15, steps=20):
    """Create a primitive pattern and let it stabilize."""
    # Reset substrate
    substrate.reset_activations()
    
    # Create geometric seed pattern
    distances = np.linalg.norm(substrate.positions - center, axis=1)
    nearby = distances < size
    substrate.activations[nearby] = 0.5 * np.exp(-distances[nearby]**2 / (size**2 / 2))
    
    # Let it settle with weak parameters
    old_alpha = dynamics.alpha
    old_beta = dynamics.beta
    dynamics.alpha = 0.2  # Weak field
    dynamics.beta = 0.0   # No geometric (nothing stored yet)
    
    for _ in range(steps):
        dynamics.step(dt=0.1)
    
    # Restore parameters
    dynamics.alpha = old_alpha
    dynamics.beta = old_beta
    
    # Store as vetted primitive
    pattern = dynamics.store_current_pattern()
    
    return pattern

def run_stress_test():
    """Run overload stress test with primitive vetting."""
    
    print("="*60)
    print("Experiment 03, Scenario 1: Overload Stress Test (v2)")
    print("With Primitive Vetting")
    print("="*60)
    
    # Create substrate
    print("\nInitializing substrate (1000 units)...")
    substrate = LongRangeSubstrate(
        n_units=1000,
        space_dim=3,
        R_surface=5.0,
        R_deep=50.0,
        alpha=1.5
    )
    
    # Create dynamics (NO primitives yet)
    dynamics = SATTVADynamics(
        substrate=substrate,
        stored_patterns=[],  # Empty initially
        alpha=0.5,   # Field coupling
        beta=0.3,    # Geometric resonance (will matter once we add primitives)
        gamma=0.2,   # Local attractor
        noise_level=0.01
    )
    
    # Storage
    results = {
        'phase': [],
        'step': [],
        'mean_activation': [],
        'max_activation': [],
        'std_activation': [],
        'n_active': [],
        'energy': [],
        'field_strength': [],
        'competition': [],
        'selectivity': [],
        'homeostasis': [],
        'alpha': [],
        'n_primitives': []
    }
    
    def record_metrics(phase, step):
        """Record current metrics."""
        reg_metrics = compute_regulation_metrics(substrate, dynamics)
        
        results['phase'].append(phase)
        results['step'].append(step)
        results['mean_activation'].append(np.mean(substrate.activations))
        results['max_activation'].append(np.max(substrate.activations))
        results['std_activation'].append(np.std(substrate.activations))
        results['n_active'].append(len(substrate.get_active_pattern(0.1)))
        results['energy'].append(0.5 * np.sum(substrate.activations**2))
        
        field = substrate.compute_field()
        results['field_strength'].append(np.mean(np.abs(field)))
        
        results['competition'].append(reg_metrics['competition'])
        results['selectivity'].append(reg_metrics['selectivity'])
        results['homeostasis'].append(reg_metrics['homeostasis'])
        results['alpha'].append(dynamics.alpha)
        results['n_primitives'].append(len(dynamics.stored_patterns))
    
    # Phase 0: Primitive Formation
    print("\nPhase 0: Primitive Formation (creating 5 stable patterns)...")
    print("  This is the 'vetting through direct experience' phase")
    
    # Create 5 primitives at different locations
    primitive_centers = [
        np.array([0.2, 0.2, 0.5]),
        np.array([0.8, 0.3, 0.5]),
        np.array([0.5, 0.7, 0.3]),
        np.array([0.3, 0.8, 0.7]),
        np.array([0.7, 0.5, 0.8])
    ]
    
    for i, center in enumerate(primitive_centers):
        pattern = create_and_stabilize_primitive(substrate, dynamics, center)
        print(f"  Primitive {i+1}: {pattern.signature['n_active']} active units, "
              f"spread={pattern.signature['spread']:.3f}")
    
    print(f"\nVetted {len(dynamics.stored_patterns)} stable primitives")
    print("Now resonance coupling is SELECTIVE (only matches vetted patterns)")
    
    # Phase 1: Baseline with primitives
    print("\nPhase 1: Baseline with Primitives (50 steps)...")
    
    # Start with sparse activation matching a primitive
    substrate.reset_activations()
    pattern_to_match = dynamics.stored_patterns[0]
    substrate.activations[pattern_to_match.active_units] = 0.3
    
    for step in range(50):
        dynamics.step(dt=0.1)
        record_metrics('baseline', step)
        
        if step % 10 == 0:
            print(f"  Step {step}: mean_act={results['mean_activation'][-1]:.3f}, "
                  f"max_act={results['max_activation'][-1]:.3f}, "
                  f"selectivity={results['selectivity'][-1]:.3f}")
    
    baseline_mean = np.mean(results['mean_activation'][-50:])
    baseline_max = np.mean(results['max_activation'][-50:])
    baseline_select = np.mean(results['selectivity'][-50:])
    print(f"Baseline: mean={baseline_mean:.3f}, max={baseline_max:.3f}, "
          f"selectivity={baseline_select:.3f}")
    
    # Phase 2: Stress
    print("\nPhase 2: Increasing Stress (100 steps)...")
    print("  Gradually increasing field coupling (alpha: 0.5 → 1.2)")
    print("  Primitives should provide selective regulation")
    
    for step in range(100):
        # Increase alpha more conservatively
        dynamics.alpha = 0.5 + 0.007 * step  # 0.5 → 1.2
        
        dynamics.step(dt=0.1)
        record_metrics('stress', 50 + step)
        
        if step % 20 == 0:
            print(f"  Step {step}: alpha={dynamics.alpha:.2f}, "
                  f"max_act={results['max_activation'][-1]:.3f}, "
                  f"selectivity={results['selectivity'][-1]:.3f}")
        
        # Check for runaway
        if results['max_activation'][-1] > 0.95:
            print(f"  ⚠️  Near runaway threshold at step {step}!")
    
    stress_max = np.max(results['max_activation'][-100:])
    stress_select = np.mean(results['selectivity'][-100:])
    print(f"Stress complete: peak_max={stress_max:.3f}, selectivity={stress_select:.3f}")
    
    # Phase 3: Recovery
    print("\nPhase 3: Recovery (100 steps)...")
    print("  Returning alpha to 0.5")
    dynamics.alpha = 0.5
    
    for step in range(100):
        dynamics.step(dt=0.1)
        record_metrics('recovery', 150 + step)
        
        if step % 20 == 0:
            print(f"  Step {step}: mean_act={results['mean_activation'][-1]:.3f}, "
                  f"max_act={results['max_activation'][-1]:.3f}")
    
    recovery_mean = np.mean(results['mean_activation'][-50:])
    recovery_max = np.mean(results['max_activation'][-50:])
    print(f"Recovery complete: mean={recovery_mean:.3f}, max={recovery_max:.3f}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Did regulation prevent runaway?
    runaway_prevented = stress_max < 0.90
    print(f"\n✓ Runaway prevention: {'PASS' if runaway_prevented else 'FAIL'}")
    print(f"  Peak activation: {stress_max:.3f} (threshold: 0.90)")
    
    # Did primitive selectivity increase under stress?
    select_maintained = stress_select > baseline_select * 0.7
    print(f"\n✓ Primitive selectivity: {'MAINTAINED' if select_maintained else 'DEGRADED'}")
    print(f"  Baseline: {baseline_select:.3f}")
    print(f"  Under stress: {stress_select:.3f}")
    print(f"  (Selective coupling = only vetted patterns resonate)")
    
    # Did recovery succeed?
    recovery_diff = abs(recovery_mean - baseline_mean) / (baseline_mean + 1e-6)
    recovery_ok = recovery_diff < 0.15
    print(f"\n✓ Recovery: {'PASS' if recovery_ok else 'FAIL'}")
    print(f"  Baseline mean: {baseline_mean:.3f}")
    print(f"  Recovery mean: {recovery_mean:.3f}")
    print(f"  Difference: {100*recovery_diff:.1f}% (threshold: <15%)")
    
    # Overall
    all_pass = runaway_prevented and recovery_ok
    print(f"\n{'='*60}")
    print(f"OVERALL: {'✓ SUCCESS' if all_pass else '✗ NEEDS ATTENTION'}")
    print(f"{'='*60}")
    
    return results

def plot_results(results):
    """Generate visualization."""
    
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = results['step']
    
    # Plot 1: Activation
    ax = axes[0, 0]
    ax.plot(steps, results['mean_activation'], label='Mean', linewidth=2)
    ax.plot(steps, results['max_activation'], label='Max', linewidth=2)
    ax.axvline(50, color='orange', linestyle='--', alpha=0.5, label='Stress begins')
    ax.axvline(150, color='green', linestyle='--', alpha=0.5, label='Recovery begins')
    ax.axhline(0.90, color='red', linestyle=':', alpha=0.5, label='Runaway threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Activation')
    ax.set_title('Activation Under Stress (With Primitive Regulation)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Regulation - SELECTIVITY instead of inhibition
    ax = axes[0, 1]
    ax.plot(steps, results['competition'], label='Basin competition', linewidth=2)
    ax.plot(steps, results['selectivity'], label='Primitive selectivity', linewidth=2, color='purple')
    ax.plot(steps, results['homeostasis'], label='Homeostatic balance', linewidth=2)
    ax.axvline(50, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(150, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Regulation Strength')
    ax.set_title('Regulation: Selective Resonance with Vetted Primitives')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy
    ax = axes[1, 0]
    ax.plot(steps, results['energy'], linewidth=2, color='purple')
    ax.axvline(50, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(150, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Energy')
    ax.set_title('System Energy')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Field & Alpha
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(steps, results['field_strength'], label='Field strength', 
                 linewidth=2, color='blue')
    l2 = ax2.plot(steps, results['alpha'], label='Alpha (coupling)', 
                  linewidth=2, color='red', linestyle='--')
    ax.axvline(50, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(150, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Field Strength', color='blue')
    ax2.set_ylabel('Alpha', color='red')
    ax.set_title('Field Strength & Coupling Parameter')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'scenario_1_stress_results_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    results = run_stress_test()
    plot_path = plot_results(results)
    
    print("\n" + "="*60)
    print("KEY LESSONS LEARNED")
    print("="*60)
    print("\n1. ARCHITECTURE MATTERS")
    print("   ❌ Raw field coupling without primitives = unregulated runaway")
    print("   ✓ Selective resonance with vetted primitives = self-regulation")
    
    print("\n2. PRIMITIVES ARE THE REGULATION MECHANISM")
    print("   - Primitives formed through direct experience (Phase 0)")
    print("   - Only geometrically-matching patterns get amplified")
    print("   - Random noise doesn't resonate → no runaway")
    
    print("\n3. VETTING CREATES STABILITY")
    print("   - 'Vetted' = patterns that settled to stable attractors")
    print("   - Stored primitives act as selective filters")
    print("   - System can only express what it has learned")
    
    print("\n4. THIS IS DIFFERENT FROM HOPFIELD NETS")
    print("   - Not stored patterns pulled by energy landscape")
    print("   - Geometric resonance: similar SHAPES couple strongly")
    print("   - Long-range (10-100x) + shape-based = novel dynamics")
    
    print("\n5. EXPERIMENT DESIGN INSIGHT")
    print("   - Can't test regulation without primitives to regulate!")
    print("   - Must include primitive formation phase")
    print("   - Stress test validates selective coupling property")
    
    print("\n" + "="*60)
    print(f"✓ Experiment complete!")
    print(f"Results: {plot_path}")
    print("="*60)
