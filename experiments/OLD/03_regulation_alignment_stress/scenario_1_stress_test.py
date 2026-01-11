#!/usr/bin/env python3
"""
Experiment 03, Scenario 1: Overload Stress Test

Push system toward runaway by increasing field coupling.
Measure regulation response and recovery.
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

def compute_regulation_metrics(substrate):
    """Compute regulation mechanism strengths."""
    
    activations = substrate.activations
    
    # Basin competition: How much are strong patterns suppressing weak?
    active_units = substrate.get_active_pattern(threshold=0.1)
    if len(active_units) > 1:
        active_acts = activations[active_units]
        competition = np.std(active_acts)  # High std = strong competition
    else:
        competition = 0.0
    
    # Local inhibition: Negative correlation with neighbors
    positions = substrate.positions
    inhibition = 0.0
    n_samples = min(100, substrate.n_units)
    sample_units = np.random.choice(substrate.n_units, n_samples, replace=False)
    
    for unit in sample_units:
        # Find nearby units
        distances = np.linalg.norm(positions - positions[unit:unit+1], axis=1)
        nearby = (distances < 0.1) & (distances > 0)
        
        if np.sum(nearby) > 0:
            # If unit is active, are neighbors suppressed?
            if activations[unit] > 0.3:
                nearby_mean = np.mean(activations[nearby])
                inhibition += (activations[unit] - nearby_mean)
    
    inhibition /= n_samples
    
    # Homeostatic balance: How far from equilibrium?
    mean_act = np.mean(activations)
    homeostasis = 1.0 / (1.0 + abs(mean_act - 0.3))  # Target ~0.3
    
    return {
        'competition': competition,
        'inhibition': max(0, inhibition),
        'homeostasis': homeostasis
    }

def run_stress_test():
    """Run overload stress test."""
    
    print("="*60)
    print("Experiment 03, Scenario 1: Overload Stress Test")
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
    
    # Sparse initial activation - only activate 50 random units
    n_initial_active = 50
    active_indices = np.random.choice(1000, size=n_initial_active, replace=False)
    substrate.activations = np.zeros(1000)
    substrate.activations[active_indices] = np.random.rand(n_initial_active) * 0.3
    
    # Create dynamics
    dynamics = SATTVADynamics(
        substrate=substrate,
        alpha=0.5,   # Field coupling (will increase)
        beta=0.3,    # Geometric resonance
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
        'inhibition': [],
        'homeostasis': [],
        'alpha': []
    }
    
    def record_metrics(phase, step):
        """Record current metrics."""
        reg_metrics = compute_regulation_metrics(substrate)
        
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
        results['inhibition'].append(reg_metrics['inhibition'])
        results['homeostasis'].append(reg_metrics['homeostasis'])
        results['alpha'].append(dynamics.alpha)
    
    # Phase 1: Baseline
    print("\nPhase 1: Baseline (50 steps)...")
    for step in range(50):
        dynamics.step(dt=0.1)
        record_metrics('baseline', step)
        
        if step % 10 == 0:
            print(f"  Step {step}: mean_act={results['mean_activation'][-1]:.3f}, "
                  f"max_act={results['max_activation'][-1]:.3f}")
    
    baseline_mean = np.mean(results['mean_activation'][-50:])
    baseline_max = np.mean(results['max_activation'][-50:])
    print(f"Baseline established: mean={baseline_mean:.3f}, max={baseline_max:.3f}")
    
    # Phase 2: Stress
    print("\nPhase 2: Increasing Stress (100 steps)...")
    print("  Gradually increasing field coupling (alpha: 0.5 → 1.5)")
    
    for step in range(100):
        # Increase alpha
        dynamics.alpha = 0.5 + 0.01 * step
        
        dynamics.step(dt=0.1)
        record_metrics('stress', 50 + step)
        
        if step % 20 == 0:
            print(f"  Step {step}: alpha={dynamics.alpha:.2f}, "
                  f"max_act={results['max_activation'][-1]:.3f}, "
                  f"competition={results['competition'][-1]:.3f}")
        
        # Check for runaway
        if results['max_activation'][-1] > 0.95:
            print(f"  ⚠️  Near runaway threshold at step {step}!")
    
    stress_max = np.max(results['max_activation'][-100:])
    print(f"Stress phase complete: peak_max={stress_max:.3f}")
    
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
    runaway_prevented = stress_max < 0.98
    print(f"\n✓ Runaway prevention: {'PASS' if runaway_prevented else 'FAIL'}")
    print(f"  Peak activation: {stress_max:.3f} (threshold: 0.98)")
    
    # Did regulation mechanisms activate?
    baseline_comp = np.mean(results['competition'][:50])
    stress_comp = np.mean(results['competition'][50:150])
    comp_increased = stress_comp > baseline_comp * 1.2
    print(f"\n✓ Regulation activation: {'PASS' if comp_increased else 'FAIL'}")
    print(f"  Competition: {baseline_comp:.3f} → {stress_comp:.3f} "
          f"({100*(stress_comp/baseline_comp - 1):.1f}% increase)")
    
    # Did recovery succeed?
    recovery_diff = abs(recovery_mean - baseline_mean) / baseline_mean
    recovery_ok = recovery_diff < 0.10
    print(f"\n✓ Recovery: {'PASS' if recovery_ok else 'FAIL'}")
    print(f"  Baseline mean: {baseline_mean:.3f}")
    print(f"  Recovery mean: {recovery_mean:.3f}")
    print(f"  Difference: {100*recovery_diff:.1f}% (threshold: <10%)")
    
    # Overall
    all_pass = runaway_prevented and comp_increased and recovery_ok
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
    ax.axhline(0.95, color='red', linestyle=':', alpha=0.5, label='Runaway threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Activation')
    ax.set_title('Activation Under Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Regulation
    ax = axes[0, 1]
    ax.plot(steps, results['competition'], label='Basin competition', linewidth=2)
    ax.plot(steps, results['inhibition'], label='Local inhibition', linewidth=2)
    ax.plot(steps, results['homeostasis'], label='Homeostatic balance', linewidth=2)
    ax.axvline(50, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(150, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Regulation Strength')
    ax.set_title('Regulation Mechanisms Response')
    ax.legend()
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
    ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'scenario_1_stress_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    results = run_stress_test()
    plot_path = plot_results(results)
    
    print("\n✓ Experiment 03, Scenario 1 complete!")
    print(f"Results saved to: {plot_path}")
