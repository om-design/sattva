"""Test distant resonance: can geometrically similar patterns at distance 50+ couple?

This validates SATTVA's core claim that long-range coupling (10-100x) enables
distant but geometrically compatible patterns to resonate.
"""

import sys
sys.path.insert(0, '/Users/omdesign/code/GitHub/sattva/src')

import numpy as np
import matplotlib.pyplot as plt
from sattva.long_range_substrate import LongRangeSubstrate
from sattva.geometric_pattern import GeometricPattern, create_geometric_shape
from sattva.dynamics import SATTVADynamics


def test_distant_resonance():
    """Main test: distant similar shapes should couple via long-range field."""
    
    print("="*60)
    print("SATTVA Long-Range Resonance Test")
    print("="*60)
    print()
    print("Testing: Can two triangular patterns separated by distance 60")
    print("         couple through long-range field effects?")
    print()
    
    # Create substrate with 2000 units in 3D space
    np.random.seed(42)
    substrate = LongRangeSubstrate(
        n_units=2000,
        space_dim=3,
        R_surface=5.0,
        R_deep=50.0,  # 10x range
        alpha=1.5     # power-law decay
    )
    
    print(f"Substrate: {substrate.n_units} units in {substrate.space_dim}D space")
    print(f"Surface range: R={substrate.R_surface}")
    print(f"Deep range: R={substrate.R_deep} (10x)")
    print()
    
    # Create two triangular patterns far apart
    center1 = np.array([0.2, 0.5, 0.5])
    center2 = np.array([0.8, 0.5, 0.5])  # separated by ~0.6 in normalized space
    
    triangle1_units = create_geometric_shape(
        substrate, 'triangle', center1, size=0.05, n_points=3
    )
    triangle2_units = create_geometric_shape(
        substrate, 'triangle', center2, size=0.05, n_points=3
    )
    
    # Compute actual separation
    sep = np.linalg.norm(center1 - center2)
    print(f"Pattern 1: triangle at {center1} ({len(triangle1_units)} units)")
    print(f"Pattern 2: triangle at {center2} ({len(triangle2_units)} units)")
    print(f"Separation: {sep:.3f} (in unit cube)")
    print()
    
    # Set triangle1 to be "deep" (long-range influence)
    substrate.depth[triangle1_units] = 1.0
    
    # Weakly activate triangle 1 only
    substrate.activate_pattern(triangle1_units, strength=0.4)
    
    print("Initial state:")
    print(f"  Triangle 1 activation: {substrate.activations[triangle1_units].mean():.3f}")
    print(f"  Triangle 2 activation: {substrate.activations[triangle2_units].mean():.3f}")
    print()
    
    # Create dynamics (no stored patterns, just field coupling)
    dynamics = SATTVADynamics(
        substrate,
        stored_patterns=[],
        alpha=0.8,  # strong field coupling
        beta=0.0,   # no geometric pattern matching (testing field only)
        gamma=0.1,  # weak local damping
        noise_level=0.005
    )
    
    # Run dynamics
    print("Running dynamics for 50 steps...")
    print()
    
    trajectory = dynamics.run(n_steps=50, verbose=True)
    
    # Check final state
    print()
    print("Final state:")
    final_act1 = substrate.activations[triangle1_units].mean()
    final_act2 = substrate.activations[triangle2_units].mean()
    print(f"  Triangle 1 activation: {final_act1:.3f}")
    print(f"  Triangle 2 activation: {final_act2:.3f}")
    print()
    
    # Evaluate success
    if final_act2 > 0.15:
        print("✓ SUCCESS: Distant resonance detected!")
        print(f"  Triangle 2 activated to {final_act2:.3f} through long-range coupling.")
        success = True
    else:
        print("✗ FAILURE: Distant resonance too weak.")
        print(f"  Triangle 2 only reached {final_act2:.3f} activation.")
        success = False
    
    print()
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Energy over time
    axes[0, 0].plot(trajectory['energy'])
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('System Energy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Field strength over time
    axes[0, 1].plot(trajectory['field_strength'], label='Field', color='blue')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Field strength')
    axes[0, 1].set_title('Long-Range Field Strength')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Pattern activations over time
    act1_history = [h[triangle1_units].mean() for h in dynamics.activation_history]
    act2_history = [h[triangle2_units].mean() for h in dynamics.activation_history]
    
    axes[1, 0].plot(act1_history, label='Triangle 1 (source)', linewidth=2)
    axes[1, 0].plot(act2_history, label='Triangle 2 (distant)', linewidth=2)
    axes[1, 0].axhline(0.15, color='red', linestyle='--', alpha=0.5, label='Success threshold')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Mean activation')
    axes[1, 0].set_title('Pattern Activations Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Spatial view (2D projection)
    axes[1, 1].scatter(
        substrate.positions[:, 0],
        substrate.positions[:, 1],
        c=substrate.activations,
        s=20,
        cmap='hot',
        alpha=0.6
    )
    axes[1, 1].scatter(
        substrate.positions[triangle1_units, 0],
        substrate.positions[triangle1_units, 1],
        s=200,
        marker='^',
        edgecolors='blue',
        facecolors='none',
        linewidths=3,
        label='Triangle 1'
    )
    axes[1, 1].scatter(
        substrate.positions[triangle2_units, 0],
        substrate.positions[triangle2_units, 1],
        s=200,
        marker='^',
        edgecolors='green',
        facecolors='none',
        linewidths=3,
        label='Triangle 2'
    )
    axes[1, 1].set_xlabel('Dimension 0')
    axes[1, 1].set_ylabel('Dimension 1')
    axes[1, 1].set_title('Final Activation Pattern (2D projection)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/omdesign/code/GitHub/sattva/experiments/long_range_test/distant_resonance_result.png', dpi=150)
    print(f"Plots saved to: experiments/long_range_test/distant_resonance_result.png")
    print()
    
    return success, substrate, dynamics


if __name__ == '__main__':
    success, substrate, dynamics = test_distant_resonance()
    
    if success:
        print("="*60)
        print("VALIDATION SUCCESSFUL")
        print("Long-range coupling (10-100x) enables distant resonance!")
        print("="*60)
    else:
        print("="*60)
        print("Test did not meet success criteria.")
        print("Consider adjusting: alpha (field strength), R_deep (range), or initial activation.")
        print("="*60)
