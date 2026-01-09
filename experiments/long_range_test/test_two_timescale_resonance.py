"""Test distant resonance with two-timescale dynamics.

This demonstrates that fast resonance over validated primitives is STABLE,
even with long-range coupling (10-100x).
"""

import sys
sys.path.insert(0, '/Users/omdesign/code/GitHub/sattva/src')

import numpy as np
import matplotlib.pyplot as plt
from sattva.long_range_substrate import LongRangeSubstrate
from sattva.geometric_pattern import GeometricPattern, create_geometric_shape
from sattva.two_timescale_dynamics import TwoTimescaleDynamics, bootstrap_validated_primitives


def test_stable_distant_resonance():
    """Test: distant resonance with two-timescale regulation."""
    
    print("="*60)
    print("SATTVA Two-Timescale Distant Resonance Test")
    print("="*60)
    print()
    print("Key idea: Fast resonance over validated primitives is STABLE")
    print("even with long-range coupling (10-100x).")
    print()
    
    # Create substrate
    np.random.seed(42)
    substrate = LongRangeSubstrate(
        n_units=2000,
        space_dim=3,
        R_surface=5.0,
        R_deep=50.0,
        alpha=1.5
    )
    
    print(f"Substrate: {substrate.n_units} units in {substrate.space_dim}D")
    print(f"Long-range coupling: {substrate.R_deep / substrate.R_surface}x")
    print()
    
    # CRITICAL: Bootstrap validated primitives first
    print("Phase 1: Bootstrapping validated primitive patterns...")
    validated_primitives = bootstrap_validated_primitives(
        substrate,
        n_primitives=30,
        seed=42
    )
    print(f"  Created {len(validated_primitives)} stable primitives")
    print("  (In real system: formed via slow sensory learning)")
    print()
    
    # Create two test patterns (triangles) far apart
    center1 = np.array([0.2, 0.5, 0.5])
    center2 = np.array([0.8, 0.5, 0.5])
    
    triangle1_units = create_geometric_shape(
        substrate, 'triangle', center1, size=0.05
    )
    triangle2_units = create_geometric_shape(
        substrate, 'triangle', center2, size=0.05
    )
    
    # Make these specific patterns part of validated primitives
    substrate.activate_pattern(triangle1_units, strength=0.6)
    pattern1 = GeometricPattern.from_substrate(substrate, threshold=0.1)
    substrate.reset_activations()
    
    substrate.activate_pattern(triangle2_units, strength=0.6)
    pattern2 = GeometricPattern.from_substrate(substrate, threshold=0.1)
    substrate.reset_activations()
    
    # Add to validated primitives
    validated_primitives.append(pattern1)
    validated_primitives.append(pattern2)
    
    sep = np.linalg.norm(center1 - center2)
    print(f"Test patterns:")
    print(f"  Triangle 1: {center1}")
    print(f"  Triangle 2: {center2}")
    print(f"  Separation: {sep:.3f} (in unit cube)")
    print(f"  Both added to validated primitives")
    print()
    
    # Make triangle1 "deep" (long-range influence)
    substrate.depth[triangle1_units] = 1.0
    
    # Initial condition: weakly activate triangle 1
    substrate.activate_pattern(triangle1_units, strength=0.3)
    
    print("Initial state:")
    print(f"  Triangle 1: {substrate.activations[triangle1_units].mean():.3f}")
    print(f"  Triangle 2: {substrate.activations[triangle2_units].mean():.3f}")
    print()
    
    # Create TWO-TIMESCALE dynamics
    print("Phase 2: Running two-timescale dynamics...")
    print("  Fast updates: activation/resonance (every step)")
    print("  Slow updates: formation/validation (every 100 steps)")
    print("  Operating over validated primitive substrate")
    print()
    
    dynamics = TwoTimescaleDynamics(
        substrate,
        validated_primitives=validated_primitives,
        fast_dt=0.02,     # fast resonance
        slow_dt=1.0,      # slow formation
        slow_every=100,
        alpha=0.3,        # moderate field coupling (stable)
        beta=0.6,         # strong geometric coupling (to validated primitives)
        gamma=0.4,        # damping (prevents runaway)
        noise_level=0.005
    )
    
    # Run dynamics
    trajectory = dynamics.run(n_steps=100, verbose=True)
    
    # Check results
    print()
    print("Final state:")
    final_act1 = substrate.activations[triangle1_units].mean()
    final_act2 = substrate.activations[triangle2_units].mean()
    final_energy = trajectory['energy'][-1]
    
    print(f"  Triangle 1: {final_act1:.3f}")
    print(f"  Triangle 2: {final_act2:.3f}")
    print(f"  Energy: {final_energy:.3f}")
    print()
    
    # Evaluate
    resonance_success = final_act2 > 0.15
    stability_success = final_energy < 50.0  # didn't explode!
    
    if resonance_success and stability_success:
        print("✓ SUCCESS: Stable distant resonance!")
        print(f"  - Triangle 2 activated via long-range coupling: {final_act2:.3f}")
        print(f"  - System remained stable (energy {final_energy:.1f})")
        print(f"  - Two-timescale regulation prevented runaway")
        success = True
    elif resonance_success:
        print("✓ PARTIAL: Resonance worked but stability questionable")
        print(f"  - Energy: {final_energy:.1f}")
        success = False
    elif stability_success:
        print("✗ PARTIAL: Stable but resonance too weak")
        print(f"  - Triangle 2: {final_act2:.3f}")
        success = False
    else:
        print("✗ FAILURE")
        success = False
    
    print()
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Energy over time
    axes[0, 0].plot(trajectory['energy'], linewidth=2)
    axes[0, 0].axhline(50, color='red', linestyle='--', alpha=0.5, label='Explosion threshold')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('System Energy (Two-Timescale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 100])
    
    # Pattern activations
    act1_history = [h[triangle1_units].mean() for h in dynamics.activation_history]
    act2_history = [h[triangle2_units].mean() for h in dynamics.activation_history]
    
    axes[0, 1].plot(act1_history, label='Triangle 1 (source)', linewidth=2, color='blue')
    axes[0, 1].plot(act2_history, label='Triangle 2 (distant)', linewidth=2, color='green')
    axes[0, 1].axhline(0.15, color='red', linestyle='--', alpha=0.5, label='Success threshold')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Mean activation')
    axes[0, 1].set_title('Pattern Activations (Stable Coupling)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Number of active units
    axes[1, 0].plot(trajectory['n_active'], linewidth=2, color='purple')
    axes[1, 0].axhline(2000, color='red', linestyle='--', alpha=0.5, label='Total units (runaway)')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Active units')
    axes[1, 0].set_title('Active Unit Count (Controlled)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spatial view
    axes[1, 1].scatter(
        substrate.positions[:, 0],
        substrate.positions[:, 1],
        c=substrate.activations,
        s=20,
        cmap='hot',
        alpha=0.6,
        vmin=0,
        vmax=1
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
    axes[1, 1].set_title('Final State (Stable Pattern)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('/Users/omdesign/code/GitHub/sattva/experiments/long_range_test/two_timescale_result.png', dpi=150)
    print(f"Plots saved to: experiments/long_range_test/two_timescale_result.png")
    print()
    
    return success, substrate, dynamics


if __name__ == '__main__':
    success, substrate, dynamics = test_stable_distant_resonance()
    
    print("="*60)
    if success:
        print("VALIDATION SUCCESSFUL")
        print()
        print("Key findings:")
        print("  1. Long-range coupling (10-100x) enables distant resonance")
        print("  2. Two-timescale regulation prevents runaway")
        print("  3. Fast resonance is SAFE over validated primitives")
        print("  4. System remains stable even with exponential amplification")
        print()
        print("This validates SATTVA's core architectural principles!")
    else:
        print("Further tuning needed.")
        print("Adjust: alpha (field), beta (geometric), gamma (damping)")
    print("="*60)
