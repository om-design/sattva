"""Two-timescale dynamics for SATTVA: fast resonance over validated primitives.

Key insight: Fast resonance is safe when operating over slowly-formed,
validated geometric primitives.
"""

import numpy as np
from typing import List, Optional, Dict
from .long_range_substrate import LongRangeSubstrate
from .geometric_pattern import GeometricPattern


class TwoTimescaleDynamics:
    """SATTVA dynamics with separated fast (resonance) and slow (formation) timescales.
    
    Fast timescale: Pattern activation and resonance (milliseconds)
    Slow timescale: Pattern formation and validation (hours/days)
    
    This separation prevents runaway by ensuring fast resonance operates
    only over pre-validated stable patterns.
    """
    
    def __init__(self,
                 substrate: LongRangeSubstrate,
                 validated_primitives: List[GeometricPattern],
                 fast_dt: float = 0.01,
                 slow_dt: float = 1.0,
                 slow_every: int = 100,
                 alpha: float = 0.3,  # field coupling (reduced for stability)
                 beta: float = 0.5,   # geometric coupling
                 gamma: float = 0.4,  # damping
                 noise_level: float = 0.01):
        """
        Args:
            substrate: the long-range substrate
            validated_primitives: pre-validated geometric patterns (the stable substrate)
            fast_dt: fast timestep for activation dynamics
            slow_dt: slow timestep for formation dynamics
            slow_every: run slow update every N fast steps
            alpha: long-range field coupling strength
            beta: geometric pattern coupling strength
            gamma: damping strength (prevents runaway)
            noise_level: exploration noise
        """
        self.substrate = substrate
        self.validated_primitives = validated_primitives
        self.forming_patterns: List[Dict] = []  # candidate patterns
        
        self.fast_dt = fast_dt
        self.slow_dt = slow_dt
        self.slow_every = slow_every
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_level = noise_level
        
        # Tracking
        self.step_count = 0
        self.activation_history = []
        self.energy_history = []
        
    def energy(self) -> float:
        """Compute current energy."""
        return 0.5 * np.sum(self.substrate.activations ** 2)
    
    def fast_update(self) -> Dict:
        """Fast dynamics: activation and resonance over validated primitives only.
        
        This is SAFE because we're only operating over pre-validated patterns.
        """
        # 1. Long-range field from all activations
        field = self.substrate.compute_field(threshold=0.05)
        
        # 2. Geometric coupling to validated primitives
        current = GeometricPattern.from_substrate(self.substrate, threshold=0.05)
        geometric_force = np.zeros(self.substrate.n_units)
        
        if current.signature['n_active'] > 0:
            for primitive in self.validated_primitives:
                resonance = current.resonance_strength(primitive)
                
                if resonance > 0.2:  # significant match
                    # Pull toward validated primitive
                    for i, unit_idx in enumerate(primitive.active_units):
                        target = primitive.activations[i]
                        current_act = self.substrate.activations[unit_idx]
                        geometric_force[unit_idx] += resonance * (target - current_act)
        
        # 3. Damping (prevents runaway)
        damping = -self.gamma * self.substrate.activations
        
        # 4. Noise
        noise = self.noise_level * np.random.randn(self.substrate.n_units)
        
        # 5. Total update
        total_force = (
            self.alpha * field +
            self.beta * geometric_force +
            damping +
            noise
        )
        
        # Update with fast timescale
        self.substrate.activations += self.fast_dt * total_force
        
        # Clip to valid range
        self.substrate.activations = np.clip(self.substrate.activations, 0, 1)
        
        # Metrics
        energy = self.energy()
        n_active = len(self.substrate.get_active_pattern(0.1))
        
        return {
            'energy': energy,
            'field_strength': np.mean(np.abs(field)),
            'n_active': n_active,
            'max_activation': np.max(self.substrate.activations)
        }
    
    def slow_update(self):
        """Slow dynamics: pattern formation and validation.
        
        This is where new patterns can form, but it happens rarely
        and requires validation before being added to primitives.
        """
        # Check if any forming patterns have stabilized
        for candidate in self.forming_patterns:
            candidate['age'] += 1
            
            # Check stability (has it persisted?)
            if candidate['age'] > 10:  # persisted for 10 slow steps
                # Validate: does it predict well?
                if candidate['stability_score'] > 0.7:
                    # Promote to validated primitive
                    self.validated_primitives.append(candidate['pattern'])
                    self.forming_patterns.remove(candidate)
                else:
                    # Failed to stabilize, discard
                    self.forming_patterns.remove(candidate)
        
        # Could add: Hebbian strengthening of co-activated primitives
        # But for now, we keep primitives fixed (operating over stable substrate)
    
    def step(self) -> Dict:
        """Take one step (fast update, occasional slow update)."""
        # Fast update every step
        info = self.fast_update()
        
        # Slow update occasionally
        if self.step_count % self.slow_every == 0:
            self.slow_update()
        
        # Track
        self.step_count += 1
        self.activation_history.append(self.substrate.activations.copy())
        self.energy_history.append(info['energy'])
        
        return info
    
    def run(self, n_steps: int, verbose: bool = False) -> Dict:
        """Run dynamics for multiple steps."""
        trajectory = {
            'energy': [],
            'field_strength': [],
            'n_active': [],
            'max_activation': []
        }
        
        for step in range(n_steps):
            info = self.step()
            
            for key in trajectory:
                trajectory[key].append(info[key])
            
            if verbose and step % 10 == 0:
                print(f"Step {step:3d}: energy={info['energy']:.3f}, "
                      f"n_active={info['n_active']}, "
                      f"max_act={info['max_activation']:.3f}")
        
        return trajectory
    
    def reset(self):
        """Reset dynamics."""
        self.substrate.reset_activations()
        self.step_count = 0
        self.activation_history = []
        self.energy_history = []


def bootstrap_validated_primitives(substrate: LongRangeSubstrate,
                                   n_primitives: int = 20,
                                   seed: int = 42) -> List[GeometricPattern]:
    """Create a set of validated primitive patterns.
    
    These act as the stable substrate over which fast resonance operates.
    In a real system, these would be formed through slow learning with
    sensory validation. Here we hand-craft them.
    
    Args:
        substrate: the substrate
        n_primitives: number of primitives to create
        seed: random seed
        
    Returns:
        list of validated geometric patterns
    """
    np.random.seed(seed)
    primitives = []
    
    # Create diverse geometric shapes scattered in space
    shapes = ['triangle', 'square', 'circle', 'line']
    
    for i in range(n_primitives):
        # Random position
        center = np.random.rand(substrate.space_dim)
        
        # Random shape
        shape = shapes[i % len(shapes)]
        
        # Random size
        size = 0.03 + 0.05 * np.random.rand()
        
        # Create pattern
        from .geometric_pattern import create_geometric_shape
        units = create_geometric_shape(substrate, shape, center, size)
        
        if len(units) > 0:
            # Activate to create pattern
            temp_activations = substrate.activations.copy()
            substrate.activations[units] = 0.5 + 0.3 * np.random.rand()
            
            # Extract as pattern
            pattern = GeometricPattern.from_substrate(substrate, threshold=0.1)
            
            if pattern.signature['n_active'] > 0:
                primitives.append(pattern)
            
            # Reset
            substrate.activations = temp_activations
    
    print(f"Created {len(primitives)} validated primitive patterns")
    return primitives
