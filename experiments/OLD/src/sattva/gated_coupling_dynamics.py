"""Activation-gated coupling dynamics with depth annealing.

Key insights:
1. Coupling strength gated by activation mass (critical threshold)
2. Sparse patterns have limited reach (context-dependent)
3. Rich patterns have wide reach (foundational)
4. Depth anneals over time if pattern unused (allows correction of bad primitives)
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .long_range_substrate import LongRangeSubstrate
from .geometric_pattern import GeometricPattern


@dataclass
class PrimitivePattern:
    """A primitive pattern with depth, usage tracking, and annealing."""
    pattern: GeometricPattern
    depth: float  # 0=surface, 1=deep primitive
    formation_time: int = 0
    last_activation_time: int = 0
    activation_count: int = 0
    
    # Annealing parameters
    depth_decay_rate: float = 0.999  # slow decay per step when unused
    unused_threshold: int = 1000  # steps before considering unused
    
    def update_usage(self, current_time: int, was_active: bool):
        """Update usage statistics and apply depth annealing."""
        if was_active:
            self.last_activation_time = current_time
            self.activation_count += 1
        else:
            # Depth annealing: reduce influence of unused patterns
            steps_since_use = current_time - self.last_activation_time
            if steps_since_use > self.unused_threshold:
                self.depth *= self.depth_decay_rate
                # Floor at 0.1 (don't completely eliminate)
                self.depth = max(0.1, self.depth)


class GatedCouplingDynamics:
    """Dynamics with activation-gated coupling and depth annealing.
    
    Core principles:
    - Long-range coupling requires critical mass of activation
    - Coupling strength = f(depth, activation_mass)
    - Unused primitives gradually lose influence (depth annealing)
    - Damping dominates excitation for stability
    """
    
    def __init__(self,
                 substrate: LongRangeSubstrate,
                 primitive_patterns: List[PrimitivePattern],
                 # Gating parameters
                 activation_threshold: int = 5,  # min units for long-range
                 activation_scale: float = 10.0,  # sigmoid scale
                 # Coupling parameters (conservative for stability)
                 alpha: float = 0.15,  # base field coupling (reduced)
                 beta: float = 0.2,    # base geometric coupling (reduced)
                 gamma: float = 0.6,   # strong damping (dominant)
                 # Other
                 fast_dt: float = 0.02,
                 noise_level: float = 0.01):
        """
        Args:
            substrate: the substrate
            primitive_patterns: list of PrimitivePattern objects
            activation_threshold: min active units for long-range coupling
            activation_scale: sigmoid scale for mass gating
            alpha: base field coupling (will be gated)
            beta: base geometric coupling (will be gated)
            gamma: damping (should be > alpha + beta for stability)
            fast_dt: timestep
            noise_level: exploration noise
        """
        self.substrate = substrate
        self.primitive_patterns = primitive_patterns
        
        self.activation_threshold = activation_threshold
        self.activation_scale = activation_scale
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Check stability condition
        if alpha + beta >= gamma:
            print(f"WARNING: Excitation ({alpha + beta:.2f}) >= Damping ({gamma:.2f})")
            print(f"         System may be unstable. Recommend gamma > {alpha + beta:.2f}")
        
        self.fast_dt = fast_dt
        self.noise_level = noise_level
        
        # Tracking
        self.step_count = 0
        self.activation_history = []
        self.energy_history = []
        self.gating_history = []  # track gating factor
    
    def compute_activation_mass_gating(self, n_active: int) -> float:
        """Compute gating factor based on activation mass.
        
        Returns value in [0, 1] where:
        - Below threshold: near 0 (no long-range coupling)
        - Above threshold: approaches 1 (full coupling)
        
        Args:
            n_active: number of currently active units
            
        Returns:
            gating factor in [0, 1]
        """
        # Sigmoid gating
        x = (n_active - self.activation_threshold) / self.activation_scale
        return 1.0 / (1.0 + np.exp(-x))
    
    def compute_gated_field(self, mass_gating: float) -> np.ndarray:
        """Compute long-range field with activation mass gating.
        
        Args:
            mass_gating: gating factor from activation mass
            
        Returns:
            field array
        """
        # Get active units
        active_mask = self.substrate.activations > 0.05
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            return np.zeros(self.substrate.n_units)
        
        # For each active unit, compute its depth (from primitive membership)
        depths = self.substrate.depth.copy()
        
        # Update depths from primitive patterns
        for prim in self.primitive_patterns:
            for unit_idx in prim.pattern.active_units:
                if unit_idx < len(depths):
                    depths[unit_idx] = max(depths[unit_idx], prim.depth)
        
        # Compute field with gated range
        field = np.zeros(self.substrate.n_units)
        distances = self.substrate.compute_distance_matrix()
        
        for i in active_indices:
            # Depth-dependent range
            R_i = self.substrate.R_surface + depths[i] * (self.substrate.R_deep - self.substrate.R_surface)
            
            # Gate the range by activation mass
            effective_R = self.substrate.R_surface + mass_gating * (R_i - self.substrate.R_surface)
            
            # Power-law kernel with gated range
            kernel = 1.0 / (1.0 + (distances[i, :] / effective_R) ** self.substrate.alpha)
            kernel[i] = 0.0
            
            # Contribution to field (also gated by mass)
            field += mass_gating * self.substrate.activations[i] * kernel
        
        return field
    
    def compute_geometric_coupling(self, mass_gating: float) -> np.ndarray:
        """Compute geometric coupling to primitives (gated by mass)."""
        current = GeometricPattern.from_substrate(self.substrate, threshold=0.05)
        
        if current.signature['n_active'] == 0:
            return np.zeros(self.substrate.n_units)
        
        force = np.zeros(self.substrate.n_units)
        
        for prim in self.primitive_patterns:
            resonance = current.resonance_strength(prim.pattern)
            
            # Gate by both resonance strength AND activation mass
            if resonance > 0.2:
                gated_strength = mass_gating * resonance
                
                for i, unit_idx in enumerate(prim.pattern.active_units):
                    if unit_idx < len(force):
                        target = prim.pattern.activations[i]
                        current_act = self.substrate.activations[unit_idx]
                        force[unit_idx] += gated_strength * (target - current_act)
        
        return force
    
    def update_primitive_usage(self):
        """Update usage statistics and apply depth annealing."""
        # Check which primitives are currently active
        for prim in self.primitive_patterns:
            # Is this primitive currently activated?
            if len(prim.pattern.active_units) > 0:
                overlap = np.mean([
                    self.substrate.activations[i] 
                    for i in prim.pattern.active_units 
                    if i < len(self.substrate.activations)
                ])
                was_active = overlap > 0.3
            else:
                was_active = False
            
            prim.update_usage(self.step_count, was_active)
    
    def step(self) -> Dict:
        """Take one dynamics step."""
        # 1. Compute activation mass and gating
        n_active = len(self.substrate.get_active_pattern(0.1))
        mass_gating = self.compute_activation_mass_gating(n_active)
        
        # 2. Compute forces with gating
        field = self.alpha * self.compute_gated_field(mass_gating)
        geometric = self.beta * self.compute_geometric_coupling(mass_gating)
        damping = -self.gamma * self.substrate.activations
        noise = self.noise_level * np.random.randn(self.substrate.n_units)
        
        # 3. Update
        total_force = field + geometric + damping + noise
        self.substrate.activations += self.fast_dt * total_force
        self.substrate.activations = np.clip(self.substrate.activations, 0, 1)
        
        # 4. Update primitive usage and depth annealing
        self.update_primitive_usage()
        
        # 5. Metrics
        energy = 0.5 * np.sum(self.substrate.activations ** 2)
        
        self.step_count += 1
        self.activation_history.append(self.substrate.activations.copy())
        self.energy_history.append(energy)
        self.gating_history.append(mass_gating)
        
        return {
            'energy': energy,
            'n_active': n_active,
            'mass_gating': mass_gating,
            'max_activation': np.max(self.substrate.activations),
            'mean_primitive_depth': np.mean([p.depth for p in self.primitive_patterns])
        }
    
    def run(self, n_steps: int, verbose: bool = False) -> Dict:
        """Run dynamics."""
        trajectory = {
            'energy': [],
            'n_active': [],
            'mass_gating': [],
            'max_activation': [],
            'mean_primitive_depth': []
        }
        
        for step in range(n_steps):
            info = self.step()
            
            for key in trajectory:
                trajectory[key].append(info[key])
            
            if verbose and step % 10 == 0:
                print(f"Step {step:3d}: energy={info['energy']:.3f}, "
                      f"n_active={info['n_active']}, "
                      f"gating={info['mass_gating']:.3f}, "
                      f"max_act={info['max_activation']:.3f}")
        
        return trajectory
    
    def reset(self):
        """Reset dynamics."""
        self.substrate.reset_activations()
        self.step_count = 0
        self.activation_history = []
        self.energy_history = []
        self.gating_history = []


def bootstrap_gated_primitives(substrate: LongRangeSubstrate,
                              n_primitives: int = 20,
                              initial_depth: float = 0.8,
                              seed: int = 42) -> List[PrimitivePattern]:
    """Bootstrap primitive patterns with depth tracking.
    
    Args:
        substrate: the substrate
        n_primitives: number to create
        initial_depth: starting depth value (0.8 = fairly primitive)
        seed: random seed
        
    Returns:
        list of PrimitivePattern objects
    """
    np.random.seed(seed)
    primitives = []
    
    shapes = ['triangle', 'square', 'circle', 'line']
    
    for i in range(n_primitives):
        center = np.random.rand(substrate.space_dim)
        shape = shapes[i % len(shapes)]
        size = 0.03 + 0.05 * np.random.rand()
        
        from .geometric_pattern import create_geometric_shape
        units = create_geometric_shape(substrate, shape, center, size)
        
        if len(units) > 0:
            temp = substrate.activations.copy()
            substrate.activations[units] = 0.5 + 0.3 * np.random.rand()
            
            pattern = GeometricPattern.from_substrate(substrate, threshold=0.1)
            
            if pattern.signature['n_active'] > 0:
                prim = PrimitivePattern(
                    pattern=pattern,
                    depth=initial_depth,
                    formation_time=0
                )
                primitives.append(prim)
            
            substrate.activations = temp
    
    print(f"Bootstrapped {len(primitives)} gated primitive patterns")
    return primitives
