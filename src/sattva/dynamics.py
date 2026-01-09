"""SATTVA dynamics: settling with long-range coupling and geometric resonance."""

import numpy as np
from typing import List, Optional
from .long_range_substrate import LongRangeSubstrate
from .geometric_pattern import GeometricPattern


class SATTVADynamics:
    """Settling dynamics with long-range field coupling and geometric resonance.
    
    Dynamics equation:
    du/dt = -dV/du + alpha*field + beta*geometric + noise
    
    where:
    - dV/du: local attractor forces (energy gradient)
    - field: long-range field influence (10-100x range)
    - geometric: geometric pattern resonance (distant similar shapes couple)
    - noise: exploration/thermal fluctuations
    """
    
    def __init__(self, 
                 substrate: LongRangeSubstrate,
                 stored_patterns: Optional[List[GeometricPattern]] = None,
                 alpha: float = 0.5,  # field coupling strength
                 beta: float = 0.3,   # geometric coupling strength
                 gamma: float = 0.2,  # local attractor strength
                 noise_level: float = 0.01):
        """
        Args:
            substrate: the long-range substrate
            stored_patterns: learned geometric patterns (for memory)
            alpha: strength of long-range field coupling
            beta: strength of geometric resonance
            gamma: strength of local attractor forces
            noise_level: exploration noise amplitude
        """
        self.substrate = substrate
        self.stored_patterns = stored_patterns or []
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_level = noise_level
        
        # History for analysis
        self.activation_history = []
        self.energy_history = []
    
    def local_energy(self, activations: np.ndarray) -> float:
        """Compute local energy function.
        
        Simple quadratic energy for now: E = 0.5 * sum(u^2)
        This creates attractors at u=0, which we'll counteract with field/geometric forces.
        
        Args:
            activations: current activation levels
            
        Returns:
            total energy
        """
        return 0.5 * np.sum(activations ** 2)
    
    def local_gradient(self) -> np.ndarray:
        """Compute gradient of local energy (attractor forces).
        
        For E = 0.5 * sum(u^2), gradient is u.
        
        Returns:
            energy gradient for each unit
        """
        return self.substrate.activations
    
    def geometric_coupling_force(self) -> np.ndarray:
        """Compute forces from geometric pattern resonance.
        
        If current activation geometrically matches a stored pattern,
        pull toward that pattern.
        
        Returns:
            coupling force for each unit
        """
        if len(self.stored_patterns) == 0:
            return np.zeros(self.substrate.n_units)
        
        # Extract current pattern
        current = GeometricPattern.from_substrate(self.substrate, threshold=0.05)
        
        if current.signature['n_active'] == 0:
            return np.zeros(self.substrate.n_units)
        
        force = np.zeros(self.substrate.n_units)
        
        for pattern in self.stored_patterns:
            # Compute geometric similarity
            resonance = current.resonance_strength(pattern)
            
            if resonance > 0.3:  # significant geometric match
                # Pull active units in pattern toward higher activation
                for i, unit_idx in enumerate(pattern.active_units):
                    target_activation = pattern.activations[i]
                    current_activation = self.substrate.activations[unit_idx]
                    
                    # Force proportional to resonance and distance to target
                    force[unit_idx] += resonance * (target_activation - current_activation)
        
        return force
    
    def step(self, dt: float = 0.1) -> dict:
        """Take one dynamics step.
        
        Args:
            dt: time step size
            
        Returns:
            dictionary with step info (energy, field strength, etc.)
        """
        # Compute forces
        local_force = -self.gamma * self.local_gradient()
        field = self.alpha * self.substrate.compute_field()
        geometric_force = self.beta * self.geometric_coupling_force()
        noise = self.noise_level * np.random.randn(self.substrate.n_units)
        
        # Total force
        total_force = local_force + field + geometric_force + noise
        
        # Update activations
        self.substrate.activations += dt * total_force
        
        # Keep in valid range [0, 1]
        self.substrate.activations = np.clip(self.substrate.activations, 0, 1)
        
        # Compute metrics
        energy = self.local_energy(self.substrate.activations)
        field_strength = np.mean(np.abs(field))
        geometric_strength = np.mean(np.abs(geometric_force))
        n_active = len(self.substrate.get_active_pattern(threshold=0.1))
        
        # Store history
        self.activation_history.append(self.substrate.activations.copy())
        self.energy_history.append(energy)
        
        return {
            'energy': energy,
            'field_strength': field_strength,
            'geometric_strength': geometric_strength,
            'n_active': n_active,
            'max_activation': np.max(self.substrate.activations)
        }
    
    def run(self, n_steps: int, verbose: bool = False) -> dict:
        """Run dynamics for multiple steps.
        
        Args:
            n_steps: number of time steps
            verbose: print progress
            
        Returns:
            dictionary with trajectory info
        """
        trajectory = {
            'energy': [],
            'field_strength': [],
            'geometric_strength': [],
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
    
    def store_current_pattern(self, name: str = None) -> GeometricPattern:
        """Store current activation pattern as a memory.
        
        Args:
            name: optional name for pattern
            
        Returns:
            the stored pattern
        """
        pattern = GeometricPattern.from_substrate(self.substrate, threshold=0.1)
        self.stored_patterns.append(pattern)
        
        if name:
            pattern.name = name
        
        return pattern
    
    def reset(self):
        """Reset dynamics and clear history."""
        self.substrate.reset_activations()
        self.activation_history = []
        self.energy_history = []
