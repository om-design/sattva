"""SATTVA dynamics: settling with long-range coupling and geometric resonance."""

import numpy as np
from typing import List, Optional
from .long_range_substrate import LongRangeSubstrate
from .geometric_pattern import GeometricPattern


class SATTVADynamics:
    """Settling dynamics with long-range field coupling and geometric resonance.
    
    Biologically-realistic dynamics equation:
    du/dt = -gamma*(u - u_rest) + alpha*field + beta*geometric + noise
    
    where:
    - gamma*(u - u_rest): restoration toward resting potential (STRONG, local)
    - field: long-range field influence (WEAK but 10-100x range)
    - geometric: geometric pattern resonance (WEAK, selective)
    - noise: exploration/thermal fluctuations
    
    Key biological principles:
    1. Neurons have resting potential (~u_rest), don't decay to zero
    2. Chemical dynamics (gamma) >> field effects (alpha) by ~50-100x
    3. Field is weak per-unit but has massive range (10-100x typical connections)
    4. Only strong geometric matches can influence settling through weak field
    5. This scale separation enables regulation without ad-hoc inhibition
    """
    
    def __init__(self, 
                 substrate: LongRangeSubstrate,
                 stored_patterns: Optional[List[GeometricPattern]] = None,
                 alpha: float = 0.02,  # field coupling strength (WEAK - biological)
                 beta: float = 0.05,   # geometric coupling strength (WEAK)
                 gamma: float = 1.5,   # local restoration strength (STRONG - biological)
                 u_rest: float = 0.1,  # resting potential (biological baseline)
                 activation_threshold: float = 0.25,  # threshold for field contribution
                 local_weight: float = 0.3,  # direct connection weight
                 noise_level: float = 0.01,
                 hebbian_learning_rate: float = 0.01):
        """
        Args:
            substrate: the long-range substrate
            stored_patterns: learned geometric patterns (for memory)
            alpha: strength of long-range field coupling (WEAK - orders of magnitude less than gamma)
            beta: strength of geometric resonance (WEAK - selective amplification only)
            gamma: strength of local restoration forces (STRONG - dominates settling)
            u_rest: resting potential - baseline activation level (biological: neurons don't go to zero)
            activation_threshold: units must exceed this to contribute to field (biological firing threshold)
            local_weight: strength of direct synaptic connections (geometric structure)
            noise_level: exploration noise amplitude
            
        Expert Intuition Architecture:
        - Direct connections form geometric structure of concepts
        - Only ACTIVE units (>threshold) contribute to long-range field
        - Multiple similar patterns create "rhyming resonance"
        - Strong cumulative resonance recruits distant patterns
        - Fast, pattern-based recognition emerges naturally
        """
        self.substrate = substrate
        self.stored_patterns = stored_patterns or []
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.u_rest = u_rest
        self.activation_threshold = activation_threshold
        self.local_weight = local_weight
        self.noise_level = noise_level
        
        # Hebbian learning
        self.hebbian_learning_rate = hebbian_learning_rate
        self.learning_enabled = False  # Enable explicitly for learning phases

        # Myelination: off by default (slow, long-term infrastructure effect)
        self.myelination_enabled = False
        
        # History for analysis
        self.activation_history = []
        self.energy_history = []
        self.field_history = []
        self.resonance_history = []
        self.infrastructure_history = []
        
        # Step counter for infrastructure management
        self.step_count = 0
    
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
    
    def restoration_force(self) -> np.ndarray:
        """Compute biological restoration force toward resting potential.
        
        Biological principle: neurons restore toward resting potential, not zero.
        This creates stable non-zero attractor states.
        
        Returns:
            restoration force for each unit: -gamma * (u - u_rest)
        """
        return -self.gamma * (self.substrate.activations - self.u_rest)
    
    def compute_rhyming_resonance(self) -> tuple[np.ndarray, float]:
        """Compute "rhyming" resonance from stored primitives.
        
        Expert intuition: current pattern doesn't need exact match.
        Partial similarity ("rhyming") with multiple primitives combines.
        Strong cumulative resonance enables long-range recruitment.
        
        Returns:
            (force, total_resonance_strength)
        """
        if len(self.stored_patterns) == 0:
            return np.zeros(self.substrate.n_units), 0.0
        
        # Extract current pattern (only active units)
        current = GeometricPattern.from_substrate(
            self.substrate, 
            threshold=self.activation_threshold
        )
        
        if current.signature['n_active'] == 0:
            return np.zeros(self.substrate.n_units), 0.0
        
        force = np.zeros(self.substrate.n_units)
        total_resonance = 0.0
        
        for pattern in self.stored_patterns:
            # Compute geometric similarity ("rhyming strength")
            similarity = current.similarity(pattern)
            
            # LOWER threshold than exact match - partial similarity counts!
            if similarity > 0.2:  # "rhymes" with this pattern
                # Compute resonance strength
                resonance = similarity * np.sqrt(
                    np.mean(current.activations) * np.mean(pattern.activations)
                )
                total_resonance += resonance
                
                # Selective amplification of matching units
                for i, unit_idx in enumerate(pattern.active_units):
                    if unit_idx < len(self.substrate.activations):
                        target = pattern.activations[i]
                        current_act = self.substrate.activations[unit_idx]
                        force[unit_idx] += resonance * (target - current_act)
        
        return force, total_resonance
    
    def step(self, dt: float = 0.1) -> dict:
        """Take one dynamics step with expert intuition architecture.
        
        Combines:
        1. Direct connections (local geometric structure)
        2. Threshold field (only active units contribute)
        3. Rhyming resonance (partial similarity aggregates)
        4. Chemical restoration (stability)
        
        Args:
            dt: time step size
            
        Returns:
            dictionary with step info
        """
        # 1. DIRECT CONNECTIONS (geometric structure)
        local_input = self.local_weight * self.substrate.compute_local_input()
        
        # 2. RESTORATION (strong, toward resting potential)
        restoration = self.restoration_force()
        
        # 3. LONG-RANGE FIELD (only from ACTIVE units)
        field_raw = self.substrate.compute_field(
            activation_threshold=self.activation_threshold
        )
        field = self.alpha * field_raw
        
        # 4. RHYMING RESONANCE (cumulative from primitives)
        rhyming_force, total_resonance = self.compute_rhyming_resonance()
        geometric_force = self.beta * rhyming_force
        
        # 5. NOISE (exploration)
        noise = self.noise_level * np.random.randn(self.substrate.n_units)
        
        # TOTAL: restoration dominates, field/geometric provide bias
        total_force = restoration + local_input + field + geometric_force + noise
        
        # Update
        self.substrate.activations += dt * total_force
        
        # Biological bounds
        self.substrate.activations = np.clip(self.substrate.activations, 0, 1)
        
        # Metrics
        energy = self.local_energy(self.substrate.activations)
        field_strength = np.mean(np.abs(field_raw))
        n_active = len(self.substrate.get_active_pattern(threshold=self.activation_threshold))
        
        # Store history
        self.activation_history.append(self.substrate.activations.copy())
        self.energy_history.append(energy)
        self.field_history.append(field_strength)
        self.resonance_history.append(total_resonance)
        
        # Hebbian learning (if enabled)
        if self.learning_enabled:
            self.hebbian_update(dt)

        # Optional slow myelination (off by default here)
        if self.myelination_enabled:
            # Connection-level myelination (accretion)
            self.substrate.accretion_dynamics(dt, activation_threshold=self.activation_threshold)
            
            # Infrastructure management (every 100 steps - slow process)
            self.step_count += 1
            if self.step_count % 100 == 0:
                infra_info = self.substrate.manage_infrastructure(dt * 100)
                self.infrastructure_history.append(infra_info)
        
        return {
            'energy': energy,
            'field_strength': field_strength,
            'resonance_strength': total_resonance,
            'n_active': n_active,
            'max_activation': np.max(self.substrate.activations),
            'mean_activation': np.mean(self.substrate.activations)
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
        
        Uses activation threshold to store only truly active units,
        not units at resting potential.
        
        Args:
            name: optional name for pattern
            
        Returns:
            the stored pattern
        """
        # Store only units above activation threshold (truly firing)
        pattern = GeometricPattern.from_substrate(
            self.substrate, 
            threshold=self.activation_threshold
        )
        self.stored_patterns.append(pattern)
        
        if name:
            pattern.name = name
        
        return pattern
    
    def learn_from_experience(self, n_repetitions: int = 5, consolidation_steps: int = 30) -> GeometricPattern:
        """Learn a new primitive through repeated exposure (Phase 2: Learning).
        
        Expert learning: repeated experiences consolidate into stable primitives.
        Similar to Hebbian learning: "neurons that fire together, wire together".
        
        Args:
            n_repetitions: number of exposure cycles
            consolidation_steps: steps per cycle for settling
            
        Returns:
            learned primitive pattern
        """
        # Store initial activation as seed
        seed_activation = self.substrate.activations.copy()
        
        # Repeated exposure with consolidation
        for rep in range(n_repetitions):
            # Restore seed (simulates repeated experience)
            self.substrate.activations = seed_activation.copy()
            
            # Let settle with REDUCED dynamics (both field AND restoration)
            old_alpha, old_gamma = self.alpha, self.gamma
            self.alpha *= 0.3   # Weaker field during learning
            self.gamma *= 0.4   # WEAKER restoration (key fix: don't pull to u_rest too strongly)
            
            for _ in range(consolidation_steps):
                self.step(dt=0.1)
            
            self.alpha, self.gamma = old_alpha, old_gamma
            
            # Update seed with consolidated pattern (learning effect)
            mask = self.substrate.activations > self.activation_threshold
            seed_activation[mask] = 0.7 * seed_activation[mask] + 0.3 * self.substrate.activations[mask]
        
        # Final consolidation with reduced restoration
        self.substrate.activations = seed_activation
        old_gamma = self.gamma
        self.gamma *= 0.5  # Keep pattern above threshold
        
        for _ in range(consolidation_steps):
            self.step(dt=0.1)
        
        self.gamma = old_gamma
        
        # Check if similar to existing primitives (refinement vs new)
        current = GeometricPattern.from_substrate(
            self.substrate,
            threshold=self.activation_threshold
        )
        
        best_match = None
        best_match_idx = -1
        best_similarity = 0.0
        
        for idx, pattern in enumerate(self.stored_patterns):
            similarity = current.similarity(pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern
                best_match_idx = idx
        
        # If very similar to existing, refine it (don't store duplicate)
        if best_similarity > 0.7 and best_match is not None:
            # Refinement: blend current with existing (weighted average)
            # Find overlapping units
            current_set = set(current.active_units)
            match_set = set(best_match.active_units)
            overlap = current_set & match_set
            
            if len(overlap) > 0:
                # Update overlapping units
                for unit in overlap:
                    # Find indices in both patterns
                    curr_idx = list(current.active_units).index(unit)
                    match_idx = list(best_match.active_units).index(unit)
                    # Blend activations
                    best_match.activations[match_idx] = (
                        0.8 * best_match.activations[match_idx] + 
                        0.2 * current.activations[curr_idx]
                    )
            
            return best_match
        else:
            # New primitive: store it
            return self.store_current_pattern()
    
    def assess_primitive_quality(self, pattern: GeometricPattern) -> dict:
        """Assess quality of a primitive for vetting.
        
        Expert learning: not all experiences become primitives.
        Quality filtering ensures library remains useful.
        
        Args:
            pattern: pattern to assess
            
        Returns:
            quality metrics dict
        """
        # CRITICAL: Empty patterns have zero quality
        n_active = pattern.signature.get('n_active', 0)
        if n_active == 0:
            return {
                'quality': 0.0,
                'compactness': 0.0,
                'distinctiveness': 0.0,
                'stability': 0.0,
                'size': 0.0,
                'vetted': False
            }
        
        # Metric 1: Compactness (good primitives are coherent)
        spread = pattern.signature.get('spread', 1.0)
        compactness_score = 1.0 / (1.0 + spread)
        
        # Metric 2: Distinctiveness (not too similar to existing)
        max_similarity = 0.0
        for existing in self.stored_patterns:
            if existing is not pattern:  # Use identity comparison, not array equality
                sim = pattern.similarity(existing)
                max_similarity = max(max_similarity, sim)
        distinctiveness_score = 1.0 - max_similarity
        
        # Metric 3: Stability (activations not too weak)
        stability_score = min(1.0, n_active / 50.0)  # Prefer 50+ units
        
        # Metric 4: Not too large (focused patterns better than diffuse)
        size_score = 1.0 if n_active < 200 else 0.5
        
        # Overall quality
        quality = (compactness_score * 0.3 + 
                  distinctiveness_score * 0.3 + 
                  stability_score * 0.2 + 
                  size_score * 0.2)
        
        return {
            'quality': quality,
            'compactness': compactness_score,
            'distinctiveness': distinctiveness_score,
            'stability': stability_score,
            'size': size_score,
            'vetted': quality > 0.5
        }
    
    def prune_poor_primitives(self, quality_threshold: float = 0.4) -> int:
        """Remove low-quality primitives from library (Phase 2: Curation).
        
        Expert libraries need maintenance. Poor primitives dilute
        rhyming resonance and slow recognition.
        
        Args:
            quality_threshold: minimum quality to keep
            
        Returns:
            number of primitives removed
        """
        initial_count = len(self.stored_patterns)
        
        # Assess all primitives
        keep = []
        for pattern in self.stored_patterns:
            metrics = self.assess_primitive_quality(pattern)
            if metrics['quality'] >= quality_threshold:
                keep.append(pattern)
        
        self.stored_patterns = keep
        removed = initial_count - len(self.stored_patterns)
        
        return removed
    
    def get_learning_rate(self) -> float:
        """Compute current learning rate based on library size.
        
        Expert advantage: rich libraries enable faster learning.
        Network effects create accelerating returns.
        
        Returns:
            learning rate multiplier (>1.0 means faster than novice)
        """
        n_primitives = len(self.stored_patterns)
        
        # Novice (0-10 primitives): slow learning (1.0x)
        # Intermediate (10-50): moderate (1.5x)
        # Expert (50+): fast (2.0x+)
        
        if n_primitives < 10:
            return 1.0
        elif n_primitives < 50:
            return 1.0 + 0.5 * (n_primitives - 10) / 40
        else:
            return 2.0
    
    def hebbian_update(self, dt: float = 0.1):
        """Hebbian learning: "neurons that fire together, wire together".
        
        Conductance-gated plasticity: Myelinated connections resist change.
        This protects foundational knowledge while allowing specialized learning.
        
        Biological principle:
        - High conductance (general concepts) → Low plasticity (protected)
        - Low conductance (specialized) → High plasticity (learns easily)
        
        Args:
            dt: timestep
        """
        from scipy.sparse import lil_matrix
        
        # Find active units
        active_indices = np.where(self.substrate.activations > self.activation_threshold)[0]
        
        if len(active_indices) < 2:
            return  # Need pairs for Hebbian
        
        # Get plasticity matrix (inverse of conductance)
        plasticity = self.substrate.get_plasticity_matrix()
        
        # Convert connections to lil for modification
        if not isinstance(self.substrate.connections, lil_matrix):
            self.substrate.connections = self.substrate.connections.tolil()
        
        # Hebbian update for co-active pairs
        for i in active_indices:
            for j in active_indices:
                if i != j and self.substrate.connections[i, j] != 0:
                    # Hebbian signal
                    hebbian = self.substrate.activations[i] * self.substrate.activations[j]
                    
                    # Plasticity gates learning
                    # High-conductance connections barely change
                    plasticity_ij = float(plasticity[i, j])
                    
                    # Update weight
                    delta_w = self.hebbian_learning_rate * plasticity_ij * hebbian * dt
                    current = float(self.substrate.connections[i, j])
                    self.substrate.connections[i, j] = np.clip(
                        current + delta_w,
                        0.0,  # Non-negative
                        2.0   # Maximum strength
                    )
        
        # Convert back to CSR
        self.substrate.connections = self.substrate.connections.tocsr()
    
    def enable_learning(self, enabled: bool = True):
        """Enable or disable Hebbian learning.
        
        Args:
            enabled: whether to enable learning
        """
        self.learning_enabled = enabled
    
    def get_myelinated_primitives(self, conductance_threshold: float = 0.5):
        """Extract geometric primitives from myelinated connection topology.
        
        Primitives EMERGE from usage patterns!
        This is the biological grounding of learned knowledge.
        
        Args:
            conductance_threshold: minimum conductance to include
            
        Returns:
            list of primitives (connected components of thick connections)
        """
        return self.substrate.extract_myelinated_primitives(conductance_threshold)
    
    def reset(self):
        """Reset dynamics and clear history."""
        self.substrate.reset_activations()
        self.activation_history = []
        self.energy_history = []
        self.field_history = []
        self.resonance_history = []
        self.infrastructure_history = []
        self.step_count = 0
