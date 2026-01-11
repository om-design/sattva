"""Long-range coupling substrate for SATTVA.

This implements the core architectural commitment: resonance coupling
operates at 10-100x the range of typical neural network connections.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LongRangeSubstrate:
    """Field-based substrate with power-law long-range coupling.
    
    Key features:
    - Units have positions in abstract geometric space
    - Depth parameter determines coupling range (primitive vs. complex)
    - Power-law kernels ensure 10-100x coupling range
    - Field computation captures resonance effects
    """
    
    n_units: int
    space_dim: int = 3
    R_surface: float = 5.0   # surface pattern coupling range
    R_deep: float = 50.0     # deep pattern coupling range (10x)
    alpha: float = 1.5       # power-law exponent
    
    def __post_init__(self):
        # Initialize positions randomly in unit cube
        self.positions = np.random.rand(self.n_units, self.space_dim)
        
        # Initialize depth (0=surface, 1=deep) - start mostly surface
        self.depth = np.random.beta(2, 5, self.n_units)  # skewed toward surface
        
        # Activation levels
        self.activations = np.zeros(self.n_units)
        
        # Myelination parameters (must be set before building connections)
        self.BASE_CONDUCTANCE = 1.0
        self.MAX_ACCRETION = 2.0
        
        # Direct connections (local geometric structure)
        # Sparse connectivity based on spatial proximity
        # This also initializes conductance and connection_usage
        self._build_local_connections()
        
        # Energy budget for infrastructure management
        self.energy_budget = 1000.0  # Starting energy
        self.energy_income_rate = 10.0  # Metabolic generation per step
        
        # Hysteresis thresholds for myelination
        self.MYELINATION_USAGE_THRESHOLD = 0.3  # Must exceed to build
        self.DEMYELINATION_USAGE_THRESHOLD = 0.05  # Must drop below to remove
        self.MYELINATION_COST = 100.0  # One-time cost to build
        self.MAINTENANCE_COST_PER_UNIT = 0.01  # Per conductance unit per step
        self.DEMYELINATION_COST = 150.0  # HIGHER than building!
        
        # Cache for efficiency
        self._distance_matrix: Optional[np.ndarray] = None
        
    def set_positions_from_embeddings(self, embeddings: np.ndarray):
        """Set positions from semantic embeddings (e.g., SBERT).
        
        Args:
            embeddings: (n_units, embedding_dim) array
        """
        # Normalize embeddings to unit cube
        emb_min = embeddings.min(axis=0)
        emb_max = embeddings.max(axis=0)
        normalized = (embeddings - emb_min) / (emb_max - emb_min + 1e-8)
        
        # Project to desired space_dim if needed
        if normalized.shape[1] > self.space_dim:
            # Simple PCA-like projection
            from numpy.linalg import svd
            U, s, Vt = svd(normalized - normalized.mean(axis=0), full_matrices=False)
            self.positions = U[:, :self.space_dim] @ np.diag(s[:self.space_dim])
            # Renormalize
            self.positions = (self.positions - self.positions.min()) / (self.positions.max() - self.positions.min())
        else:
            self.positions = normalized[:, :self.space_dim]
    
    def _build_local_connections(self):
        """Build direct synaptic connections based on proximity.
        
        These form the geometric structure of concepts.
        Short-range, strong connections that define pattern topology.
        """
        from scipy.sparse import lil_matrix
        
        # Sparse connectivity matrix
        self.connections = lil_matrix((self.n_units, self.n_units))
        
        # Connect nearby units (local geometric structure)
        local_radius = 0.15  # Connection radius in position space
        
        for i in range(self.n_units):
            distances = np.linalg.norm(self.positions - self.positions[i], axis=1)
            nearby = (distances < local_radius) & (distances > 0)
            
            # Connection strength inversely proportional to distance
            for j in np.where(nearby)[0]:
                strength = (local_radius - distances[j]) / local_radius
                self.connections[i, j] = strength * 0.5  # Moderate strength
        
        # Convert to efficient format
        self.connections = self.connections.tocsr()
        
        # Initialize conductance tracking (same sparsity as connections)
        # Must be same shape and structure for arithmetic operations
        from scipy.sparse import csr_matrix
        self.conductance = csr_matrix((self.n_units, self.n_units))  # Start at zero
        self.connection_usage = csr_matrix((self.n_units, self.n_units))
    
    def compute_local_input(self) -> np.ndarray:
        """Compute direct synaptic input from connected neighbors.
        
        Input is weighted by conductance (myelination).
        High-conductance connections deliver more input ('wider pipe').
        
        Returns:
            local input for each unit from direct connections
        """
        # Defensive check - ensure conductance exists
        if self.conductance is None:
            from scipy.sparse import csr_matrix
            self.conductance = csr_matrix((self.n_units, self.n_units))
            self.connection_usage = csr_matrix((self.n_units, self.n_units))
        
        # Effective weight = synaptic_strength × (BASE_CONDUCTANCE + conductance)
        # Start with base conductance for all connections
        effective_weights = self.connections * self.BASE_CONDUCTANCE
        
        # Add myelination component if any exists
        if self.conductance.nnz > 0:
            # Element-wise multiply: connections × conductance
            # Only non-zero where both are non-zero
            myelination_boost = self.connections.multiply(self.conductance)
            effective_weights = effective_weights + myelination_boost
        
        return effective_weights @ self.activations
    
    def accretion_dynamics(self, dt: float = 0.1, activation_threshold: float = 0.25):
        """Connections between co-active units accrete conductance (myelination).
        
        Biological principle: Myelination forms on AXONS (connections), not cell bodies.
        Accretion builds gradually from usage. Geometric primitives EMERGE from topology.
        
        This is SLOW (like biological myelination - weeks/months).
        Only passive decay here - active demyelination is separate (expensive).
        
        Args:
            dt: timestep
            activation_threshold: units above this are considered 'active'
        """
        from scipy.sparse import lil_matrix
        
        # Find currently active units
        active_indices = np.where(self.activations > activation_threshold)[0]
        
        if len(active_indices) < 2:
            return  # Need at least 2 active units for co-activation
        
        # Convert to lil format for efficient modification
        if not isinstance(self.conductance, lil_matrix):
            self.conductance = self.conductance.tolil()
            self.connection_usage = self.connection_usage.tolil()
        
        # Accrete conductance on co-active connections
        alpha_accrete = 0.002  # VERY slow (biological timescale)
        
        for i in active_indices:
            for j in active_indices:
                if i != j and self.connections[i, j] != 0:
                    # Co-activation signal
                    co_signal = self.activations[i] * self.activations[j]
                    
                    # Accrete toward maximum (gradual buildup)
                    current = self.conductance[i, j]
                    target = self.MAX_ACCRETION * np.tanh(100 * co_signal)
                    self.conductance[i, j] = current + dt * alpha_accrete * (target - current)
                    
                    # Track usage (exponential moving average)
                    alpha_usage = 0.001
                    current_usage = float(self.connection_usage[i, j])
                    self.connection_usage[i, j] = (
                        (1 - alpha_usage) * current_usage + alpha_usage * co_signal
                    )
        
        # Very slow passive decay (natural degradation)
        # Active demyelination is separate and MORE expensive
        alpha_decay = 0.0001  # 20x slower than accretion
        self.conductance = self.conductance.multiply(1 - alpha_decay * dt)
        
        # Clip to bounds
        self.conductance.data = np.clip(self.conductance.data, 0.0, self.MAX_ACCRETION)
        
        # Convert back to CSR for efficient computation
        self.conductance = self.conductance.tocsr()
        self.connection_usage = self.connection_usage.tocsr()
    
    def manage_infrastructure(self, dt: float = 0.1, check_interval: int = 100):
        """Manage myelination infrastructure with energy-gated hysteresis.
        
        Biological economics:
        - Building myelination costs energy (oligodendrocytes, insulation)
        - Maintenance costs energy (keeping infrastructure)
        - Demyelination costs MORE energy (active removal)
        - Creates hysteresis: once built, stays unless energy crisis
        
        This implements "use it or lose it" with realistic time constants.
        Foundational concepts (high usage) stay myelinated even when use decreases.
        
        Args:
            dt: timestep
            check_interval: only check periodically (expensive operation)
        """
        from scipy.sparse import lil_matrix
        
        # Pay maintenance costs for existing myelination
        total_conductance = np.sum(self.conductance.data)
        maintenance_cost = total_conductance * self.MAINTENANCE_COST_PER_UNIT * dt
        self.energy_budget -= maintenance_cost
        
        # Energy income (metabolism)
        self.energy_budget += self.energy_income_rate * dt
        
        # Periodic infrastructure assessment (expensive)
        # In biology, this happens on slow timescales
        # For simulation, we check every N steps
        # (This should be called less frequently than step())
        
        # Convert to lil for modification
        if not isinstance(self.conductance, lil_matrix):
            self.conductance = self.conductance.tolil()
            self.connection_usage = self.connection_usage.tolil()
        
        cx = self.connections.tocoo()
        n_myelinated = 0
        n_demyelinated = 0
        
        for i, j, weight in zip(cx.row, cx.col, cx.data):
            if weight > 0:  # Connection exists
                current_conductance = float(self.conductance[i, j])
                usage = float(self.connection_usage[i, j])
                
                # STATE 1: Not myelinated, considering building
                if current_conductance < 0.1:
                    if usage > self.MYELINATION_USAGE_THRESHOLD:
                        # High usage - consider investing
                        if self.energy_budget > self.MYELINATION_COST:
                            # Build myelination (pay cost)
                            self.conductance[i, j] = self.MAX_ACCRETION * 0.5  # Start at 50%
                            self.energy_budget -= self.MYELINATION_COST
                            n_myelinated += 1
                
                # STATE 2: Myelinated, considering removal
                elif current_conductance > 0.5:
                    # Only demyelinate if usage VERY low AND energy crisis
                    if usage < self.DEMYELINATION_USAGE_THRESHOLD and self.energy_budget < -100:
                        # Energy crisis - must demyelinate
                        self.conductance[i, j] = 0.0
                        self.energy_budget -= self.DEMYELINATION_COST  # Costs more!
                        n_demyelinated += 1
        
        # Convert back to CSR
        self.conductance = self.conductance.tocsr()
        self.connection_usage = self.connection_usage.tocsr()
        
        return {
            'energy_budget': self.energy_budget,
            'myelinated': n_myelinated,
            'demyelinated': n_demyelinated,
            'total_conductance': total_conductance
        }
    
    def get_plasticity_matrix(self):
        """Get plasticity for each connection.
        
        High conductance (myelinated) = Low plasticity (protected).
        
        Returns:
            plasticity values for each connection (sparse)
        """
        from scipy.sparse import lil_matrix
        
        # Build plasticity matrix matching connections structure
        plasticity = lil_matrix((self.n_units, self.n_units))
        
        # For each connection, plasticity = 1 / (BASE + conductance)
        cx = self.connections.tocoo()
        for i, j, conn_weight in zip(cx.row, cx.col, cx.data):
            if conn_weight > 0:  # Connection exists
                cond = self.conductance[i, j] if self.conductance[i, j] != 0 else 0.0
                total_cond = self.BASE_CONDUCTANCE + cond
                plasticity[i, j] = 1.0 / total_cond
        
        return plasticity.tocsr()
    
    def extract_myelinated_primitives(self, conductance_threshold: float = 0.5):
        """Extract geometric primitives as connected components of myelinated connections.
        
        Primitives EMERGE from usage patterns!
        High-conductance subgraphs = foundational concepts.
        
        Args:
            conductance_threshold: minimum conductance to include
            
        Returns:
            list of primitive dicts with units and conductance info
        """
        import networkx as nx
        
        # Build graph of high-conductance connections
        G = nx.Graph()
        
        cx = self.conductance.tocoo()  # Convert to COO for iteration
        for i, j, c in zip(cx.row, cx.col, cx.data):
            if c > conductance_threshold:
                G.add_edge(i, j, conductance=c)
        
        # Find connected components
        primitives = []
        for component in nx.connected_components(G):
            if len(component) >= 3:  # Minimum size for primitive
                units = sorted(list(component))
                
                # Extract conductance subgraph
                total_accretion = 0
                connections_list = []
                for u in units:
                    for v in units:
                        if u < v and G.has_edge(u, v):
                            c = G[u][v]['conductance']
                            total_accretion += c
                            connections_list.append((u, v, c))
                
                primitives.append({
                    'units': units,
                    'n_units': len(units),
                    'connections': connections_list,
                    'total_accretion': total_accretion,
                    'avg_conductance': total_accretion / len(connections_list) if connections_list else 0
                })
        
        # Sort by total accretion (most myelinated first)
        primitives.sort(key=lambda p: p['total_accretion'], reverse=True)
        
        return primitives
    
    def compute_distance_matrix(self):
        """Precompute pairwise distances (call once, cache)."""
        if self._distance_matrix is None:
            # Efficient pairwise distance computation
            diff = self.positions[:, None, :] - self.positions[None, :, :]
            self._distance_matrix = np.sqrt(np.sum(diff**2, axis=2))
        return self._distance_matrix
    
    def compute_field(self, threshold: float = 0.01, activation_threshold: Optional[float] = None) -> np.ndarray:
        """Compute long-range field influence at each position.
        
        Uses power-law kernel: K(r,d) = A/(1 + (r/R(d))^alpha)
        Only ACTIVE units (above activation threshold) contribute.
        
        This is the "rhyming resonance" mechanism:
        - Only units actively firing contribute to field
        - Resting units (near u_rest) do not contribute
        - Multiple active patterns create cumulative field
        - Strong cumulative field has longer effective range
        
        Args:
            threshold: minimum activation to contribute to field (DEPRECATED, use activation_threshold)
            activation_threshold: activation level required to contribute (biological threshold)
            
        Returns:
            field: (n_units,) array of field strength at each position
        """
        field = np.zeros(self.n_units)
        
        # Use biological activation threshold if provided
        if activation_threshold is not None:
            active_threshold = activation_threshold
        else:
            active_threshold = threshold
        
        # Find ACTIVE units (above threshold, not just >0)
        active_mask = self.activations > active_threshold
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            return field
        
        # Get or compute distance matrix
        distances = self.compute_distance_matrix()
        
        # Compute depth-dependent ranges for all units
        R = self.R_surface + self.depth * (self.R_deep - self.R_surface)
        
        # Compute field contribution from each active unit
        for i in active_indices:
            # Power-law kernel
            kernel = 1.0 / (1.0 + (distances[i, :] / R[i]) ** self.alpha)
            kernel[i] = 0.0  # unit doesn't influence itself
            
            field += self.activations[i] * kernel
        
        return field
    
    def get_active_pattern(self, threshold: float = 0.1) -> np.ndarray:
        """Get indices of currently active units.
        
        Args:
            threshold: minimum activation to be considered active
            
        Returns:
            indices of active units
        """
        return np.where(self.activations > threshold)[0]
    
    def reset_activations(self):
        """Reset all activations to zero."""
        self.activations = np.zeros(self.n_units)
    
    def activate_pattern(self, unit_indices: np.ndarray, strength: float = 0.5):
        """Activate specific units with given strength.
        
        Args:
            unit_indices: which units to activate
            strength: activation level (0-1)
        """
        self.activations[unit_indices] = strength
    
    def reset_activations(self):
        """Reset all activations to resting level."""
        self.activations = np.ones(self.n_units) * 0.1  # u_rest
    
    def activate_pattern(self, unit_indices, strength: float = 0.5):
        """Activate specific units.
        
        Args:
            unit_indices: list of unit indices to activate
            strength: activation level
        """
        self.activations[unit_indices] = strength
    
    def get_active_pattern(self, threshold: float = 0.25):
        """Get currently active units above threshold.
        
        Args:
            threshold: activation threshold
            
        Returns:
            indices of active units
        """
        return np.where(self.activations > threshold)[0]
    
    def visualize_field(self, slice_dim: int = 2, slice_val: float = 0.5):
        """Visualize field in 2D slice (for debugging/visualization).
        
        Args:
            slice_dim: which dimension to slice along
            slice_val: value to slice at
        """
        if self.space_dim < 2:
            raise ValueError("Need at least 2 dimensions to visualize")
        
        # Find units near the slice
        mask = np.abs(self.positions[:, slice_dim] - slice_val) < 0.1
        
        field = self.compute_field()
        
        import matplotlib.pyplot as plt
        
        dims = [d for d in range(self.space_dim) if d != slice_dim][:2]
        
        plt.figure(figsize=(10, 8))
        
        # Plot units
        plt.scatter(
            self.positions[mask, dims[0]],
            self.positions[mask, dims[1]],
            c=self.activations[mask],
            s=100,
            cmap='Reds',
            alpha=0.6,
            edgecolors='black'
        )
        
        # Plot field strength
        plt.scatter(
            self.positions[mask, dims[0]],
            self.positions[mask, dims[1]],
            c=field[mask],
            s=30,
            cmap='Blues',
            alpha=0.3
        )
        
        plt.colorbar(label='Activation (red) / Field (blue)')
        plt.xlabel(f'Dimension {dims[0]}')
        plt.ylabel(f'Dimension {dims[1]}')
        plt.title(f'Substrate Field Visualization (slice at dim {slice_dim} = {slice_val})')
        plt.grid(True, alpha=0.3)
        
        return plt
