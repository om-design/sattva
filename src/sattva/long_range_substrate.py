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
    
    def compute_distance_matrix(self):
        """Precompute pairwise distances (call once, cache)."""
        if self._distance_matrix is None:
            # Efficient pairwise distance computation
            diff = self.positions[:, None, :] - self.positions[None, :, :]
            self._distance_matrix = np.sqrt(np.sum(diff**2, axis=2))
        return self._distance_matrix
    
    def compute_field(self, threshold: float = 0.01) -> np.ndarray:
        """Compute long-range field influence at each position.
        
        Uses power-law kernel: K(r,d) = A/(1 + (r/R(d))^alpha)
        Only active units (activation > threshold) contribute.
        
        Args:
            threshold: minimum activation to contribute to field
            
        Returns:
            field: (n_units,) array of field strength at each position
        """
        field = np.zeros(self.n_units)
        
        # Find active units
        active_mask = self.activations > threshold
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
