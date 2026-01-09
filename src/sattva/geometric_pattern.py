"""Geometric pattern matching for SATTVA.

Patterns are defined by their geometric shape, not semantic content.
Similar shapes can resonate even if semantically distant.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from scipy.spatial.distance import pdist


@dataclass
class GeometricPattern:
    """A pattern defined by geometric activation shape.
    
    Key insight: patterns are similar if they form similar geometric
    configurations, regardless of semantic meaning.
    """
    
    active_units: np.ndarray  # indices of active units
    positions: np.ndarray     # (n_active, space_dim) positions
    activations: np.ndarray   # (n_active,) activation strengths
    
    def __post_init__(self):
        """Compute geometric signature upon creation."""
        self.signature = self.compute_signature()
    
    @classmethod
    def from_substrate(cls, substrate, threshold: float = 0.1):
        """Extract current active pattern from substrate.
        
        Args:
            substrate: LongRangeSubstrate instance
            threshold: minimum activation to include
            
        Returns:
            GeometricPattern instance
        """
        active_units = substrate.get_active_pattern(threshold)
        
        if len(active_units) == 0:
            # Empty pattern
            return cls(
                active_units=np.array([]),
                positions=np.zeros((0, substrate.space_dim)),
                activations=np.array([])
            )
        
        return cls(
            active_units=active_units,
            positions=substrate.positions[active_units],
            activations=substrate.activations[active_units]
        )
    
    def compute_signature(self) -> Dict[str, Any]:
        """Compute geometry-invariant features.
        
        These features should be similar for geometrically similar patterns,
        regardless of position, rotation, or scale.
        
        Returns:
            Dictionary of geometric features
        """
        if len(self.active_units) < 2:
            return {
                'centroid': np.zeros(self.positions.shape[1]) if len(self.positions.shape) > 1 else np.array([0]),
                'spread': 0.0,
                'distance_dist': np.zeros(5),
                'n_active': len(self.active_units)
            }
        
        # Centroid (for reference, but not used in similarity)
        centroid = np.average(self.positions, weights=self.activations, axis=0)
        
        # Center positions
        centered = self.positions - centroid
        
        # Second moment (spread) - weighted by activation
        spread = np.average(np.sum(centered**2, axis=1), weights=self.activations)
        
        # Pairwise distance distribution (shape descriptor)
        if len(self.active_units) > 1:
            distances = pdist(self.positions)
            hist, _ = np.histogram(distances, bins=5, range=(0, 1.5))
            distance_dist = hist / (hist.sum() + 1e-8)
        else:
            distance_dist = np.zeros(5)
        
        return {
            'centroid': centroid,
            'spread': spread,
            'distance_dist': distance_dist,
            'n_active': len(self.active_units)
        }
    
    def similarity(self, other: 'GeometricPattern') -> float:
        """Compute geometric similarity with another pattern.
        
        This is NOT semantic similarity - it's based purely on shape.
        Two patterns can be semantically distant but geometrically similar.
        
        Args:
            other: another GeometricPattern
            
        Returns:
            similarity score in [0, 1]
        """
        # Handle empty patterns
        if self.signature['n_active'] == 0 or other.signature['n_active'] == 0:
            return 0.0
        
        # Compare spread (scale-invariant if we normalize)
        spread_ratio = min(self.signature['spread'], other.signature['spread']) / \
                      (max(self.signature['spread'], other.signature['spread']) + 1e-8)
        
        # Compare distance distributions (shape similarity)
        dist_similarity = 1.0 - 0.5 * np.sum(np.abs(
            self.signature['distance_dist'] - other.signature['distance_dist']
        ))
        
        # Compare number of active units (similar complexity)
        n_ratio = min(self.signature['n_active'], other.signature['n_active']) / \
                 max(self.signature['n_active'], other.signature['n_active'])
        
        # Combine features
        similarity = 0.3 * spread_ratio + 0.5 * dist_similarity + 0.2 * n_ratio
        
        return float(np.clip(similarity, 0, 1))
    
    def resonance_strength(self, other: 'GeometricPattern') -> float:
        """How strongly should these patterns mutually excite?
        
        Combines geometric similarity with activation strength.
        
        Args:
            other: another GeometricPattern
            
        Returns:
            resonance strength
        """
        geom_sim = self.similarity(other)
        
        # Weight by activation strengths
        self_strength = np.mean(self.activations) if len(self.activations) > 0 else 0.0
        other_strength = np.mean(other.activations) if len(other.activations) > 0 else 0.0
        
        # Resonance is geometric similarity * sqrt(product of strengths)
        # sqrt because resonance depends on both, but not linearly
        resonance = geom_sim * np.sqrt(self_strength * other_strength)
        
        return float(resonance)


def create_geometric_shape(substrate, shape_type: str, center: np.ndarray, 
                          size: float = 0.1, n_points: int = 10) -> np.ndarray:
    """Create a geometric shape pattern in the substrate.
    
    Useful for testing and validation.
    
    Args:
        substrate: LongRangeSubstrate instance
        shape_type: 'triangle', 'square', 'circle', 'line'
        center: center position in space
        size: size of shape
        n_points: number of units to use
        
    Returns:
        indices of units forming the shape
    """
    # Generate shape points
    if shape_type == 'triangle':
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        points = np.array([
            [np.cos(a) * size, np.sin(a) * size, 0] for a in angles
        ])
    elif shape_type == 'square':
        points = np.array([
            [-size, -size, 0], [size, -size, 0],
            [size, size, 0], [-size, size, 0]
        ])
    elif shape_type == 'circle':
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.array([
            [np.cos(a) * size, np.sin(a) * size, 0] for a in angles
        ])
    elif shape_type == 'line':
        t = np.linspace(-1, 1, n_points)
        points = np.array([[size * ti, 0, 0] for ti in t])
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    # Add center offset
    points = points + center
    
    # Find nearest units in substrate
    indices = []
    for point in points:
        distances = np.linalg.norm(substrate.positions - point, axis=1)
        nearest = np.argmin(distances)
        if nearest not in indices:  # avoid duplicates
            indices.append(nearest)
    
    return np.array(indices)
