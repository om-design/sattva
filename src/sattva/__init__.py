"""SATTVA: Semantic Attractor Training of Transforming Vector Associations.

A geometric field theory approach to AI cognition where:
- Information is encoded in geometric activation patterns
- Long-range coupling (10-100x) enables distant resonance
- Creativity emerges from geometric compatibility, not randomness
"""

from .long_range_substrate import LongRangeSubstrate
from .geometric_pattern import GeometricPattern, create_geometric_shape
from .dynamics import SATTVADynamics
from .two_timescale_dynamics import TwoTimescaleDynamics, bootstrap_validated_primitives
from .gated_coupling_dynamics import GatedCouplingDynamics, bootstrap_gated_primitives, PrimitivePattern
from .semantic_space import SemanticSpace
from .attractor_core import HopfieldCore

__all__ = [
    'LongRangeSubstrate',
    'GeometricPattern',
    'create_geometric_shape',
    'SATTVADynamics',
    'TwoTimescaleDynamics',
    'bootstrap_validated_primitives',
    'GatedCouplingDynamics',
    'bootstrap_gated_primitives',
    'PrimitivePattern',
    'SemanticSpace',
    'HopfieldCore'
]
