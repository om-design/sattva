"""SATTVA: GA-first core.

Patterns are GA multivectors over units; intuition is GA resonance.
Legacy modules have been archived under 'old'.
"""

from .ga_sattva_core import (
    GAUnitSet,
    GAPattern,
    GASATTVADynamics,
    create_ga_primitive,
    pattern_from_units,
)

__all__ = [
    "GAUnitSet",
    "GAPattern",
    "GASATTVADynamics",
    "create_ga_primitive",
    "pattern_from_units",
]
