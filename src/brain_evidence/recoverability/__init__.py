"""Neural-recoverability geometry and target construction."""

from .geometry import fisher_center, fisher_inner, fisher_norm_sq
from .interaction import (
    AnchorLocalValidity,
    aggregate_interactions,
    anchor_local_interaction,
    factorial_interaction,
)
from .projection import NRAEvidence, NRAInteractionBatch, recoverable_target

__all__ = [
    "AnchorLocalValidity",
    "NRAEvidence",
    "NRAInteractionBatch",
    "aggregate_interactions",
    "anchor_local_interaction",
    "factorial_interaction",
    "fisher_center",
    "fisher_inner",
    "fisher_norm_sq",
    "recoverable_target",
]
