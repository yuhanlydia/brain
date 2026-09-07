"""Matched controls used by brain-evidence experiments."""

from .matched_negative import (
    AnchorLocalControlPair,
    AnchorLocalControlResult,
    MatchedNegativeResult,
    TrialMetadata,
    sample_anchor_local_controls,
    sample_matched_negatives,
)

__all__ = [
    "AnchorLocalControlPair",
    "AnchorLocalControlResult",
    "MatchedNegativeResult",
    "TrialMetadata",
    "sample_anchor_local_controls",
    "sample_matched_negatives",
]
