"""Matched-wrong-brain dependence diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BrainDependence:
    """A method's larger-is-better brain gap and its difference from CE."""

    brain_gap: float
    ce_brain_gap: float
    difference_in_gap: float


def brain_dependence(
    *,
    real_score: float,
    wrong_brain_score: float,
    ce_real_score: float,
    ce_wrong_brain_score: float,
) -> BrainDependence:
    """Compute real-minus-wrong and difference-in-gap against CE.

    Inputs must share a larger-is-better orientation, such as IoU or negative
    WER. Under that convention a positive brain gap means the real recording
    outperforms its matched-wrong control.
    """

    scores = (
        float(real_score),
        float(wrong_brain_score),
        float(ce_real_score),
        float(ce_wrong_brain_score),
    )
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("brain-dependence scores must be finite")
    method_gap = scores[0] - scores[1]
    ce_gap = scores[2] - scores[3]
    return BrainDependence(
        brain_gap=method_gap,
        ce_brain_gap=ce_gap,
        difference_in_gap=method_gap - ce_gap,
    )


__all__ = ["BrainDependence", "brain_dependence"]
