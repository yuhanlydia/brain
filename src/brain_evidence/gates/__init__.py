"""Detached evidence gates and ablations."""

from .counterfactual import dual_cosine_gate, legacy_dual_cosine_gate
from .reliability import (
    CalibrationProvenance,
    NullCalibration,
    ReliabilityCalibration,
    calibrated_reliability,
    neural_reliability,
    uncalibrated_reliability,
)

__all__ = [
    "CalibrationProvenance",
    "NullCalibration",
    "ReliabilityCalibration",
    "calibrated_reliability",
    "dual_cosine_gate",
    "legacy_dual_cosine_gate",
    "neural_reliability",
    "uncalibrated_reliability",
]
