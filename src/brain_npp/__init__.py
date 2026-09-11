"""Neural posterior primitives for candidate-based inference."""

from .posterior import (
    NeuralPosterior,
    ScoreSemantics,
    build_neural_posterior,
    normalize_candidate_prior,
)
from .protocols import PosteriorTeacherStudent, Rollout

__all__ = [
    "NeuralPosterior",
    "ScoreSemantics",
    "build_neural_posterior",
    "normalize_candidate_prior",
    "PosteriorTeacherStudent",
    "Rollout",
]
