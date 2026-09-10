"""Neural posterior primitives for candidate-based inference."""

from .posterior import (
    NeuralPosterior,
    build_neural_posterior,
    normalize_candidate_prior,
)
from .protocols import PosteriorTeacherStudent

__all__ = [
    "NeuralPosterior",
    "build_neural_posterior",
    "normalize_candidate_prior",
    "PosteriorTeacherStudent",
]
