"""Cross-fitted Gaussian encoding likelihood for projected raw brain data.

This module implements the fixed P1 calibration specification.  It does not
select hyperparameters and its density is over a seeded projection of the raw
brain vector, rather than over the full voxel vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECTION_SEED = 1729
PROJECTION_DIMENSION = 64
RIDGE_ALPHA = 1.0
COVARIANCE_SHRINKAGE = 0.1
COVARIANCE_JITTER = 1e-6
MODEL_SEMANTICS = "p(projected_brain | fixed_image_features, subject)"
SCORE_SEMANTICS = "normalized_gaussian_log_density_projected_brain"


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _ids(value: Sequence[Any], name: str, rows: int) -> np.ndarray:
    raw = list(value)
    if any(not isinstance(item, (str, np.str_)) or not item for item in raw):
        raise ValueError(f"{name} entries must be non-empty strings")
    array = np.asarray(raw)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"{name} must have one entry for every input row")
    return np.asarray(array, dtype=np.str_)


def _group_folds(image_ids: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    groups = np.unique(image_ids)
    if folds < 2 or len(groups) < folds:
        raise ValueError(f"at least {folds} distinct image groups are required")
    shuffled = groups.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    by_image = {str(group): int(index % folds) for index, group in enumerate(shuffled)}
    return np.asarray([by_image[str(group)] for group in image_ids], dtype=np.int64), by_image


def _projection(brain_dimension: int) -> np.ndarray:
    output_dimension = min(PROJECTION_DIMENSION, brain_dimension)
    rng = np.random.default_rng(PROJECTION_SEED)
    return rng.standard_normal((brain_dimension, output_dimension)) / np.sqrt(brain_dimension)


def _fit_ridge(features: np.ndarray, targets: np.ndarray, image_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, counts = np.unique(image_ids, return_counts=True)
    count_by_image = dict(zip(np.unique(image_ids), counts))
    weights = np.asarray([1.0 / count_by_image[group] for group in image_ids])
    weights /= weights.sum()
    feature_mean = np.sum(features * weights[:, None], axis=0)
    target_mean = np.sum(targets * weights[:, None], axis=0)
    centered_features = features - feature_mean
    centered_targets = targets - target_mean
    root_weight = np.sqrt(weights)[:, None]
    weighted_features = centered_features * root_weight
    weighted_targets = centered_targets * root_weight
    gram = weighted_features.T @ weighted_features
    coefficients = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(features.shape[1]),
        weighted_features.T @ weighted_targets,
    )
    intercept = target_mean - feature_mean @ coefficients
    return coefficients, intercept


def _residual_covariance(residuals: np.ndarray) -> np.ndarray:
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(residuals) - 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - COVARIANCE_SHRINKAGE) * covariance
        + COVARIANCE_SHRINKAGE * diagonal
        + COVARIANCE_JITTER * np.eye(residuals.shape[1])
    )
    if not np.all(np.isfinite(covariance)) or np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise ValueError("residual covariance is not positive definite")
    return covariance


def _covariance_log_determinant(covariance: np.ndarray) -> float:
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ValueError("covariance must be positive definite and symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    log_determinant = float(2.0 * np.log(np.diag(cholesky)).sum())
    if not np.isfinite(log_determinant):
        raise ValueError("covariance must be positive definite and finite")
    return log_determinant


def gaussian_log_density(observations: Any, means: Any, covariance: Any) -> np.ndarray:
    """Return normalized multivariate Gaussian log density for paired rows."""
    observed = _matrix(observations, "observations")
    expected = _matrix(means, "means")
    cov = _matrix(covariance, "covariance")
    if observed.shape != expected.shape:
        raise ValueError("observations and means must have the same shape")
    dimension = observed.shape[1]
    if cov.shape != (dimension, dimension):
        raise ValueError("covariance dimension must match observations")
    log_determinant = _covariance_log_determinant(cov)
    try:
        solved = np.linalg.solve(cov, (observed - expected).T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    quadratic = np.sum((observed - expected) * solved, axis=1)
    result = -0.5 * (quadratic + log_determinant + dimension * np.log(2.0 * np.pi))
    if not np.all(np.isfinite(result)):
        raise ValueError("Gaussian log density was nonfinite")
    return result


@dataclass(frozen=True)
class GaussianEncodingLikelihood:
    projection: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    covariance: np.ndarray
    provenance: Mapping[str, Any]

    @classmethod
    def fit(
        cls,
        image_features: Any,
        brain: Any,
        *,
        image_ids: Sequence[Any],
        sample_ids: Sequence[Any],
        feature_provenance: str,
        brain_provenance: str,
        inner_folds: int = 5,
    ) -> "GaussianEncodingLikelihood":
        features = _matrix(image_features, "image_features")
        raw_brain = _matrix(brain, "brain")
        if len(features) != len(raw_brain):
            raise ValueError("image_features and brain must have the same number of rows")
        groups = _ids(image_ids, "image_ids", len(features))
        samples = _ids(sample_ids, "sample_ids", len(features))
        if not feature_provenance or not brain_provenance:
            raise ValueError("feature and brain provenance must be explicit and non-empty")
        fold_ids, fold_by_image = _group_folds(groups, inner_folds, PROJECTION_SEED + 1)
        projection = _projection(raw_brain.shape[1])
        targets = raw_brain @ projection
        oof_predictions = np.empty_like(targets)
        for fold in range(inner_folds):
            train = fold_ids != fold
            held_out = ~train
            coefficients, intercept = _fit_ridge(features[train], targets[train], groups[train])
            oof_predictions[held_out] = features[held_out] @ coefficients + intercept
        covariance = _residual_covariance(targets - oof_predictions)
        coefficients, intercept = _fit_ridge(features, targets, groups)
        provenance = {
            "model_semantics": MODEL_SEMANTICS,
            "score_semantics": SCORE_SEMANTICS,
            "train_sample_ids": samples.tolist(),
            "train_image_ids": sorted(set(groups.tolist())),
            "inner_fold_by_image": fold_by_image,
            "projection_seed": PROJECTION_SEED,
            "projection_dimension": int(projection.shape[1]),
            "ridge_alpha": RIDGE_ALPHA,
            "covariance_shrinkage": COVARIANCE_SHRINKAGE,
            "covariance_jitter": COVARIANCE_JITTER,
            "feature_provenance": feature_provenance,
            "brain_provenance": brain_provenance,
        }
        return cls(projection, coefficients, intercept, covariance, provenance)

    def score(self, brain: Any, candidate_features: Any, *, no_evidence: bool = False) -> np.ndarray:
        raw_brain = _matrix(brain, "brain")
        features = _matrix(candidate_features, "candidate_features")
        if raw_brain.shape[1] != self.projection.shape[0]:
            raise ValueError("brain feature dimension does not match the fitted model")
        if features.shape[1] != self.coefficients.shape[0]:
            raise ValueError("image feature dimension does not match the fitted model")
        if no_evidence:
            return np.zeros((len(raw_brain), len(features)), dtype=np.float64)
        observations = raw_brain @ self.projection
        means = features @ self.coefficients + self.intercept
        differences = observations[:, None, :] - means[None, :, :]
        dimension = differences.shape[2]
        log_determinant = _covariance_log_determinant(self.covariance)
        solved = np.linalg.solve(self.covariance, differences.reshape(-1, dimension).T).T
        quadratic = np.sum(differences.reshape(-1, dimension) * solved, axis=1)
        scores = -0.5 * (
            quadratic + log_determinant + dimension * np.log(2.0 * np.pi)
        )
        scores = scores.reshape(len(raw_brain), len(features))
        if not np.all(np.isfinite(scores)):
            raise ValueError("Gaussian scores were nonfinite")
        return scores

    def score_paired(
        self, brain: Any, image_features: Any, *, no_evidence: bool = False
    ) -> np.ndarray:
        """Score corresponding brain and image-feature rows in linear memory."""
        raw_brain = _matrix(brain, "brain")
        features = _matrix(image_features, "image_features")
        if len(raw_brain) != len(features):
            raise ValueError("brain and image_features must have the same number of rows")
        if raw_brain.shape[1] != self.projection.shape[0]:
            raise ValueError("brain feature dimension does not match the fitted model")
        if features.shape[1] != self.coefficients.shape[0]:
            raise ValueError("image feature dimension does not match the fitted model")
        if no_evidence:
            return np.zeros(len(raw_brain), dtype=np.float64)
        observations = raw_brain @ self.projection
        means = features @ self.coefficients + self.intercept
        return gaussian_log_density(observations, means, self.covariance)

    def save(self, path: str | Path) -> None:
        metadata = json.dumps(dict(self.provenance), sort_keys=True, separators=(",", ":"))
        with Path(path).open("wb") as artifact:
            np.savez_compressed(
                artifact,
                projection=self.projection,
                coefficients=self.coefficients,
                intercept=self.intercept,
                covariance=self.covariance,
                metadata=np.asarray(metadata),
            )

    @classmethod
    def load(cls, path: str | Path) -> "GaussianEncodingLikelihood":
        try:
            with np.load(path, allow_pickle=False) as artifact:
                required = {"projection", "coefficients", "intercept", "covariance", "metadata"}
                if not required.issubset(artifact.files):
                    raise ValueError("calibration artifact is missing required fields")
                provenance = json.loads(str(artifact["metadata"]))
                model = cls(
                    np.asarray(artifact["projection"], dtype=np.float64),
                    np.asarray(artifact["coefficients"], dtype=np.float64),
                    np.asarray(artifact["intercept"], dtype=np.float64),
                    np.asarray(artifact["covariance"], dtype=np.float64),
                    provenance,
                )
        except (OSError, json.JSONDecodeError, KeyError) as error:
            raise ValueError("invalid calibration artifact") from error
        model._validate_loaded()
        return model

    def _validate_loaded(self) -> None:
        projection = _matrix(self.projection, "projection")
        coefficients = _matrix(self.coefficients, "coefficients")
        covariance = _matrix(self.covariance, "covariance")
        intercept = np.asarray(self.intercept, dtype=np.float64)
        dimension = projection.shape[1]
        if coefficients.shape[1] != dimension or intercept.shape != (dimension,):
            raise ValueError("artifact coefficient dimensions are inconsistent")
        if covariance.shape != (dimension, dimension):
            raise ValueError("artifact covariance must be positive definite")
        _covariance_log_determinant(covariance)
        required = {"train_sample_ids", "train_image_ids", "inner_fold_by_image",
                    "projection_seed", "projection_dimension", "feature_provenance",
                    "brain_provenance", "score_semantics", "model_semantics"}
        if not isinstance(self.provenance, dict) or not required.issubset(self.provenance):
            raise ValueError("artifact provenance is incomplete")
        if self.provenance["score_semantics"] != SCORE_SEMANTICS:
            raise ValueError("artifact score semantics are unsupported")


@dataclass(frozen=True)
class CrossFitResult:
    scores: np.ndarray
    fold_ids: np.ndarray
    models: Mapping[int, GaussianEncodingLikelihood]


def crossfit_gaussian_encoding(
    image_features: Any,
    brain: Any,
    *,
    image_ids: Sequence[Any],
    sample_ids: Sequence[Any],
    feature_provenance: str,
    brain_provenance: str,
    outer_folds: int = 5,
    inner_folds: int = 5,
) -> CrossFitResult:
    """Fit complete outer image-group folds and return paired OOF densities."""
    features = _matrix(image_features, "image_features")
    raw_brain = _matrix(brain, "brain")
    if len(features) != len(raw_brain):
        raise ValueError("image_features and brain must have the same number of rows")
    groups = _ids(image_ids, "image_ids", len(features))
    samples = _ids(sample_ids, "sample_ids", len(features))
    fold_ids, _ = _group_folds(groups, outer_folds, PROJECTION_SEED + 2)
    scores = np.empty(len(features), dtype=np.float64)
    models: dict[int, GaussianEncodingLikelihood] = {}
    for fold in range(outer_folds):
        train = fold_ids != fold
        held_out = ~train
        model = GaussianEncodingLikelihood.fit(
            features[train],
            raw_brain[train],
            image_ids=groups[train],
            sample_ids=samples[train],
            feature_provenance=feature_provenance,
            brain_provenance=brain_provenance,
            inner_folds=inner_folds,
        )
        scores[held_out] = model.score_paired(raw_brain[held_out], features[held_out])
        models[fold] = model
    if not np.all(np.isfinite(scores)):
        raise ValueError("cross-fitted scores were nonfinite")
    return CrossFitResult(scores=scores, fold_ids=fold_ids, models=models)


__all__ = [
    "CrossFitResult",
    "GaussianEncodingLikelihood",
    "crossfit_gaussian_encoding",
    "gaussian_log_density",
]
