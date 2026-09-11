from __future__ import annotations

import json

import numpy as np
import pytest

from brain_npp.calibration import (
    GaussianEncodingLikelihood,
    crossfit_gaussian_encoding,
    gaussian_log_density,
)


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_features = np.array(
        [[-2.0], [-2.0], [-1.0], [-1.0], [1.0], [1.0], [2.0], [2.0]]
    )
    brain = np.column_stack(
        [2.0 * image_features[:, 0] + 0.1, -image_features[:, 0] + 0.2]
    )
    brain += np.array(
        [[0.05, 0.01], [-0.05, -0.01], [0.02, -0.03], [-0.02, 0.03],
         [0.03, 0.02], [-0.03, -0.02], [0.04, -0.01], [-0.04, 0.01]]
    )
    image_ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    sample_ids = np.array([f"trial-{i}" for i in range(8)])
    return image_features, brain, image_ids, sample_ids


def test_gaussian_log_density_matches_independent_analytic_density() -> None:
    observations = np.array([[1.0, -1.0], [0.0, 2.0]])
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    covariance = np.diag([4.0, 9.0])

    got = gaussian_log_density(observations, means, covariance)

    differences = observations - means
    quadratic = differences[:, 0] ** 2 / 4.0 + differences[:, 1] ** 2 / 9.0
    expected = -0.5 * (quadratic + np.log(36.0) + 2.0 * np.log(2.0 * np.pi))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_gaussian_log_density_rejects_a_nonsymmetric_covariance() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_log_density(
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            np.array([[1.0, 2.0], [0.0, 1.0]]),
        )


def test_ids_reject_mixed_values_before_numpy_coerces_them_to_strings() -> None:
    features, brain, image_ids, sample_ids = _dataset()
    mixed = ["a", "a", "b", "b", "c", "c", "d", 7]
    with pytest.raises(ValueError, match="non-empty strings"):
        crossfit_gaussian_encoding(features, brain, image_ids=mixed,
            sample_ids=sample_ids, feature_provenance="x", brain_provenance="y",
            outer_folds=2, inner_folds=2)


def test_crossfit_keeps_repetitions_in_the_same_held_out_fold() -> None:
    features, brain, image_ids, sample_ids = _dataset()

    result = crossfit_gaussian_encoding(
        features,
        brain,
        image_ids=image_ids,
        sample_ids=sample_ids,
        feature_provenance="fixture-features-v1",
        brain_provenance="fixture-voxels-v1",
        outer_folds=2,
        inner_folds=2,
    )

    for image_id in np.unique(image_ids):
        assert np.unique(result.fold_ids[image_ids == image_id]).size == 1
    for fold_id, model in result.models.items():
        held_out = set(image_ids[result.fold_ids == fold_id])
        assert held_out.isdisjoint(model.provenance["train_image_ids"])


def test_crossfit_uses_rowwise_paired_scoring_without_calling_all_pairs(monkeypatch) -> None:
    features, brain, image_ids, sample_ids = _dataset()

    def reject_all_pairs(*args, **kwargs):
        raise AssertionError("crossfit must not call all-pairs score")

    monkeypatch.setattr(GaussianEncodingLikelihood, "score", reject_all_pairs)

    result = crossfit_gaussian_encoding(
        features, brain, image_ids=image_ids, sample_ids=sample_ids,
        feature_provenance="features", brain_provenance="brain",
        outer_folds=2, inner_folds=2,
    )

    assert result.scores.shape == (len(features),)
    assert np.all(np.isfinite(result.scores))


def test_paired_scores_equal_the_diagonal_of_all_pairs_scores() -> None:
    features, brain, image_ids, sample_ids = _dataset()
    model = GaussianEncodingLikelihood.fit(
        features, brain, image_ids=image_ids, sample_ids=sample_ids,
        feature_provenance="features", brain_provenance="brain", inner_folds=2,
    )

    paired = model.score_paired(brain[:3], features[:3])

    np.testing.assert_allclose(paired, np.diag(model.score(brain[:3], features[:3])))


def test_held_out_brain_cannot_change_its_fold_model_parameters() -> None:
    features, brain, image_ids, sample_ids = _dataset()
    baseline = crossfit_gaussian_encoding(
        features, brain, image_ids=image_ids, sample_ids=sample_ids,
        feature_provenance="features", brain_provenance="brain",
        outer_folds=2, inner_folds=2,
    )
    held_fold = int(baseline.fold_ids[0])
    perturbed_brain = brain.copy()
    perturbed_brain[baseline.fold_ids == held_fold] += 10_000.0

    perturbed = crossfit_gaussian_encoding(
        features, perturbed_brain, image_ids=image_ids, sample_ids=sample_ids,
        feature_provenance="features", brain_provenance="brain",
        outer_folds=2, inner_folds=2,
    )

    first, second = baseline.models[held_fold], perturbed.models[held_fold]
    np.testing.assert_array_equal(first.projection, second.projection)
    np.testing.assert_allclose(first.coefficients, second.coefficients)
    np.testing.assert_allclose(first.intercept, second.intercept)
    np.testing.assert_allclose(first.covariance, second.covariance)


def test_serialization_preserves_scores_and_required_provenance(tmp_path) -> None:
    features, brain, image_ids, sample_ids = _dataset()
    model = GaussianEncodingLikelihood.fit(
        features, brain, image_ids=image_ids, sample_ids=sample_ids,
        feature_provenance="clip-fixture@abc", brain_provenance="nsd-voxels@def",
        inner_folds=2,
    )
    before = model.score(brain[:2], features[[0, 2, 4]])
    path = tmp_path / "calibration.npz"

    model.save(path)
    restored = GaussianEncodingLikelihood.load(path)

    np.testing.assert_allclose(restored.score(brain[:2], features[[0, 2, 4]]), before)
    assert restored.provenance["train_sample_ids"] == sample_ids.tolist()
    assert restored.provenance["train_image_ids"] == sorted(set(image_ids))
    assert restored.provenance["feature_provenance"] == "clip-fixture@abc"
    assert restored.provenance["brain_provenance"] == "nsd-voxels@def"
    assert restored.provenance["score_semantics"] == "normalized_gaussian_log_density_projected_brain"
    assert restored.provenance["projection_seed"] == 1729
    assert "inner_fold_by_image" in restored.provenance
    metadata = json.loads(str(np.load(path, allow_pickle=False)["metadata"]))
    assert metadata["model_semantics"].startswith("p(projected_brain")


def test_covariance_is_positive_definite_and_no_evidence_is_exactly_zero() -> None:
    features, brain, image_ids, sample_ids = _dataset()
    model = GaussianEncodingLikelihood.fit(
        features, brain, image_ids=image_ids, sample_ids=sample_ids,
        feature_provenance="features", brain_provenance="brain", inner_folds=2,
    )

    assert np.all(np.linalg.eigvalsh(model.covariance) > 0.0)
    scores = model.score(brain[:2], features[:3], no_evidence=True)
    np.testing.assert_array_equal(scores, np.zeros((2, 3)))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("few_groups", "at least"),
        ("nonfinite_features", "finite"),
        ("nonfinite_brain", "finite"),
        ("row_mismatch", "same number"),
        ("id_mismatch", "image_ids"),
    ],
)
def test_fit_fails_closed_on_invalid_training_data(mutation: str, match: str) -> None:
    features, brain, image_ids, sample_ids = _dataset()
    if mutation == "few_groups":
        image_ids[:] = "a"
    elif mutation == "nonfinite_features":
        features[0, 0] = np.nan
    elif mutation == "nonfinite_brain":
        brain[0, 0] = np.inf
    elif mutation == "row_mismatch":
        brain = brain[:-1]
    elif mutation == "id_mismatch":
        image_ids = image_ids[:-1]

    with pytest.raises(ValueError, match=match):
        GaussianEncodingLikelihood.fit(
            features, brain, image_ids=image_ids, sample_ids=sample_ids,
            feature_provenance="features", brain_provenance="brain", inner_folds=2,
        )


@pytest.mark.parametrize("bad_id", [None, np.nan, np.inf, 1])
def test_fit_requires_nonempty_string_identifiers(bad_id) -> None:
    features, brain, image_ids, sample_ids = _dataset()
    invalid_image_ids = image_ids.astype(object)
    invalid_image_ids[0] = bad_id

    with pytest.raises(ValueError, match="image_ids.*strings"):
        GaussianEncodingLikelihood.fit(
            features, brain, image_ids=invalid_image_ids, sample_ids=sample_ids,
            feature_provenance="features", brain_provenance="brain", inner_folds=2,
        )


def test_fit_rejects_typed_identifier_collision_instead_of_normalizing_it() -> None:
    features, brain, image_ids, sample_ids = _dataset()
    invalid_sample_ids = sample_ids.astype(object)
    invalid_sample_ids[:2] = [1, "1"]

    with pytest.raises(ValueError, match="sample_ids.*strings"):
        GaussianEncodingLikelihood.fit(
            features, brain, image_ids=image_ids, sample_ids=invalid_sample_ids,
            feature_provenance="features", brain_provenance="brain", inner_folds=2,
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_score_fails_closed_on_nonfinite_or_dimension_mismatch(bad: float) -> None:
    features, brain, image_ids, sample_ids = _dataset()
    model = GaussianEncodingLikelihood.fit(
        features, brain, image_ids=image_ids, sample_ids=sample_ids,
        feature_provenance="features", brain_provenance="brain", inner_folds=2,
    )
    invalid_brain = brain[:1].copy()
    invalid_brain[0, 0] = bad

    with pytest.raises(ValueError, match="finite"):
        model.score(invalid_brain, features[:1])
    with pytest.raises(ValueError, match="brain feature dimension"):
        model.score(np.ones((1, 3)), features[:1])
    with pytest.raises(ValueError, match="image feature dimension"):
        model.score(brain[:1], np.ones((1, 2)))
