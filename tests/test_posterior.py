import pytest
import torch

from brain_npp.posterior import build_neural_posterior, normalize_candidate_prior


def test_build_neural_posterior_matches_hand_computed_fixture():
    log_likelihood = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64))
    log_prior = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))

    result = build_neural_posterior(log_likelihood, log_prior)

    torch.testing.assert_close(
        result.posterior.exp(),
        torch.tensor([[0.8, 0.2]], dtype=torch.float64),
    )
    assert result.information_gain.item() == pytest.approx(0.1927447570)


def test_normalize_candidate_prior_normalizes_unnormalized_log_weights():
    log_prior = torch.log(torch.tensor([[4.0, 1.0]], dtype=torch.float64))

    normalized = normalize_candidate_prior(log_prior)

    torch.testing.assert_close(
        normalized.exp(), torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    )


def test_build_neural_posterior_excludes_padded_candidates_from_normalization():
    log_likelihood = torch.log(torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float64))
    log_prior = torch.zeros_like(log_likelihood)
    candidate_mask = torch.tensor([[True, False, True]])

    result = build_neural_posterior(log_likelihood, log_prior, candidate_mask)

    torch.testing.assert_close(
        result.posterior.exp(),
        torch.tensor([[0.125, 0.0, 0.875]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.prior.exp(), torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64)
    )
    assert torch.isneginf(result.posterior[0, 1])
    assert torch.isneginf(result.prior[0, 1])


def test_build_neural_posterior_rejects_all_masked_rows():
    log_scores = torch.zeros((1, 2))

    with pytest.raises(ValueError, match="at least one valid candidate"):
        build_neural_posterior(log_scores, log_scores, torch.tensor([[False, False]]))


def test_build_neural_posterior_rejects_mismatched_input_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        build_neural_posterior(torch.zeros((1, 2)), torch.zeros((1, 3)))


def test_build_neural_posterior_rejects_non_batch_first_inputs():
    with pytest.raises(ValueError, match="rank 2"):
        build_neural_posterior(torch.zeros(2), torch.zeros(2))


def test_build_neural_posterior_is_invariant_to_likelihood_logit_shifts():
    log_likelihood = torch.tensor([[-1.0, -2.0, -3.0]])
    log_prior = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))

    original = build_neural_posterior(log_likelihood, log_prior)
    shifted = build_neural_posterior(log_likelihood + 1234.5, log_prior)

    torch.testing.assert_close(shifted.posterior, original.posterior)
    torch.testing.assert_close(shifted.information_gain, original.information_gain)


def test_build_neural_posterior_remains_finite_for_very_negative_scores():
    log_likelihood = torch.tensor([[-10000.0, -10001.0]], dtype=torch.float64)
    log_prior = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))

    result = build_neural_posterior(log_likelihood, log_prior)

    assert torch.isfinite(result.posterior).all()
    torch.testing.assert_close(
        result.posterior.exp(),
        torch.softmax(torch.tensor([[0.0, -1.0]], dtype=torch.float64), dim=-1),
    )


def test_build_neural_posterior_rejects_rows_without_prior_support():
    log_likelihood = torch.zeros((1, 2))
    log_prior = torch.full((1, 2), -torch.inf)

    with pytest.raises(ValueError, match="prior support"):
        build_neural_posterior(log_likelihood, log_prior)


@pytest.mark.parametrize(
    "dtype,offset",
    [(torch.float16, -10000.0), (torch.bfloat16, -10000.0),
     (torch.float32, -1e8), (torch.float64, -1e16)],
)
@pytest.mark.parametrize("masked", [False, True])
def test_large_common_prior_offset_preserves_unit_mass_and_information_gain(dtype, offset, masked):
    """Catches losing log(2) when subtracting a large uncentered normalizer."""
    mask = torch.tensor([[True, True, not masked]])
    log_prior = torch.full((1, 3), offset, dtype=dtype)
    if masked:
        log_prior[:, 2] = torch.nan
    result = build_neural_posterior(torch.zeros_like(log_prior), log_prior, mask)

    assert torch.isfinite(result.prior[mask]).all()
    assert torch.isfinite(result.posterior[mask]).all()
    assert result.prior.exp().sum().item() == pytest.approx(1.0, abs=1e-7)
    assert result.posterior.exp().sum().item() == pytest.approx(1.0, abs=1e-7)
    assert result.prior.exp()[~mask].sum().item() == 0.0
    assert result.information_gain.item() >= 0.0
    assert result.information_gain.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_prior_normalization_is_shift_invariant_with_representable_differences(dtype):
    """Catches precision loss in low-precision arithmetic even with unequal weights."""
    log_prior = torch.tensor([[0.0, -128.0, -256.0]], dtype=dtype)
    mask = torch.tensor([[True, True, False]])
    normalized = normalize_candidate_prior(log_prior - 8192.0, mask)
    torch.testing.assert_close(
        normalized.double(),
        torch.tensor([[0.0, -128.0, -torch.inf]], dtype=torch.float64),
        atol=1e-7, rtol=1e-7,
    )
    baseline = normalize_candidate_prior(log_prior, mask)
    torch.testing.assert_close(normalized, baseline)


def test_extreme_score_outside_prior_support_cannot_remove_valid_posterior():
    result = build_neural_posterior(
        torch.tensor([[-3e38, 3e38]]), torch.tensor([[0.0, -torch.inf]])
    )
    torch.testing.assert_close(result.posterior.exp(), torch.tensor([[1.0, 0.0]]))


def test_compatibility_temperature_rescales_extreme_scores_before_centering():
    result = build_neural_posterior(
        torch.tensor([[-3e38, 3e38]]), torch.zeros((1, 2)),
        score_semantics="compatibility", compatibility_temperature=1e38,
    )
    torch.testing.assert_close(
        result.posterior.exp(), torch.tensor([[0.002472623, 0.997527377]])
    )
