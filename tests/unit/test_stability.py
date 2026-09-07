from __future__ import annotations

import pytest
import torch

from brain_evidence.recoverability.stability import (
    StabilityBankEntry,
    cross_session_stability,
)

PRONUNCIATION = ("K", "AE", "T")


def _entry(
    raw_contrast: list[float],
    *,
    split: str = "train",
    fold_id: str = "fold-bank",
    session_id: str = "session-bank",
    trial_id: str = "bank-1",
    checkpoint_id: str = "checkpoint-a",
    pronunciation: object = PRONUNCIATION,
    vocabulary_support: str = "full-vocabulary",
    prefix_policy: str = "student-rollout",
    alignment_policy: str = "token-index-v1",
    subject_id: str = "subject-1",
    dtype: torch.dtype = torch.float32,
) -> StabilityBankEntry:
    return StabilityBankEntry(
        raw_contrast=torch.tensor(raw_contrast, dtype=dtype, requires_grad=True),
        split=split,
        fold_id=fold_id,
        session_id=session_id,
        trial_id=trial_id,
        checkpoint_id=checkpoint_id,
        pronunciation=pronunciation,
        vocabulary_support=vocabulary_support,
        prefix_policy=prefix_policy,
        alignment_policy=alignment_policy,
        subject_id=subject_id,
    )


def _score(
    current: torch.Tensor,
    bank: list[StabilityBankEntry],
    probabilities: torch.Tensor,
    *,
    pronunciation: object = PRONUNCIATION,
):
    return cross_session_stability(
        current,
        bank,
        probabilities,
        pronunciation=pronunciation,
        current_fold_id="fold-current",
        current_session_id="session-current",
        current_trial_id="current-1",
        checkpoint_id="checkpoint-a",
        vocabulary_support="full-vocabulary",
        prefix_policy="student-rollout",
        alignment_policy="token-index-v1",
        current_subject_id="subject-1",
    )


@pytest.mark.parametrize(
    "invalid_pronunciation",
    ["", "   ", (), ("SH", ""), ("SH", " "), ("SH", 7), 7, None],
)
def test_stability_rejects_malformed_current_pronunciation(
    invalid_pronunciation: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="pronunciation"):
        _score(
            torch.tensor([1.0, -1.0]),
            [_entry([1.0, -1.0])],
            torch.tensor([0.5, 0.5]),
            pronunciation=invalid_pronunciation,
        )


def test_stability_rejects_malformed_bank_pronunciation() -> None:
    with pytest.raises(ValueError, match="pronunciation"):
        _score(
            torch.tensor([1.0, -1.0]),
            [_entry([1.0, -1.0], pronunciation="   ")],
            torch.tensor([0.5, 0.5]),
        )


@pytest.mark.parametrize(
    "field",
    [
        "split",
        "fold_id",
        "subject_id",
        "session_id",
        "trial_id",
        "checkpoint_id",
        "vocabulary_support",
        "prefix_policy",
        "alignment_policy",
    ],
)
@pytest.mark.parametrize("bad_value", ["   ", " stored-value", "stored-value "])
def test_stability_bank_entry_rejects_noncanonical_provenance(
    field: str,
    bad_value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        _entry([1.0, -1.0], **{field: bad_value})


@pytest.mark.parametrize(
    "field",
    [
        "current_fold_id",
        "current_subject_id",
        "current_session_id",
        "current_trial_id",
        "checkpoint_id",
        "vocabulary_support",
        "prefix_policy",
        "alignment_policy",
    ],
)
@pytest.mark.parametrize("bad_value", ["   ", " current-value", "current-value "])
def test_stability_rejects_noncanonical_current_provenance(
    field: str,
    bad_value: str,
) -> None:
    lookup = {
        "pronunciation": PRONUNCIATION,
        "current_fold_id": "fold-current",
        "current_subject_id": "subject-1",
        "current_session_id": "session-current",
        "current_trial_id": "current-1",
        "checkpoint_id": "checkpoint-a",
        "vocabulary_support": "full-vocabulary",
        "prefix_policy": "student-rollout",
        "alignment_policy": "token-index-v1",
    }
    lookup[field] = bad_value

    with pytest.raises(ValueError, match=field):
        cross_session_stability(
            torch.tensor([1.0, -1.0]),
            [],
            torch.tensor([0.5, 0.5]),
            **lookup,
        )


def test_stability_lookup_enforces_training_provenance_and_leave_outs() -> None:
    current = torch.tensor([1.0, -1.0, 0.0])
    probabilities = torch.tensor([0.25, 0.25, 0.5])
    opposite = [-1.0, 1.0, 0.0]
    bank = [
        _entry([12.0, 8.0, 10.0], trial_id="kept"),
        _entry(opposite, split="validation", trial_id="not-train"),
        _entry(opposite, fold_id="fold-current", trial_id="same-fold"),
        _entry(opposite, session_id="session-current", trial_id="same-session"),
        _entry(opposite, session_id="session-other", trial_id="current-1"),
        _entry(opposite, checkpoint_id="checkpoint-b", trial_id="wrong-checkpoint"),
        _entry(opposite, pronunciation=("D", "AO", "G"), trial_id="wrong-pron"),
        _entry(opposite, subject_id="subject-2", trial_id="wrong-subject"),
    ]

    result = _score(current, bank, probabilities)

    torch.testing.assert_close(result.score, torch.tensor(1.0))
    torch.testing.assert_close(result.support_count, torch.tensor(1))
    assert result.matched_trial_ids == ("kept",)
    assert not result.missing_support


def test_stability_requires_current_subject_for_exact_subject_lookup() -> None:
    with pytest.raises(TypeError, match="current_subject_id"):
        cross_session_stability(
            torch.tensor([1.0, -1.0]),
            [_entry([1.0, -1.0])],
            torch.tensor([0.5, 0.5]),
            pronunciation=PRONUNCIATION,
            current_fold_id="fold-current",
            current_session_id="session-current",
            current_trial_id="current-1",
            checkpoint_id="checkpoint-a",
            vocabulary_support="full-vocabulary",
            prefix_policy="student-rollout",
            alignment_policy="token-index-v1",
        )


def test_stability_averages_positive_fisher_cosines_and_detaches() -> None:
    current = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
    probabilities = torch.tensor([0.25, 0.25, 0.5], requires_grad=True)
    bank = [
        _entry([2.0, -2.0, 0.0], trial_id="aligned"),
        _entry([-1.0, 1.0, 0.0], trial_id="opposite"),
    ]

    result = _score(current, bank, probabilities)

    torch.testing.assert_close(result.score, torch.tensor(0.5))
    torch.testing.assert_close(result.support_count, torch.tensor(2))
    assert not result.missing_support
    assert not result.score.requires_grad
    assert not result.missing_support.requires_grad


def test_stability_ignores_zero_norm_bank_directions() -> None:
    result = _score(
        torch.tensor([1.0, -1.0]),
        [
            _entry([0.0, 0.0], trial_id="zero"),
            _entry([3.0, -3.0], trial_id="aligned"),
        ],
        torch.tensor([0.5, 0.5]),
    )

    torch.testing.assert_close(result.score, torch.tensor(1.0))
    torch.testing.assert_close(result.support_count, torch.tensor(1))
    assert not result.missing_support


def test_fp16_stability_promotes_before_norm_arithmetic() -> None:
    result = _score(
        torch.tensor(
            [1e-4, -1e-4],
            dtype=torch.float16,
            requires_grad=True,
        ),
        [
            _entry(
                [1.0, -1.0],
                trial_id="aligned",
                dtype=torch.float16,
            )
        ],
        torch.tensor([0.5, 0.5], dtype=torch.float16, requires_grad=True),
    )

    torch.testing.assert_close(result.score, torch.tensor(1.0))
    torch.testing.assert_close(result.support_count, torch.tensor(1))
    assert result.score.dtype == torch.float32
    assert torch.isfinite(result.score)
    assert not result.missing_support


def test_fp16_stability_preserves_valid_rounded_probability_mass() -> None:
    result = _score(
        torch.tensor(
            [2e-4, -1e-4, -1e-4],
            dtype=torch.float16,
            requires_grad=True,
        ),
        [
            _entry(
                [1.0, -0.5, -0.5],
                trial_id="aligned",
                dtype=torch.float16,
            )
        ],
        torch.tensor(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=torch.float16,
            requires_grad=True,
        ),
    )

    torch.testing.assert_close(result.score, torch.tensor(1.0))
    assert not result.missing_support


def test_float64_stability_preserves_small_opposition_on_large_baseline() -> None:
    result = _score(
        torch.tensor(
            [1e10 + 1.0, 1e10 - 1.0],
            dtype=torch.float64,
            requires_grad=True,
        ),
        [
            _entry(
                [1e10 - 1.0, 1e10 + 1.0],
                trial_id="opposite",
                dtype=torch.float64,
            )
        ],
        torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True),
    )

    torch.testing.assert_close(result.score, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(result.support_count, torch.tensor(1))
    assert result.score.dtype == torch.float64
    assert not result.missing_support


@pytest.mark.parametrize("bank", [[], [_entry([1.0, -1.0], split="validation")]])
def test_missing_cross_session_support_fails_closed(bank) -> None:
    result = _score(
        torch.tensor([1.0, -1.0], requires_grad=True),
        bank,
        torch.tensor([0.5, 0.5], requires_grad=True),
    )

    torch.testing.assert_close(result.score, torch.tensor(0.0))
    torch.testing.assert_close(result.support_count, torch.tensor(0))
    assert result.missing_support
    assert result.matched_trial_ids == ()
    assert not result.score.requires_grad


@pytest.mark.parametrize(
    ("current", "bank_direction"),
    [([1.0, -1.0], [0.0, 0.0]), ([0.0, 0.0], [1.0, -1.0])],
)
def test_zero_norm_cross_session_support_fails_closed(current, bank_direction) -> None:
    result = _score(
        torch.tensor(current, requires_grad=True),
        [_entry(bank_direction)],
        torch.tensor([0.5, 0.5], requires_grad=True),
    )

    assert result.score.item() == 0.0
    assert result.support_count.item() == 0
    assert result.missing_support.item() is True
    assert result.matched_trial_ids == ("bank-1",)
    assert not result.score.requires_grad


def test_missing_support_closes_only_the_unsupported_token() -> None:
    result = _score(
        torch.tensor([[1.0, -1.0], [1.0, -1.0]], requires_grad=True),
        [_entry([[2.0, -2.0], [0.0, 0.0]])],
        torch.full((2, 2), 0.5, requires_grad=True),
    )

    torch.testing.assert_close(result.score, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result.support_count, torch.tensor([1, 0]))
    torch.testing.assert_close(result.missing_support, torch.tensor([False, True]))
    assert not result.score.requires_grad


@pytest.mark.parametrize(
    ("entry_override", "message"),
    [
        ({"vocabulary_support": "top-k"}, "vocabulary support"),
        ({"prefix_policy": "gold-prefix"}, "prefix policy"),
        ({"alignment_policy": "word-aligned"}, "alignment policy"),
    ],
)
def test_stability_rejects_incompatible_geometry_policies(
    entry_override: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _score(
            torch.tensor([1.0, -1.0]),
            [_entry([1.0, -1.0], **entry_override)],
            torch.tensor([0.5, 0.5]),
        )
