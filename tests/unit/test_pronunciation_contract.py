from __future__ import annotations

import pytest
import torch

from brain_evidence.controls.matched_negative import (
    TrialMetadata,
    sample_matched_negatives,
)
from brain_evidence.pronunciation import canonicalize_pronunciation
from brain_evidence.recoverability.interaction import AnchorLocalValidity
from brain_evidence.recoverability.stability import (
    StabilityBankEntry,
    cross_session_stability,
)


def _trial(trial_id: str, pronunciation: str | tuple[str, ...]) -> TrialMetadata:
    return TrialMetadata(
        trial_id=trial_id,
        subject_id="subject-1",
        session_id="session-1",
        split="train",
        fold_id="fold-1",
        pronunciation=pronunciation,
        duration=1.0,
        reliability_stratum="high",
    )


def _bank_entry(
    trial_id: str,
    pronunciation: str | tuple[str, ...],
    raw_contrast: tuple[float, float],
) -> StabilityBankEntry:
    return StabilityBankEntry(
        raw_contrast=torch.tensor(raw_contrast),
        split="train",
        fold_id="fold-bank",
        subject_id="subject-1",
        session_id=f"session-{trial_id}",
        trial_id=trial_id,
        checkpoint_id="checkpoint-a",
        pronunciation=pronunciation,
        vocabulary_support="full-vocabulary",
        prefix_policy="student-rollout",
        alignment_policy="token-index-v1",
    )


def test_pronunciation_tokens_still_normalize_surrounding_whitespace() -> None:
    assert canonicalize_pronunciation("  SH   AH0 ") == ("SH", "AH0")
    assert canonicalize_pronunciation((" SH ", "AH0 ")) == ("SH", "AH0")


def test_sampler_treats_unspaced_multichar_phone_as_one_token() -> None:
    trials = [
        _trial("anchor", "SH"),
        _trial("same-atomic-phone", ("SH",)),
        _trial("different-two-phone-sequence", ("S", "H")),
    ]

    result = sample_matched_negatives(
        trials,
        anchor_index=0,
        k=2,
        duration_tolerance=0.0,
        seed=41,
    )

    assert result.indices == (2,)
    assert result.eligible_count == 1


def test_stability_uses_the_same_atomic_phone_identity_as_sampler() -> None:
    result = cross_session_stability(
        torch.tensor([1.0, -1.0]),
        [
            _bank_entry("same-atomic-phone", ("SH",), (1.0, -1.0)),
            _bank_entry("different-two-phone-sequence", ("S", "H"), (-1.0, 1.0)),
        ],
        torch.tensor([0.5, 0.5]),
        pronunciation="SH",
        current_fold_id="fold-current",
        current_subject_id="subject-1",
        current_session_id="session-current",
        current_trial_id="trial-current",
        checkpoint_id="checkpoint-a",
        vocabulary_support="full-vocabulary",
        prefix_policy="student-rollout",
        alignment_policy="token-index-v1",
    )

    assert result.matched_trial_ids == ("same-atomic-phone",)
    torch.testing.assert_close(result.score, torch.tensor(1.0))


def test_interaction_splits_whitespace_delimited_string_like_other_modules() -> None:
    with pytest.raises(ValueError, match="negative pronunciation"):
        AnchorLocalValidity(
            anchor_pronunciation="S H",
            negative_pronunciation=("S", "H"),
            control_pronunciation="K",
        )
