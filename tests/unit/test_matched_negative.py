from __future__ import annotations

import pytest

from brain_evidence.controls.matched_negative import (
    TrialMetadata,
    sample_anchor_local_controls,
    sample_matched_negatives,
)


def _trial(
    trial_id: str,
    *,
    subject: str = "subject-1",
    session: str = "session-1",
    split: str = "train",
    fold_id: str = "fold-1",
    pronunciation: object,
    duration: float = 1.0,
    reliability_stratum: str = "high",
) -> TrialMetadata:
    return TrialMetadata(
        trial_id=trial_id,
        subject_id=subject,
        session_id=session,
        split=split,
        fold_id=fold_id,
        pronunciation=pronunciation,
        duration=duration,
        reliability_stratum=reliability_stratum,
    )


@pytest.mark.parametrize("field", ["split", "fold_id"])
@pytest.mark.parametrize("bad_value", ["   ", " partition-1", "partition-1 "])
def test_trial_metadata_requires_canonical_partition_provenance(
    field: str,
    bad_value: str,
) -> None:
    values = {"split": "train", "fold_id": "fold-1"}
    values[field] = bad_value

    with pytest.raises(ValueError, match=field):
        _trial("trial-1", pronunciation=("SH",), **values)


def test_matched_negative_does_not_cross_split_or_fold() -> None:
    trials = [
        _trial("anchor", pronunciation=("SH",)),
        _trial(
            "wrong-split",
            pronunciation=("K",),
            split="validation",
        ),
        _trial(
            "wrong-fold",
            pronunciation=("T",),
            fold_id="fold-2",
        ),
        _trial("valid", pronunciation=("N",)),
    ]

    result = sample_matched_negatives(
        trials,
        anchor_index=0,
        k=3,
        duration_tolerance=0.0,
        seed=41,
    )

    assert result.indices == (3,)
    assert result.eligible_count == 1
    assert result.shortfall == 2


def test_anchor_local_controls_do_not_cross_split_or_fold() -> None:
    trials = [
        _trial("anchor", pronunciation=("SH",)),
        _trial("valid-negative", pronunciation=("K",)),
        _trial("valid-control", pronunciation=("T",)),
        _trial(
            "wrong-split",
            pronunciation=("N",),
            split="validation",
        ),
        _trial(
            "wrong-fold",
            pronunciation=("D",),
            fold_id="fold-2",
        ),
    ]

    result = sample_anchor_local_controls(
        trials,
        anchor_index=0,
        k=4,
        duration_tolerance=0.0,
        seed=41,
    )

    assert result.effective_k == 2
    assert result.shortfall == 2
    assert {pair.negative.trial_id for pair in result.pairs} == {
        "valid-negative",
        "valid-control",
    }
    for pair in result.pairs:
        assert pair.negative.split == pair.anchor.split
        assert pair.negative.fold_id == pair.anchor.fold_id
        assert pair.control_brain.split == pair.anchor.split
        assert pair.control_brain.fold_id == pair.anchor.fold_id


@pytest.mark.parametrize(
    "invalid_pronunciation",
    ["", "   ", (), ("SH", ""), ("SH", " "), ("SH", 7), 7, None],
)
def test_trial_metadata_rejects_malformed_pronunciations(
    invalid_pronunciation: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="pronunciation"):
        trials = [
            _trial("anchor", pronunciation=("SH",)),
            _trial("malformed", pronunciation=invalid_pronunciation),
        ]
        sample_matched_negatives(
            trials,
            anchor_index=0,
            k=1,
            duration_tolerance=0.0,
            seed=41,
        )


@pytest.mark.parametrize(
    "field",
    ["trial_id", "subject_id", "session_id", "reliability_stratum"],
)
@pytest.mark.parametrize("bad_value", ["   ", " identity-1", "identity-1 "])
def test_trial_metadata_rejects_noncanonical_matching_provenance(
    field: str,
    bad_value: str,
) -> None:
    values: dict[str, object] = {
        "trial_id": "trial-1",
        "subject_id": "subject-1",
        "session_id": "session-1",
        "split": "train",
        "fold_id": "fold-1",
        "pronunciation": ("SH",),
        "duration": 1.0,
        "reliability_stratum": "high",
    }
    values[field] = bad_value

    with pytest.raises(ValueError, match=field):
        TrialMetadata(**values)


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -float("inf")])
def test_trial_metadata_requires_finite_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        _trial("trial-1", pronunciation=("SH",), duration=duration)


def test_sampler_rejects_duplicate_trial_provenance() -> None:
    trials = [
        _trial("duplicate", pronunciation=("SH",)),
        _trial("duplicate", pronunciation=("K",)),
    ]

    with pytest.raises(ValueError, match="duplicate trial_id"):
        sample_matched_negatives(
            trials,
            anchor_index=0,
            k=1,
            duration_tolerance=0.0,
            seed=41,
        )


def test_seeded_sampling_is_deterministic_distinct_and_constraint_preserving() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("valid-1", pronunciation=("K", "AE", "P"), duration=1.1),
        _trial("valid-2", pronunciation=("B", "AE", "T"), duration=0.9),
        _trial("valid-3", pronunciation=("D", "AO", "G"), duration=1.2),
        _trial("same-pronunciation", pronunciation=("K", "AE", "T")),
        _trial(
            "wrong-subject",
            subject="subject-2",
            pronunciation=("F", "IH", "SH"),
        ),
        _trial(
            "wrong-session",
            session="session-2",
            pronunciation=("F", "IH", "SH"),
        ),
        _trial("too-long", pronunciation=("F", "IH", "SH"), duration=1.31),
    ]

    first = sample_matched_negatives(
        trials,
        anchor_index=0,
        k=3,
        duration_tolerance=0.3,
        seed=41,
    )
    second = sample_matched_negatives(
        trials,
        anchor_index=0,
        k=3,
        duration_tolerance=0.3,
        seed=41,
    )

    assert first == second
    assert len(first.indices) == 3
    assert len(set(first.indices)) == 3
    assert {trials[index].trial_id for index in first.indices} == {
        "valid-1",
        "valid-2",
        "valid-3",
    }
    assert first.shortfall == 0
    assert first.complete
    assert first.valid_fraction == 1.0


def test_hard_negative_mode_prefers_closest_phonetic_sequence() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("far", pronunciation=("D", "AO", "G")),
        _trial("closest", pronunciation=("K", "AE", "P")),
        _trial("middle", pronunciation=("K", "IH", "P")),
    ]

    result = sample_matched_negatives(
        trials,
        anchor_index=0,
        k=2,
        duration_tolerance=0.0,
        seed=7,
        prefer_phonetic_hard_negatives=True,
    )

    assert [trials[index].trial_id for index in result.indices] == [
        "closest",
        "middle",
    ]


def test_sampler_reports_shortfall_without_crossing_constraints() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("only-valid", pronunciation=("K", "AE", "P")),
        _trial(
            "wrong-session",
            session="session-2",
            pronunciation=("D", "AO", "G"),
        ),
    ]

    result = sample_matched_negatives(
        trials,
        anchor_index=0,
        k=3,
        duration_tolerance=0.1,
        seed=11,
    )

    assert result.indices == (1,)
    assert result.eligible_count == 1
    assert result.shortfall == 2
    assert not result.complete
    assert result.valid_fraction == 1 / 3


def test_matched_negative_requires_anchor_reliability_stratum() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial(
            "wrong-stratum",
            pronunciation=("K", "AE", "P"),
            reliability_stratum="low",
        ),
        _trial("valid", pronunciation=("D", "AO", "G")),
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
    assert result.shortfall == 1


def test_anchor_local_pairs_are_distinct_and_provenanced() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("negative-b-1", pronunciation=("K", "AE", "P"), duration=1.05),
        _trial("negative-b-2", pronunciation=("K", "AE", "P"), duration=0.95),
        _trial("negative-c", pronunciation=("D", "AO", "G")),
        _trial("negative-d", pronunciation=("F", "IH", "SH")),
    ]

    first = sample_anchor_local_controls(
        trials,
        anchor_index=0,
        k=3,
        duration_tolerance=0.1,
        seed=43,
    )
    second = sample_anchor_local_controls(
        trials,
        anchor_index=0,
        k=3,
        duration_tolerance=0.1,
        seed=43,
    )

    assert first == second
    assert first.effective_k == 3
    assert first.complete
    assert not first.should_skip
    assert len({pair.negative.pronunciation for pair in first.pairs}) == 3
    for pair in first.pairs:
        assert pair.anchor == trials[pair.anchor_index]
        assert pair.negative == trials[pair.negative_index]
        assert pair.control_brain == trials[pair.control_brain_index]
        assert pair.negative.pronunciation != pair.anchor.pronunciation
        assert pair.control_brain.pronunciation not in {
            pair.anchor.pronunciation,
            pair.negative.pronunciation,
        }
        assert pair.negative.subject_id == pair.anchor.subject_id
        assert pair.control_brain.subject_id == pair.anchor.subject_id
        assert pair.negative.session_id == pair.anchor.session_id
        assert pair.control_brain.session_id == pair.anchor.session_id
        assert pair.control_brain.reliability_stratum == pair.anchor.reliability_stratum


def test_confirmatory_shortfall_marks_anchor_to_skip() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("negative-b-1", pronunciation=("K", "AE", "P")),
        _trial("negative-b-2", pronunciation=("K", "AE", "P")),
        _trial("negative-c", pronunciation=("D", "AO", "G")),
    ]

    result = sample_anchor_local_controls(
        trials,
        anchor_index=0,
        k=4,
        duration_tolerance=0.0,
        seed=41,
        confirmatory=True,
    )

    assert result.effective_k == 2
    assert result.eligible_pronunciation_count == 2
    assert result.shortfall == 2
    assert not result.complete
    assert result.should_skip
    assert len({pair.negative.pronunciation for pair in result.pairs}) == 2


def test_anchor_local_sampler_does_not_use_an_unmatched_control_brain() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("negative", pronunciation=("K", "AE", "P")),
        _trial("same-negative", pronunciation=("K", "AE", "P")),
        _trial("too-long-control", pronunciation=("D", "AO", "G"), duration=1.2),
    ]

    result = sample_anchor_local_controls(
        trials,
        anchor_index=0,
        k=4,
        duration_tolerance=0.1,
        seed=7,
        confirmatory=True,
    )

    assert result.pairs == ()
    assert result.shortfall == 4
    assert result.should_skip


def test_anchor_local_negative_requires_anchor_reliability_stratum() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial(
            "wrong-stratum-negative",
            pronunciation=("K", "AE", "P"),
            reliability_stratum="low",
        ),
        _trial("valid-control", pronunciation=("F", "IH", "SH")),
    ]

    result = sample_anchor_local_controls(
        trials,
        anchor_index=0,
        k=1,
        duration_tolerance=0.0,
        seed=7,
        prefer_phonetic_hard_negatives=True,
    )

    assert result.pairs == ()
    assert result.eligible_pronunciation_count == 0
    assert result.shortfall == 1


def test_anchor_local_control_requires_anchor_reliability_stratum() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("negative", pronunciation=("K", "AE", "P")),
        _trial(
            "wrong-stratum-control",
            pronunciation=("D", "AO", "G"),
            reliability_stratum="low",
        ),
        _trial("valid-control", pronunciation=("F", "IH", "SH")),
    ]

    result = sample_anchor_local_controls(
        trials,
        anchor_index=0,
        k=1,
        duration_tolerance=0.0,
        seed=7,
        prefer_phonetic_hard_negatives=True,
    )

    assert result.pairs[0].negative.trial_id == "negative"
    assert result.pairs[0].control_brain.trial_id == "valid-control"
    assert result.pairs[0].negative.reliability_stratum == "high"
    assert result.pairs[0].control_brain.reliability_stratum == "high"


def test_confirmatory_sampling_requires_at_least_four_negatives() -> None:
    trials = [
        _trial("anchor", pronunciation=("K", "AE", "T")),
        _trial("negative", pronunciation=("K", "AE", "P")),
        _trial("control", pronunciation=("D", "AO", "G")),
    ]

    with pytest.raises(ValueError, match="at least 4"):
        sample_anchor_local_controls(
            trials,
            anchor_index=0,
            k=3,
            duration_tolerance=0.0,
            seed=41,
            confirmatory=True,
        )
