import math

import pytest
import torch

from brain_evidence.objectives.rewards import (
    dense_box_reward,
    group_relative_standardize,
    wer_reward,
)


def test_wer_reward_is_negative_normalized_word_error() -> None:
    assert wer_reward("read this", "read that") == pytest.approx(-0.5)


def test_dense_box_reward_adds_iou_and_threshold_bonus() -> None:
    # These boxes have IoU 1/7, which clears the selected 0.1 threshold.
    reward = dense_box_reward(
        "[0, 0, 2, 2]",
        (1, 1, 3, 3),
        iou_threshold=0.1,
        threshold_bonus=0.25,
        invalid_penalty=2.0,
    )
    below_threshold = dense_box_reward(
        "[0, 0, 2, 2]",
        (1, 1, 3, 3),
        iou_threshold=0.5,
        threshold_bonus=0.25,
        invalid_penalty=2.0,
    )

    assert reward == pytest.approx((1.0 / 7.0) + 0.25)
    assert below_threshold == pytest.approx(1.0 / 7.0)


@pytest.mark.parametrize("prediction", ["not a box", "[0, 0, 0, 1]"])
def test_dense_box_reward_penalizes_invalid_predictions(prediction: str) -> None:
    reward = dense_box_reward(
        prediction,
        (0, 0, 1, 1),
        iou_threshold=0.5,
        threshold_bonus=0.25,
        invalid_penalty=2.0,
    )

    assert reward == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("iou_threshold", math.nan),
        ("iou_threshold", math.inf),
        ("threshold_bonus", math.nan),
        ("threshold_bonus", math.inf),
        ("invalid_penalty", math.nan),
        ("invalid_penalty", math.inf),
    ],
)
def test_dense_box_reward_rejects_nonfinite_scalars(
    parameter: str, value: float
) -> None:
    arguments = {
        "iou_threshold": 0.5,
        "threshold_bonus": 0.25,
        "invalid_penalty": 2.0,
    }
    arguments[parameter] = value

    with pytest.raises(ValueError, match="finite"):
        dense_box_reward("[0, 0, 1, 1]", (0, 0, 1, 1), **arguments)


def test_group_relative_standardize_skips_equal_reward_groups() -> None:
    rewards = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = group_relative_standardize(rewards)

    expected_scale = math.sqrt(3.0 / 2.0)
    torch.testing.assert_close(
        result.advantages,
        torch.tensor(
            [[-expected_scale, 0.0, expected_scale], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert result.skipped.tolist() == [False, True]
    assert result.skip_rate == pytest.approx(0.5)
    assert not result.advantages.requires_grad


@pytest.mark.parametrize("equal_tolerance", [math.nan, math.inf, -math.inf])
def test_group_relative_standardize_rejects_nonfinite_tolerance(
    equal_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="equal_tolerance"):
        group_relative_standardize(
            torch.tensor([[1.0, 2.0]]), equal_tolerance=equal_tolerance
        )
