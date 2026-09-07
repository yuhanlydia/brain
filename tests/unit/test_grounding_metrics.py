import pytest

from brain_evidence.metrics.grounding import box_iou, parse_box


def test_parse_box_accepts_bracketed_and_shikra_coordinates() -> None:
    assert parse_box("[0, 0, 2, 3]") == (0.0, 0.0, 2.0, 3.0)
    assert parse_box("<box>(1.5, 2), (4, 8)</box>") == (
        1.5,
        2.0,
        4.0,
        8.0,
    )


@pytest.mark.parametrize(
    "value",
    [
        "not a box",
        "[0, 0, 1]",
        "[0, 0, 2, 2] trailing text",
        "[0, 0, 2, 2)",
        "[0, 0, inf, 2]",
        "[0, 0, 0, 2]",
        "[2, 0, 1, 2]",
    ],
)
def test_parse_box_rejects_malformed_or_degenerate_coordinates(value: str) -> None:
    with pytest.raises(ValueError):
        parse_box(value)


def test_box_iou_uses_intersection_over_union() -> None:
    # The intersection is 1 and the union is 4 + 4 - 1 = 7.
    assert box_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1.0 / 7.0)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == pytest.approx(0.0)


def test_box_iou_rejects_degenerate_input_boxes() -> None:
    with pytest.raises(ValueError):
        box_iou((0, 0, 0, 1), (0, 0, 1, 1))
