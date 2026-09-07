"""Training objectives for brain-conditioned generation."""

from brain_evidence.objectives.opsd import forward_kl
from brain_evidence.objectives.rewards import (
    GroupRelativeRewards,
    dense_box_reward,
    group_relative_standardize,
    wer_reward,
)

__all__ = [
    "GroupRelativeRewards",
    "dense_box_reward",
    "forward_kl",
    "group_relative_standardize",
    "wer_reward",
]
