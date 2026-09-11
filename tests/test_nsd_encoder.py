from __future__ import annotations

import pytest
import torch

from brain_npp.nsd_encoder import encode_subject_brain, restricted_scheduler_getattr, source_provenance


class FakeBrainX(torch.nn.Module):
    num_voxels = {1: 3, 2: 2, 5: 4, 7: 5}

    def __init__(self):
        super().__init__()
        self.modal = None

    def forward(self, brain, modal):
        self.modal = modal
        return brain.sum(1, keepdim=True)


def test_subject_routing_uses_own_branch_and_rejects_padding_or_crop():
    model = FakeBrainX()
    assert encode_subject_brain(model, torch.ones(2, 2), "subj02").shape == (2, 1)
    assert model.modal == "fmri2"
    with pytest.raises(ValueError, match="expects 2 voxels"):
        encode_subject_brain(model, torch.ones(2, 3), "subj02")


def test_safe_getattr_allows_only_onecycle_cosine_scheduler_helper():
    from torch.optim.lr_scheduler import OneCycleLR

    assert callable(restricted_scheduler_getattr(OneCycleLR, "_annealing_cos"))
    with pytest.raises(ValueError, match="refused"):
        restricted_scheduler_getattr(OneCycleLR, "__dict__")
    with pytest.raises(ValueError, match="refused"):
        restricted_scheduler_getattr(object, "_annealing_cos")


def test_source_provenance_hashes_model_and_perceiver_dependency(tmp_path):
    model = tmp_path / "model.py"; perceiver = tmp_path / "perceiver.py"
    model.write_text("from perceiver import Perceiver\n")
    perceiver.write_text("class Perceiver: pass\n")
    got = source_provenance(model)
    assert got == {"model.py": __import__("hashlib").sha256(model.read_bytes()).hexdigest(),
                   "perceiver.py": __import__("hashlib").sha256(perceiver.read_bytes()).hexdigest()}
