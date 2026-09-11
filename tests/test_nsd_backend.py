import numpy as np

import torch
from brain_npp.adapters.nsd_backend import _validate_vad_generation, pending_example, posterior_probabilities, scoring_grad_context
import pytest


def test_completed_evaluation_example_is_not_pending_for_generation():
    row={"image_id":"i","question_id":7}
    assert pending_example(row,set()) is True
    assert pending_example(row,{"i:7"}) is False


def test_raw_zero_brain_is_scored_by_gaussian_instead_of_flattened():
    class Calibration:
        def score(self,record,brain,candidates):
            assert np.array_equal(brain,np.zeros((1,2),dtype=np.float32))
            return np.array([[0.,2.]])
    probabilities=posterior_probabilities(Calibration(),object(),np.zeros((1,2),dtype=np.float32),
                                          np.zeros((2,3)),[-np.log(2),-np.log(2)])
    assert probabilities[1] > .8 and probabilities != [.5,.5]

def test_real_vad_method_name_rejects_warped_sampling_policy():
    with pytest.raises(ValueError,match="unwarped"):
        _validate_vad_generation("vad-style-full-image",{"temperature":.7})
    _validate_vad_generation("vad-style-full-image",{"temperature":1,"top_p":1,"top_k":0})


def test_base_and_snapshot_scoring_are_detached_while_live_tracks_gradients():
    value=torch.tensor(2.,requires_grad=True)
    with scoring_grad_context('base'):base=value.square()
    with scoring_grad_context('snapshot'):snapshot=value.square()
    with scoring_grad_context('live'):live=value.square()
    assert base.grad_fn is None and snapshot.grad_fn is None and live.grad_fn is not None
