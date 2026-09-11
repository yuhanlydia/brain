import math

import pytest
import torch

from brain_npp.baselines import (
    DAPD_PAIRS,
    credit_image_contrastive_loss,
    dapd_loss,
    full_vocabulary_teacher_target,
    masked_cross_entropy,
    vad_budgeted_asvc_loss,
)


def _p(values, *, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def test_full_vocabulary_targets_are_arithmetic_and_ce_is_separate():
    teachers = _p([[[[0.8, 0.2], [0.2, 0.8]]]])
    weights = _p([[0.75, 0.25]])
    assert torch.allclose(full_vocabulary_teacher_target(teachers, mode="exact"), _p([[[0.8, 0.2]]]))
    assert torch.allclose(full_vocabulary_teacher_target(teachers, mode="map", weights=weights), _p([[[0.8, 0.2]]]))
    assert torch.allclose(full_vocabulary_teacher_target(teachers, mode="uniform"), _p([[[0.5, 0.5]]]))
    assert torch.allclose(full_vocabulary_teacher_target(teachers, mode="posterior", weights=weights), _p([[[0.65, 0.35]]]))

    logits = _p([[[math.log(0.25), math.log(0.75)]]], grad=True)
    loss = masked_cross_entropy(logits, torch.tensor([[0]]), torch.tensor([[True]]))
    assert torch.allclose(loss, _p(-math.log(0.25)))
    loss.backward()
    assert torch.allclose(logits.grad, _p([[[-0.75, 0.75]]]))


def test_targets_detach_teachers_and_are_candidate_permutation_invariant():
    teachers = _p([[[[0.7, 0.3], [0.1, 0.9]]]], grad=True)
    weights = _p([[0.2, 0.8]], grad=True)
    got = full_vocabulary_teacher_target(teachers, mode="posterior", weights=weights)
    permuted = full_vocabulary_teacher_target(teachers[:, :, [1, 0]], mode="posterior", weights=weights[:, [1, 0]])
    assert not got.requires_grad
    assert torch.allclose(got, permuted)


def test_credit_uses_full_vocab_reverse_kl_and_detaches_both_teachers():
    student = _p([[[math.log(0.6), math.log(0.4)]]], grad=True)
    positive = _p([[[0.8, 0.2]]], grad=True)
    negative = _p([[[0.25, 0.75]]], grad=True)
    loss = credit_image_contrastive_loss(student, positive, negative, torch.tensor([[True]]), negative_power=0.1)
    q_raw = _p([0.8 / 0.25**0.1, 0.2 / 0.75**0.1])
    q = q_raw / q_raw.sum()
    expected_rkl = (torch.tensor([0.6, 0.4], dtype=torch.float64) * (torch.log(torch.tensor([0.6, 0.4], dtype=torch.float64)) - q.log())).sum()
    forward = (q * (q.log() - torch.log(torch.tensor([0.6, 0.4], dtype=torch.float64)))).sum()
    assert torch.allclose(loss, expected_rkl)
    assert not torch.allclose(loss, forward)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert positive.grad is None and negative.grad is None


def test_credit_mask_is_authoritative_and_empty_mask_is_rejected():
    s = _p([[[0.0, 0.0], [float("nan"), 0.0]]], grad=True)
    t = _p([[[0.5, 0.5], [float("nan"), 0.5]]])
    mask = torch.tensor([[True, False]])
    assert torch.isfinite(credit_image_contrastive_loss(s, t, t.clone(), mask))
    with pytest.raises(ValueError, match="empty"):
        credit_image_contrastive_loss(s[:, :1], t[:, :1], t[:, :1], torch.tensor([[False]]))


def test_vad_topk_tail_preserves_mass_and_identical_teachers_are_identity():
    probs = _p([[[0.40, 0.30, 0.20, 0.10]]])
    logits = probs.log().detach().requires_grad_()
    result = vad_budgeted_asvc_loss(logits, probs, probs, torch.tensor([[True]]), top_k=2)
    assert torch.allclose(result.student_support, _p([[[0.4, 0.3, 0.3]]]))
    assert torch.allclose(result.target, result.student_support.detach())
    assert torch.allclose(result.rho, _p([[0.0]]))
    assert torch.allclose(result.loss, _p(0.0), atol=1e-14)


def test_vad_signed_budget_matches_independent_reference_and_importance_cap():
    student_p = _p([[[0.5, 0.3, 0.2]]])
    clear = _p([[[0.6, 0.1, 0.3]]], grad=True)
    degraded = _p([[[0.2, 0.5, 0.3]]], grad=True)
    logits = student_p.log().detach().requires_grad_()
    result = vad_budgeted_asvc_loss(
        logits, clear, degraded, torch.tensor([[True]]), top_k=3,
        sampled_log_probs=_p([[5.0]]), old_log_probs=_p([[0.0]]), anchor_eta=0.0,
    )
    center = lambda x: x - x.mean()
    ps, pp, pm = map(center, (student_p[0, 0].log(), clear[0, 0].log(), degraded[0, 0].log()))
    r, u = pp - ps, pp - pm
    budget = torch.clamp(torch.dot(r, u), min=0) / (torch.dot(u, u) + 1e-3) * u.norm()
    up, un = u.clamp(min=0), u.clamp(max=0)
    sp, sn = torch.clamp(torch.dot(r, up), min=0), torch.clamp(torch.dot(r, un), min=0)
    pi_p, pi_n = sp / (sp + sn + 1e-8), sn / (sp + sn + 1e-8)
    shift = budget * (torch.minimum(pi_p, _p(0.8)) * up / (up.norm() + 1e-8) + pi_n * un / (un.norm() + 1e-8))
    expected_q = torch.softmax(ps + shift.clamp(-20, 20), dim=-1)
    assert torch.allclose(result.target[0, 0], expected_q)
    assert torch.allclose(result.importance, _p([[2.0]]))
    result.loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert clear.grad is None and degraded.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vad_extreme_finite_logits_have_finite_loss_and_gradients(dtype):
    logits = torch.tensor([[[1000.0, -1000.0, -2000.0]]], dtype=dtype, requires_grad=True)
    teacher = torch.tensor([[[0.6, 0.3, 0.1]]], dtype=dtype)
    result = vad_budgeted_asvc_loss(logits, teacher, teacher, torch.tensor([[True]]), top_k=3)
    assert torch.isfinite(result.target).all()
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_teacher_and_importance_dtypes_must_match_student_logits():
    logits = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    teacher64 = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
    mask = torch.tensor([[True]])
    with pytest.raises(ValueError, match="dtype"):
        vad_budgeted_asvc_loss(logits, teacher64, teacher64, mask)
    teacher = teacher64.float()
    with pytest.raises(ValueError, match="floating-point.*dtype"):
        vad_budgeted_asvc_loss(
            logits, teacher, teacher, mask,
            sampled_log_probs=torch.tensor([[0]]), old_log_probs=torch.tensor([[0]]),
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scalar_hyperparameters_are_rejected(bad):
    logits = _p([[[0.0, 0.0]]])
    teacher = _p([[[0.5, 0.5]]])
    mask = torch.tensor([[True]])
    candidates = teacher.unsqueeze(2)
    with pytest.raises(ValueError, match="finite"):
        full_vocabulary_teacher_target(candidates, mode="exact", exact_index=bad)
    with pytest.raises(ValueError, match="finite"):
        credit_image_contrastive_loss(logits, teacher, teacher, mask, negative_power=bad)
    with pytest.raises(ValueError, match="finite"):
        vad_budgeted_asvc_loss(logits, teacher, teacher, mask, element_clip=bad)
    live, anchors, masks = _dapd_fixture()
    with pytest.raises(ValueError, match="finite"):
        dapd_loss(live, anchors, masks, temperature=bad)


def _dapd_fixture():
    live = {
        "rollout_none": _p([[[0.2, -0.1]], [[0.3, 0.4]]], grad=True),
        "reference_reference": _p([[[0.0, 0.5]], [[-0.2, 0.7]]], grad=True),
        "rollout_rollout": _p([[[0.4, -0.3]], [[0.1, 0.2]]], grad=True),
        "reference_none": _p([[[-0.1, 0.3]], [[0.6, -0.4]]], grad=True),
    }
    anchors = {name: _p([[[0.7, -0.2]], [[-0.3, 0.8]]], grad=True) for name in DAPD_PAIRS}
    masks = {"rollout": torch.tensor([[True], [False]]), "reference": torch.tensor([[True], [True]])}
    return live, anchors, masks


def test_dapd_all_six_terms_match_manual_clipped_calculation_and_detach():
    live, anchors, masks = _dapd_fixture()
    result = dapd_loss(live, anchors, masks)
    live_key = {
        "entangled_rollout": "rollout_none", "inference_reference": "reference_reference",
        "privileged_rollout": "rollout_rollout", "entangled_reference": "reference_none",
        "inference_rollout": "rollout_rollout", "privileged_reference": "reference_reference",
    }
    weights = dict(zip(DAPD_PAIRS, [2/15, 2/15, 2/15, 2/5, 2/5, 4/5]))
    completion = dict(zip(DAPD_PAIRS, ["rollout", "reference", "rollout", "reference", "rollout", "reference"]))
    expected = _p(0.0)
    for name in DAPD_PAIRS:
        q = torch.softmax(anchors[name].detach() / 1.1, -1)
        logq = torch.log_softmax(anchors[name].detach() / 1.1, -1)
        logp = torch.log_softmax(live[live_key[name]] / 1.1, -1)
        token = (q * (logq - logp)).clamp(max=0.05).sum(-1)
        expected = expected + weights[name] * token[masks[completion[name]]].mean()
    assert torch.allclose(result.loss, expected)
    assert set(result.terms) == set(DAPD_PAIRS)
    result.loss.backward()
    assert all(x.grad is None for x in anchors.values())
    assert all(x.grad is not None for x in live.values())


def test_dapd_component_clip_is_not_final_kl_clip_and_masks_can_differ():
    live, anchors, masks = _dapd_fixture()
    result = dapd_loss(live, anchors, masks, component_clip=0.001)
    q = torch.softmax(anchors["entangled_rollout"][0, 0].detach() / 1.1, -1)
    logp = torch.log_softmax(live["rollout_none"][0, 0] / 1.1, -1)
    final_kl_clipped = torch.clamp((q * (q.log() - logp)).sum(), max=0.001)
    assert not torch.allclose(result.terms["entangled_rollout"], final_kl_clipped)
    swapped = {"rollout": torch.tensor([[False], [True]]), "reference": masks["reference"]}
    assert not torch.allclose(result.loss, dapd_loss(live, anchors, swapped, component_clip=0.001).loss)


@pytest.mark.parametrize("which", ["shape", "probability", "nonfinite"])
def test_baselines_reject_invalid_active_inputs(which):
    logits = _p([[[0.0, 0.0]]])
    probs = _p([[[0.5, 0.5]]])
    mask = torch.tensor([[True]])
    if which == "shape":
        with pytest.raises(ValueError, match="shape"):
            credit_image_contrastive_loss(logits, probs[:, :, :1], probs, mask)
    elif which == "probability":
        with pytest.raises(ValueError, match="probabil"):
            vad_budgeted_asvc_loss(logits, _p([[[0.8, 0.8]]]), probs, mask)
    else:
        with pytest.raises(ValueError, match="finite"):
            credit_image_contrastive_loss(_p([[[float("nan"), 0.0]]]), probs, probs, mask)
