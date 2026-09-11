import torch
import pytest

from brain_npp.adapters.nsd_vindex import PatchCache, ParameterContexts, causal_completion_batch


def test_parameter_contexts_use_original_projector_for_base_and_full_snapshot():
    model = torch.nn.Linear(2, 2, bias=False); projector = torch.nn.Linear(2, 2, bias=False)
    named = {"lora.weight": model.weight, "projector.weight": projector.weight}
    contexts = ParameterContexts(named, projector_names={"projector.weight"}, adapter_context=lambda: _Disable(model))
    original_projector = projector.weight.detach().clone()
    contexts.capture_snapshot(update_index=0)
    with torch.no_grad(): model.weight.add_(2); projector.weight.add_(3)
    live = {k:v.detach().clone() for k,v in named.items()}
    with contexts.snapshot() as update:
        assert update == 0 and torch.equal(model.weight, contexts.snapshot_state.trainable["lora.weight"])
        assert torch.equal(projector.weight, contexts.snapshot_state.trainable["projector.weight"])
    assert all(torch.equal(named[k], live[k]) for k in named)
    with contexts.base():
        assert model.disabled and torch.equal(projector.weight, original_projector)
    assert not model.disabled and all(torch.equal(named[k], live[k]) for k in named)


class _Disable:
    def __init__(self, model): self.model=model
    def __enter__(self): self.model.disabled=True
    def __exit__(self, *args): self.model.disabled=False


def test_causal_completion_batch_aligns_independent_prefixes_and_masks_padding():
    embeddings = torch.nn.Embedding(20, 3)
    prefixes = [torch.ones(2,3), torch.ones(4,3)*2]
    tokens = [torch.tensor([5,6]), torch.tensor([7])]
    batch = causal_completion_batch(prefixes, tokens, embeddings, pad_token_id=0)
    assert batch.inputs_embeds.shape == (2, 5, 3)
    assert batch.attention_mask.tolist() == [[1,1,1,1,0],[1,1,1,1,1]]
    assert batch.logit_indices.tolist() == [[1,2],[3,-1]]
    assert batch.completion_mask.tolist() == [[True,True],[True,False]]


def test_patch_cache_binds_image_model_preprocessing_and_clear_degraded(tmp_path):
    cache=PatchCache(tmp_path,max_bytes=10000,model_hash="m",preprocessing_hash="p")
    clear=cache.get_or_compute("i","clear","ih",lambda:torch.ones(1,4,3))
    degraded=cache.get_or_compute("i","degraded","ih",lambda:torch.zeros(1,4,3))
    assert clear.sum()==12 and degraded.sum()==0
    assert cache.get_or_compute("i","clear","ih",lambda:(_ for _ in ()).throw(RuntimeError())) .sum()==12
    with pytest.raises(ValueError,match="image hash"):
        cache.get_or_compute("i","clear","changed",lambda:torch.ones(1,4,3))

def test_patch_cache_recovers_corruption_and_incomplete_pair(tmp_path):
    cache=PatchCache(tmp_path,max_bytes=10000,model_hash="m",preprocessing_hash="p")
    cache.get_or_compute("a","clear","h",lambda:torch.ones(1,4,3))
    tensor,meta=cache._paths("a","clear");tensor.write_bytes(b"broken")
    assert cache.get_or_compute("a","clear","h",lambda:torch.full((1,4,3),2)).sum()==24
    tensor.unlink();assert cache.get_or_compute("a","clear","h",lambda:torch.full((1,4,3),3)).sum()==36

def test_patch_cache_evicts_old_entries_under_serialized_byte_bound(tmp_path):
    cache=PatchCache(tmp_path,max_bytes=2000,model_hash="m",preprocessing_hash="p");calls=[]
    for name in ("a","b","a"):cache.get_or_compute(name,"clear",name,lambda n=name:(calls.append(n) or torch.ones(1,4,3)))
    assert calls==["a","b","a"]
