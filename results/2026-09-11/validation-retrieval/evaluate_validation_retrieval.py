"""Validation-only Gaussian retrieval diagnostics; no hidden-test evaluation."""
import json
from pathlib import Path
import collections
import h5py
import numpy as np
import torch

from brain_npp.calibration import GaussianEncodingLikelihood
from brain_npp.controls import build_block_derangement
from brain_npp.data import load_jsonl_manifest
from brain_npp.metrics import brier_score, expected_calibration_error, cluster_bootstrap_difference
from brain_npp.provenance import sha256_file

root=Path('/root/brain-assets/nsd-full')
records=load_jsonl_manifest(root/'nsd_manifest.jsonl')
image_ids=json.loads((root/'features/image_ids.json').read_text())
image_lookup={key:i for i,key in enumerate(image_ids)}
features=np.load(root/'features/image_pooled.npy',mmap_mode='r')
out=root/'validation-retrieval';out.mkdir(exist_ok=True)
reports={}
for subject in ['subj01','subj02','subj05','subj07']:
    selected=[r for r in records if r.subject_id==subject and r.split=='val']
    gallery=sorted(set(r.image_id for r in selected))
    glookup={key:i for i,key in enumerate(gallery)}
    targets=np.array([glookup[r.image_id] for r in selected])
    x=np.asarray(features[[image_lookup[key] for key in gallery]])
    model_path=root/f'calibration/{subject}/final.npz'
    model=GaussianEncodingLikelihood.load(model_path)
    assert not set(gallery)&set(model.provenance['train_image_ids'])
    indices=[int(r.trial_id.rsplit('trial',1)[1]) for r in selected]
    with h5py.File(root.parent/f'mindeyev2-inspection/betas_all_{subject}_fp32_renorm.hdf5') as h:
        brain=np.asarray(h['betas'])[indices]
    # Cache image means once. Whitening gives the same Mahalanobis scores
    # without repeatedly projecting the full candidate bank per brain batch.
    means=x@model.coefficients+model.intercept
    cholesky=np.linalg.cholesky(model.covariance)
    white_means=np.linalg.solve(cholesky,means.T).T
    white_brain=np.linalg.solve(cholesky,(brain@model.projection).T).T
    distances=(white_brain**2).sum(1)[:,None]+(white_means**2).sum(1)[None,:]-2*white_brain@white_means.T
    scores=-0.5*(distances+2*np.log(np.diag(cholesky)).sum()+means.shape[1]*np.log(2*np.pi))
    assert np.allclose(scores[:3],model.score(brain[:3],x),rtol=1e-10,atol=1e-10)
    scores-=scores.max(1,keepdims=True)
    logp=scores-np.log(np.exp(scores).sum(1,keepdims=True))
    probabilities=np.exp(logp)
    correct=(scores.argmax(1)==targets).astype(float)
    top5=np.take_along_axis(scores,targets[:,None],axis=1)[:,0]>=np.partition(scores,-5,axis=1)[:,-5]
    mapping,unmatched=build_block_derangement(selected,seed=41,preserve_run=True,adjacency_radius=2)
    lookup={r.trial_id:i for i,r in enumerate(selected)}
    source=np.array([lookup[k] for k in sorted(mapping)])
    replacement=np.array([lookup[mapping[k]] for k in sorted(mapping)])
    for a,b in zip(source,replacement):
        r,s=selected[a],selected[b]
        assert r.image_id!=s.image_id and r.session_id==s.session_id and r.run_id==s.run_id
        assert abs(r.trial_index-s.trial_index)>2
    shuffled=(scores[replacement].argmax(1)==targets[source]).astype(float)
    delta,draws=cluster_bootstrap_difference(correct[source],shuffled,[selected[i].image_id for i in source],seed=41,samples=1000)
    repeat_groups=collections.defaultdict(list)
    for i,r in enumerate(selected):repeat_groups[r.image_id].append(i)
    repeats={}
    for count in [1,2,3]:
        averages=np.stack([white_brain[repeat_groups[key][:count]].mean(0) for key in gallery])
        ds=(averages**2).sum(1)[:,None]+(white_means**2).sum(1)[None,:]-2*averages@white_means.T
        ranked=np.argsort(ds,axis=1)
        repeats[str(count)]={'r_at_1':float((ranked[:,0]==np.arange(len(gallery))).mean()),
                             'r_at_5':float((ranked[:,:5]==np.arange(len(gallery))[:,None]).any(1).mean()),
                             'note':'ranking only; single-trial covariance is not claimed calibrated for repeat means'}
    report={'purpose':'validation-only closed-gallery retrieval; not VQA/task accuracy',
        'subject':subject,'trials':len(selected),'gallery_images':len(gallery),'gallery_policy':'all subject validation images, fixed before scoring',
        'model_sha256':sha256_file(model_path),'r_at_1':float(correct.mean()),'r_at_5':float(top5.mean()),
        'candidate_nll':float(-logp[np.arange(len(targets)),targets].mean()),
        'brier':brier_score(probabilities,targets),'ece_10bins':expected_calibration_error(probabilities,targets),
        'entropy':float(-(probabilities*logp).sum(1).mean()),'uniform_nll':float(np.log(len(gallery))),
        'matched_control_trials':len(source),'unmatched_trial_ids':list(unmatched),
        'matched_correct_r1':float(correct[source].mean()),'shuffled_r1':float(shuffled.mean()),
        'r1_difference':delta,'image_bootstrap_95ci':np.quantile(draws,[.025,.975]).tolist(),
        'repeat_rankings':repeats,'hidden_test_used':False}
    (out/f'{subject}.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/f'{subject}.shuffle.json').write_text(json.dumps({'seed':41,'adjacency_radius':2,'mapping':mapping,'unmatched':list(unmatched)},indent=2)+'\n')
    np.savez(out/f'{subject}.scores.npz',trial_ids=np.array([r.trial_id for r in selected]),image_ids=np.array(gallery),log_probabilities=logp,targets=targets)
    reports[subject]=report
    print(subject,{k:report[k] for k in ['r_at_1','r_at_5','candidate_nll','uniform_nll','r1_difference']},flush=True)
(out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
