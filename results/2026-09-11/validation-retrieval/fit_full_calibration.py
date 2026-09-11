"""Fit training-only, image-cross-fitted Gaussian likelihoods for four subjects."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from brain_npp.calibration import GaussianEncodingLikelihood
from brain_npp.data import load_jsonl_manifest, assign_grouped_folds
from brain_npp.provenance import sha256_file

assets=Path('/root/brain-assets')
root=assets/'nsd-full'
records=load_jsonl_manifest(root/'nsd_manifest.jsonl')
features=np.load(root/'features/image_pooled.npy',mmap_mode='r',allow_pickle=False)
image_ids=json.loads((root/'features/image_ids.json').read_text())
image_lookup={image_id:i for i,image_id in enumerate(image_ids)}
feature_metadata=json.loads((root/'features/clip_provenance.json').read_text())
assert feature_metadata['completed_images']==len(image_ids)==37000
assert feature_metadata['feature_sha256']==sha256_file(root/'features/image_pooled.npy')
manifest_hash=sha256_file(root/'nsd_manifest.jsonl')
training=[record for record in records if record.split=='train']
global_folds=assign_grouped_folds(training,folds=5,seed=1731)
out=root/'calibration';out.mkdir(exist_ok=True)
input_record={'manifest_sha256':manifest_hash,'image_features_sha256':feature_metadata['feature_sha256'],
              'fold_seed':1731,'outer_folds':5,'inner_folds':5,
              'calibration_code_sha256':sha256_file('/root/brain/src/brain_npp/calibration.py')}
for subject in ['subj01','subj02','subj05','subj07']:
    selected=[record for record in training if record.subject_id==subject]
    indices=np.array([int(record.trial_id.rsplit('trial',1)[1]) for record in selected])
    beta_path=assets/f'mindeyev2-inspection/betas_all_{subject}_fp32_renorm.hdf5'
    beta_hash=sha256_file(beta_path)
    with h5py.File(beta_path) as source:
        # Selection occurs before fitting; no validation/test response enters a fit.
        brain=np.asarray(source['betas'])[indices]
    x=np.asarray(features[[image_lookup[record.image_id] for record in selected]])
    ids=np.array([record.image_id for record in selected])
    sample_ids=np.array([record.trial_id for record in selected])
    folds=np.array([global_folds[record.image_id] for record in selected])
    directory=out/subject;directory.mkdir(exist_ok=True)
    identity={**input_record,'subject':subject,'beta_sha256':beta_hash}
    identity_path=directory/'inputs.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('incompatible full calibration input checkpoint')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    scores=np.empty(len(selected))
    artifacts={}
    for fold in list(range(5))+['final']:
        fitted=np.ones(len(selected),dtype=bool) if fold=='final' else folds!=fold
        path=directory/f'{fold}.npz'
        if path.exists():
            model=GaussianEncodingLikelihood.load(path)
            assert model.provenance['train_sample_ids']==sample_ids[fitted].tolist()
            reused=True
        else:
            model=GaussianEncodingLikelihood.fit(x[fitted],brain[fitted],image_ids=ids[fitted],sample_ids=sample_ids[fitted],
                feature_provenance=f"CLIP224 pooler_output; content {feature_metadata['feature_sha256']}",
                brain_provenance=f'{subject}; public renorm single-trial betas; sha256 {beta_hash}; train only',inner_folds=5)
            temporary=path.with_suffix('.tmp.npz');model.save(temporary);temporary.replace(path)
            reused=False
        if fold!='final':
            assert not set(ids[~fitted]) & set(model.provenance['train_image_ids'])
            scores[~fitted]=model.score_paired(brain[~fitted],x[~fitted])
        artifacts[str(fold)]={'path':str(path),'sha256':sha256_file(path),'train_images':len(set(ids[fitted])),
                              'train_trials':int(fitted.sum()),'reused':reused}
        print(subject,fold,'saved',flush=True)
    np.savez(directory/'oof_density.npz',trial_ids=sample_ids,image_ids=ids,fold_ids=folds,paired_log_density=scores)
    report={'subject':subject,'purpose':'full training-only likelihood fitting; no benchmark task scores',
            'model_semantics':model.provenance['model_semantics'],'train_trials':len(selected),'train_images':len(set(ids)),
            'validation_test_used_in_fitting':False,'all_oof_scores_finite':bool(np.isfinite(scores).all()),
            'oof_log_density_mean':float(scores.mean()),'artifacts':artifacts,**identity}
    (directory/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(subject,'COMPLETE',flush=True)
    del brain,x
print('ALL_SUBJECTS_COMPLETE',flush=True)
