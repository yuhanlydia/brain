"""Cache fixed zero-mean train-covariance controls for the predeclared cohort."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from brain_npp.data import load_jsonl_manifest
from brain_npp.nsd_data import EmpiricalCovarianceNoise


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[41,42,43]);args=p.parse_args()
    torch.set_num_threads(8)
    manifest=args.root/'nsd_manifest.jsonl';cohort_path=args.root/'matrix/evaluation_cohort.json'
    records=load_jsonl_manifest(manifest);cohort=json.loads(cohort_path.read_text());out=args.root/'matrix/noise';out.mkdir(exist_ok=True)
    for subject in ['subj01','subj02','subj05','subj07']:
        selected=[r for r in records if r.subject_id==subject and r.split=='train']
        source_provenance=args.root/f'{subject}.trial_provenance.jsonl'
        identity={'manifest_sha256':sha(manifest),'cohort_sha256':sha(cohort_path),'trial_provenance_sha256':sha(source_provenance),'seeds':args.seeds,'subject':subject,'code_sha256':sha(__file__),'torch':torch.__version__}
        report_path=out/f'{subject}.provenance.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report['identity']!=identity:raise ValueError('incompatible noise resume identity')
            for seed in args.seeds:
                if sha(out/f'{subject}_seed{seed}.npy')!=report['outputs'][str(seed)]['sha256']:raise ValueError('corrupt cached noise')
            print(subject,'verified completed cache',flush=True);continue
        expected={r['trial_id']:r['brain_array_sha256'] for r in map(json.loads,source_provenance.open())}
        for r in selected:
            if hashlib.sha256(np.load(r.brain_path,allow_pickle=False).tobytes()).hexdigest()!=expected[r.trial_id]:raise ValueError('brain array differs from canonical trial provenance')
        sampler=EmpiricalCovarianceNoise.from_records(selected,subject_id=subject,manifest_sha256=identity['manifest_sha256'])
        outputs={}
        for seed in args.seeds:
            array=sampler.sample(len(cohort['examples']),seed=seed).numpy()
            if not np.isfinite(array).all():raise ValueError('nonfinite control noise')
            path=out/f'{subject}_seed{seed}.npy';temporary=path.with_suffix('.tmp.npy');np.save(temporary,array,allow_pickle=False);temporary.replace(path)
            outputs[str(seed)]={'sha256':sha(path),'shape':list(array.shape),'dtype':str(array.dtype),'finite':True}
        report={'identity':identity,'policy':'zero-mean Gaussian with exact empirical TRAIN covariance using centered-reference linear combinations; fixed cohort row order','source':sampler.provenance,'image_ids':[r['image_id'] for r in cohort['examples']],'trial_ids':[r['trial_ids_by_subject'][subject][0] for r in cohort['examples']],'outputs':outputs}
        temporary=report_path.with_suffix('.tmp.json');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(report_path)
        print(subject,'completed',len(selected),'training references',outputs,flush=True)
        del sampler;gc.collect()

if __name__=='__main__':main()
