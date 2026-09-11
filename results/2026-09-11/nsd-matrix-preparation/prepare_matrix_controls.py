"""Persist matched test-control maps before comparative model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
from brain_npp.data import load_jsonl_manifest
from brain_npp.controls import build_block_derangement
from brain_npp.nsd_data import build_wrong_subject_mapping,save_control_mapping,WrongSubjectResolver


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.root;path=root/'nsd_manifest.jsonl';records=[r for r in load_jsonl_manifest(path) if r.split=='test'];lookup={r.trial_id:r for r in records};mh=hashlib.sha256(path.read_bytes()).hexdigest();out=args.output;out.mkdir(parents=True,exist_ok=True)
    cohort=json.loads((root/'matrix/evaluation_cohort.json').read_text());selected={s:[e['trial_ids_by_subject'][s][0] for e in cohort['examples']] for s in ['subj01','subj02','subj05','subj07']}
    reports={}
    for seed in [41,42,43]:
        mapping,unmatched=build_block_derangement(records,seed,preserve_run=True,adjacency_radius=2)
        for a,b in mapping.items():
            x,y=lookup[a],lookup[b]
            if not(x.subject_id==y.subject_id and x.session_id==y.session_id and x.run_id==y.run_id and x.image_id!=y.image_id and abs(x.trial_index-y.trial_index)>2):raise ValueError('invalid shuffle pair')
        p=out/f'shuffled_seed{seed}.json';save_control_mapping(p,mapping,unmatched,control='shuffled',manifest_sha256=mh)
        data=json.loads(p.read_text());data.update(seed=seed,preserve_run=True,adjacency_radius=2,population='all shared-test trials');p.write_text(json.dumps(data,indent=2)+'\n')
        reports[str(seed)]={'matched':len(mapping),'total':len(records),'first_repeat_cohort_matched':{s:sum(t in mapping for t in ids) for s,ids in selected.items()},'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    mapping,unmatched=build_wrong_subject_mapping(records);p=out/'wrong_subject.json';save_control_mapping(p,mapping,unmatched,control='wrong-subject',manifest_sha256=mh);WrongSubjectResolver.from_artifact(p,records,manifest_sha256=mh)
    reports['wrong-subject']={'matched':len(mapping),'total':len(records),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))

if __name__=='__main__':main()
