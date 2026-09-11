"""Materialize public single-trial NSD records with globally grouped splits.

No model fitting or test-label evaluation occurs here. Images flagged shared1000
are reserved for hidden test. Other image IDs use a fixed hash for development
validation (10%) vs training (90%), shared across subjects and all repetitions.
"""
import argparse
import collections
import hashlib
import io
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
from brain_npp.data import load_jsonl_manifest


def verify_download(root, path, revision):
    """Check the pinned Hub download revision and saved content hash."""
    metadata = root / '.cache/huggingface/download' / (str(path.relative_to(root)) + '.metadata')
    lines = metadata.read_text().splitlines()
    if len(lines) < 2 or lines[0] != revision:
        raise ValueError(f'wrong or missing Hub revision for {path}')
    expected = lines[1]
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
    if len(expected) == 40:
        digest.update(f'blob {path.stat().st_size}\0'.encode())
    elif len(expected) != 64:
        raise ValueError(f'unsupported Hub content hash for {path}')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Hub content checksum mismatch for {path}')
    return {'revision': revision, 'hub_content_hash': expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    root = args.assets.expanduser().resolve()
    out = root/'nsd-full'
    (out/'images').mkdir(parents=True,exist_ok=True)
    image_hashes = {}
    image_sources = {}
    input_sources = {}
    for path in sorted((root/'nsd-inspection/webdataset_avg_new').glob('*/*.tar')):
        input_sources[str(path.relative_to(root))] = verify_download(root/'nsd-inspection', path,
            '421fb80ea4ca5600c7364ed400f9a5941380c387')
        with tarfile.open(path) as archive:
            members = {m.name:m for m in archive}
            for name,member in members.items():
                if not name.endswith('.coco73k.npy'):
                    continue
                index = int(np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False).reshape(-1)[0])
                image_id = f'nsd{index:05d}'
                image_name = name.removesuffix('.coco73k.npy')+'.jpg'
                data = archive.extractfile(members[image_name]).read()
                digest = hashlib.sha256(data).hexdigest()
                if image_id in image_hashes:
                    if image_hashes[image_id]!=digest:
                        raise ValueError(f'inconsistent bytes for image {image_id}')
                else:
                    image_hashes[image_id]=digest
                    image_sources[image_id]={'archive':str(path),'member':image_name,'sha256':digest}
                    destination = out/'images'/f'{image_id}.jpg'
                    if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                        temporary = destination.with_suffix('.jpg.tmp')
                        temporary.write_bytes(data)
                        temporary.replace(destination)
    print(f'Extracted {len(image_hashes)} unique images',flush=True)

    all_rows = []
    subject_reports = {}
    image_shared = {}
    metadata = root/'mindeyev2-inspection'
    for subject in ['subj01','subj02','subj05','subj07']:
        trials = {}
        for path in sorted((metadata/'wds'/subject).glob('*/*.tar')):
            input_sources[str(path.relative_to(root))] = verify_download(metadata, path,
                '26421f100e4c6012a35ecadb272a0ec1d999202d')
            with tarfile.open(path) as archive:
                for member in archive:
                    if not member.name.endswith('.behav.npy'):
                        continue
                    array = np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False).reshape(-1,17)
                    for value in array:
                        row = tuple(int(value[i]) for i in [0,1,2,3,4,5,16])
                        index = row[5]
                        if index in trials and trials[index]!=row:
                            raise ValueError('conflicting trial metadata')
                        trials[index]=row
        assert len(trials)==30000 and set(trials)==set(range(30000))
        repetitions=collections.defaultdict(list)
        for index,row in sorted(trials.items()):
            assert row[1]==int(subject[-2:])
            repetitions[row[0]].append(index)
        brain_dir=out/'brain'/subject
        brain_dir.mkdir(parents=True,exist_ok=True)
        counts=collections.Counter()
        beta_path = metadata/f'betas_all_{subject}_fp32_renorm.hdf5'
        input_sources[str(beta_path.relative_to(root))] = verify_download(metadata, beta_path,
            '26421f100e4c6012a35ecadb272a0ec1d999202d')
        with h5py.File(metadata/f'betas_all_{subject}_fp32_renorm.hdf5') as betas, (out/f'{subject}.trial_provenance.jsonl').open('w') as prov:
            for index,row in sorted(trials.items()):
                image_id=f'nsd{row[0]:05d}'
                assert image_id in image_hashes
                if image_id in image_shared and image_shared[image_id]!=row[-1]:
                    raise ValueError('shared-image policy differs across subjects')
                image_shared[image_id]=row[-1]
                group_hash=int.from_bytes(hashlib.sha256(f'41:{image_id}'.encode()).digest()[:8],'big')
                split='test' if row[-1] else ('val' if group_hash%10==0 else 'train')
                trial_id=f'{subject}_trial{index:06d}'
                brain_path=brain_dir/f'{trial_id}.npy'
                brain=np.asarray(betas['betas'][index],dtype=np.float32)
                assert brain.ndim==1 and np.isfinite(brain).all()
                np.save(brain_path,brain,allow_pickle=False)
                all_rows.append({'trial_id':trial_id,'subject_id':subject,'session_id':f'session{row[2]:02d}',
                                 'run_id':f'run{row[3]:02d}','trial_index':row[4]-1,'image_id':image_id,
                                 'repeat_id':repetitions[row[0]].index(index),'split':split,
                                 'brain_path':str(brain_path),'image_path':str(out/'images'/f'{image_id}.jpg')})
                prov.write(json.dumps({'trial_id':trial_id,'global_beta_index':index,'source_trial_in_run':row[4],
                                       'shared1000':bool(row[-1]),'brain_array_sha256':hashlib.sha256(brain.tobytes()).hexdigest(),
                                       'all_repeat_global_trials':repetitions[row[0]]})+'\n')
                counts[split]+=1
        subject_reports[subject]={'trials':len(trials),'unique_images':len(repetitions),'split_trial_counts':dict(counts)}
        print(subject,subject_reports[subject],flush=True)

    manifest=out/'nsd_manifest.jsonl'
    with manifest.open('w') as stream:
        for row in all_rows:
            stream.write(json.dumps(row)+'\n')
    validated=load_jsonl_manifest(manifest)
    assert len(validated)==120000
    used_images={r['image_id'] for r in all_rows}
    with (out/'image_provenance.jsonl').open('w') as stream:
        for image_id in sorted(used_images):
            stream.write(json.dumps({'image_id':image_id,**image_sources[image_id]})+'\n')
    report={'purpose':'full single-trial data preparation, no fitting or benchmark scores',
            'nsd_revision':'421fb80ea4ca5600c7364ed400f9a5941380c387',
            'mindeyev2_revision':'26421f100e4c6012a35ecadb272a0ec1d999202d',
            'preprocessing':'public betas_all_subjXX_fp32_renorm; upstream preprocessing provenance, not re-fitted here',
            'test_policy':'all shared1000 image IDs; held out from fitting across every subject',
            'development_split_policy':'int.from_bytes(sha256(41:image_id)[:8],big)%10 ==0 gives val; otherwise train',
            'trial_count':len(validated),'image_count':len(used_images),'subjects':subject_reports,
            'verified_input_sources':input_sources,
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'all_images_present':True,'all_brain_arrays_finite':True,'all_groups_split_consistent':True}
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
