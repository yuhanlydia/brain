"""Select complete deterministic ten-percent TRAIN image groups for P3."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.root / 'nsd_manifest.jsonl'
    records = [json.loads(line) for line in manifest.open()]
    result = {'schema_version': 1, 'seed': 41, 'image_fraction': 0.10,
              'policy': 'per-subject TRAIN images ranked by SHA256(p3:41:image_id); first floor(0.10*N) groups; all their trials consumed once',
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'subjects': {}}
    for subject in ['subj01', 'subj02', 'subj05', 'subj07']:
        rows = [r for r in records if r['subject_id'] == subject and r['split'] == 'train']
        images = sorted({r['image_id'] for r in rows}, key=lambda x: hashlib.sha256(f'p3:41:{x}'.encode()).hexdigest())
        chosen = images[:len(images)//10]
        selected_images = set(chosen)
        trial_ids = sorted((r['trial_id'] for r in rows if r['image_id'] in selected_images),
                           key=lambda x: hashlib.sha256(f'p3-order:41:{x}'.encode()).hexdigest())
        if len(trial_ids) != len(chosen) * 3:
            raise ValueError('expected all three presentations per selected image')
        result['subjects'][subject] = {
            'eligible_training_images': len(images), 'selected_image_ids': chosen,
            'trial_ids': trial_ids, 'selected_images': len(chosen), 'selected_trials': len(trial_ids),
            'optimizer_updates': math.ceil(len(trial_ids)/16),
            'final_window_examples': len(trial_ids)%16 or 16}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({s: {k: v for k, v in r.items() if not k.endswith('_ids')} for s, r in result['subjects'].items()})
    print('sha256', hashlib.sha256(args.output.read_bytes()).hexdigest())

if __name__ == '__main__':
    main()
