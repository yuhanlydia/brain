"""Pre-outcome, category-stratified NSD shared-test evaluation cohort."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images', type=int, default=128)
    p.add_argument('--seed', type=int, default=41001)
    args = p.parse_args()
    qa_path = args.root / 'vqa_test.jsonl'
    manifest_path = args.root / 'nsd_manifest.jsonl'
    rows = [json.loads(line) for line in qa_path.open()]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['category']].append(row)
    if args.images < len(groups):
        raise ValueError('cohort too small for category coverage')
    def rank(row):
        key = f"{args.seed}:{row['nsd_image_id']}:{row['question_id']}:{row['answer_id']}"
        return hashlib.sha256(key.encode()).hexdigest()
    selected, used = [], set()
    for category in sorted(groups, key=lambda c: (len(groups[c]), c)):
        quota = args.images // len(groups)
        for row in sorted(groups[category], key=rank):
            if row['nsd_image_id'] in used:
                continue
            selected.append(row)
            used.add(row['nsd_image_id'])
            quota -= 1
            if quota == 0:
                break
    for row in sorted(rows, key=rank):
        if len(selected) >= args.images:
            break
        if row['nsd_image_id'] not in used:
            selected.append(row)
            used.add(row['nsd_image_id'])
    if len(selected) != args.images or {r['category'] for r in selected} != set(groups):
        raise ValueError('failed exact image count/category coverage')
    selected.sort(key=rank)
    trials = collections.defaultdict(list)
    for line in manifest_path.open():
        record = json.loads(line)
        if record['split'] == 'test' and int(record['image_id'][3:]) in used:
            trials[(record['subject_id'], record['image_id'])].append(record)
    pairs = []
    for row in selected:
        image_id = f"nsd{row['nsd_image_id']:05d}"
        by_subject = {}
        for subject in ['subj01', 'subj02', 'subj05', 'subj07']:
            matched = sorted(trials[(subject, image_id)], key=lambda r: r['repeat_id'])
            if [r['repeat_id'] for r in matched] != [0, 1, 2]:
                raise ValueError('expected all three shared-test presentations')
            by_subject[subject] = [r['trial_id'] for r in matched]
        pairs.append({'image_id': image_id, 'question_id': row['question_id'],
                      'answer_id': row['answer_id'], 'category': row['category'],
                      'trial_ids_by_subject': by_subject})
    result = {'schema_version': 1, 'selection_seed': args.seed, 'image_count': len(pairs),
              'purpose': 'pre-outcome category-stratified exploratory shared-test subset',
              'selection_policy': 'rare categories first; up to floor(images/categories) distinct images per category, then globally hashed unused-image fill',
              'category_counts': dict(sorted(collections.Counter(r['category'] for r in selected).items())),
              'full_test_category_counts': dict(sorted((c, len(v)) for c, v in groups.items())),
              'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [qa_path, manifest_path, args.root/'captions_test.jsonl']},
              'evaluation_policy': {'main_repeats': [0], 'repeat_diagnostics': [[0], [0,1], [0,1,2]],
                                    'caption_image_ids': [r['image_id'] for r in pairs],
                                    'reporting': 'sample micro accuracy is stratified-subset accuracy, not full-test prevalence-weighted accuracy; report category macro separately'},
              'examples': pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'images': len(pairs), 'categories': len(groups), 'trial_records': len(pairs)*4*3,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'output': str(args.output)}))

if __name__ == '__main__':
    main()
