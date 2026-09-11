"""Rebuild a declared P1 cohort from the validated full NSD manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from brain_npp.data import load_jsonl_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    args = parser.parse_args()
    root = args.assets.expanduser().resolve()
    output = root / 'p1-preparation'
    ids = json.loads(args.cohort.read_text())['trial_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('cohort must contain unique trial identifiers')
    records = {row.trial_id: row for row in load_jsonl_manifest(root / 'nsd-full/nsd_manifest.jsonl')}
    selected = [records[trial_id] for trial_id in ids]
    if any(row.split == 'test' or row.subject_id != 'subj01' for row in selected):
        raise ValueError('P1 cohort must be subject01 development trials, excluding hidden test')
    (output / 'brain').mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    rows = []
    for record in selected:
        destinations = {'brain_path': output / 'brain' / f'{record.trial_id}.npy',
                        'image_path': output / 'images' / f'{record.image_id}.jpg'}
        for field, destination in destinations.items():
            source = Path(getattr(record, field))
            if destination.exists():
                if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                    raise ValueError(f'existing P1 asset differs: {destination}')
            else:
                shutil.copyfile(source, destination)
        row = dict(record.__dict__)
        # Historical P1 is a development-only shape smoke, separate from full splits.
        row['split'] = 'train'
        row.update({key: str(value) for key, value in destinations.items()})
        rows.append(row)
    text = ''.join(json.dumps(row) + '\n' for row in rows)
    path = output / 'nsd_manifest.jsonl'
    if path.exists() and path.read_text() != text:
        raise ValueError('existing P1 manifest differs; choose a separate asset directory')
    path.write_text(text)
    load_jsonl_manifest(path)
    images = {int(row.image_id[3:]) for row in selected}
    import pyarrow.dataset as ds
    table = ds.dataset(root / 'nsd-vqa/nsd_vqa.parquet').to_table(filter=ds.field('nsd_image_id').isin(sorted(images)))
    with (output / 'vqa.jsonl').open('w') as stream:
        for row in table.to_pylist():
            stream.write(json.dumps(row) + '\n')
    print(json.dumps({'purpose': 'P1-only development cohort; full runs must refit on their own training split',
        'trials': len(rows), 'images': len(images), 'questions': len(table),
        'source_validation_trials': sum(row.split == 'val' for row in selected),
        'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
