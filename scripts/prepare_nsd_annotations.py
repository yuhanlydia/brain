"""Join public NSD-VQA and COCO caption references to an existing NSD manifest."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import zipfile

import pyarrow.dataset as ds

from brain_npp.data import load_jsonl_manifest
from brain_npp.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets.expanduser().resolve()
    root = assets / 'nsd-full'
    records = load_jsonl_manifest(root / 'nsd_manifest.jsonl')
    splits = {int(record.image_id[3:]): record.split for record in records}
    vqa_path = assets / 'nsd-vqa/nsd_vqa.parquet'
    metadata = (assets / 'nsd-vqa/.cache/huggingface/download/nsd_vqa.parquet.metadata').read_text().splitlines()
    vqa_hash = sha256_file(vqa_path)
    if metadata[:2] != ['363198d258ef8fb56f55e211142dcce848ce3219', vqa_hash]:
        raise ValueError('VQA download revision/content hash mismatch')
    source = ds.dataset(vqa_path)
    table = source.to_table(filter=ds.field('nsd_image_id').isin(sorted(splits)))
    mapping = {}
    questions = collections.Counter()
    categories = collections.Counter()
    streams = {split: (root / f'vqa_{split}.jsonl').open('w') for split in ['train', 'val', 'test']}
    try:
        for batch in table.to_batches(max_chunksize=4096):
            for row in batch.to_pylist():
                nsd_id, coco_id = row['nsd_image_id'], row['coco_image_id']
                if nsd_id in mapping and mapping[nsd_id] != coco_id:
                    raise ValueError('conflicting NSD/COCO image identifiers')
                mapping[nsd_id] = coco_id
                split = splits[nsd_id]
                streams[split].write(json.dumps(row) + '\n')
                questions[split] += 1
                categories[row['category']] += 1
    finally:
        for stream in streams.values():
            stream.close()
    missing = set(splits) - set(mapping)
    if missing:
        raise ValueError(f'{len(missing)} manifest images have no VQA annotations')

    captions = collections.defaultdict(list)
    archive_path = assets / 'coco/annotations_trainval2017.zip'
    coco_provenance = json.loads((assets / 'coco/provenance.json').read_text())
    if sha256_file(archive_path) != coco_provenance['archive_sha256']:
        raise ValueError('COCO archive content hash mismatch')
    caption_hashes = {}
    for filename in ['captions_train2017.json', 'captions_val2017.json']:
        source_path = assets / 'coco' / filename
        caption_hashes[filename] = sha256_file(source_path)
        with zipfile.ZipFile(archive_path) as archive:
            expected = hashlib.sha256(archive.read('annotations/' + filename)).hexdigest()
        if caption_hashes[filename] != expected:
            raise ValueError('caption file differs from source archive')
        for row in json.loads(source_path.read_text())['annotations']:
            captions[row['image_id']].append({'caption_id': row['id'], 'caption': row['caption']})
    images = collections.Counter()
    references = collections.Counter()
    streams = {split: (root / f'captions_{split}.jsonl').open('w') for split in ['train', 'val', 'test']}
    try:
        for nsd_id, split in sorted(splits.items()):
            values = captions[mapping[nsd_id]]
            if not values:
                raise ValueError(f'NSD image {nsd_id} has no COCO caption references')
            streams[split].write(json.dumps({'image_id': f'nsd{nsd_id:05d}', 'nsd_image_id': nsd_id,
                'coco_image_id': mapping[nsd_id], 'references': values}) + '\n')
            images[split] += 1
            references[split] += len(values)
    finally:
        for stream in streams.values():
            stream.close()
    files = [root / f'{task}_{split}.jsonl' for task in ['vqa', 'captions'] for split in streams]
    report = {
        'purpose': 'annotation preparation only; no evaluation or test-label tuning',
        'vqa_revision': '363198d258ef8fb56f55e211142dcce848ce3219',
        'vqa_annotation_type': 'automatically generated NSD-VQA annotations',
        'coco_source': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'vqa_source_sha256': vqa_hash, 'caption_source_sha256': caption_hashes,
        'coco_archive_sha256': coco_provenance['archive_sha256'],
        'manifest_sha256': hashlib.sha256((root / 'nsd_manifest.jsonl').read_bytes()).hexdigest(),
        'image_counts': dict(images), 'question_counts': dict(questions), 'caption_counts': dict(references),
        'question_categories': dict(categories),
        'output_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (root / 'annotations.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
