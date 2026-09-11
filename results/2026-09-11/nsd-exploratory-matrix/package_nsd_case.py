"""Package a completed NSD case's auditable results, excluding model tensors."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import torch


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    case = json.loads(manifest_bytes)['cases'][args.case_index]
    source = Path(case['output_dir'])
    config_bytes = Path(case['config']).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != case['config_sha256']:
        raise ValueError('Configuration differs from declared manifest')
    result = json.loads((source/'result.json').read_bytes())
    if result['status'] != 'completed_nsd_matrix_case':
        raise ValueError('Case has not completed')
    if (result['training']['cursor'], result['training']['optimizer_steps']) != (case['expected_examples'], case['expected_optimizer_updates']):
        raise ValueError('Training schedule is incomplete')
    checkpoint = source/'checkpoint.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    files = {'config.yaml': config_bytes, 'case_manifest.json': json_bytes(case),
             'checkpoint_identity.json': json_bytes({'identity': state['identity'], 'identity_hash': state['identity_hash'],
                                                      'cursor': state['cursor'], 'optimizer_steps': state['optimizer_steps'],
                                                      'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})}
    del state
    for name in ('result.json', 'schedule.json', 'selected_examples.json'):
        files[name] = (source/name).read_bytes()
    for path in sorted((source/'evaluation').iterdir()):
        if path.is_file() and path.suffix in ('.json', '.jsonl'):
            files['evaluation/'+path.name] = path.read_bytes()
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}
    files['SHA256SUMS'] = ''.join(f'{sha}  {name}\n' for name, sha in checksums.items()).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for name, value in sorted(files.items()):
            entry = tarfile.TarInfo(name); entry.size = len(value); entry.mode = 0o644
            entry.mtime = entry.uid = entry.gid = 0
            archive.addfile(entry, io.BytesIO(value))
    compressed = gzip.compress(buffer.getvalue(), compresslevel=9, mtime=0)
    args.destination.mkdir(parents=True, exist_ok=True)
    archive_path = args.destination/(case['case_id']+'.tar.gz')
    archive_path.write_bytes(compressed)
    summary = {'case_id': case['case_id'], 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
               'archive': archive_path.name, 'archive_sha256': hashlib.sha256(compressed).hexdigest(),
               'archive_bytes': len(compressed), 'file_sha256': checksums,
               'training': result['training'], 'evaluation': result['evaluation'],
               'contents': 'Predictions, per-example scores, summaries, selected examples, config, schedule and checkpoint identity; model tensors remain local.'}
    (args.destination/(case['case_id']+'.summary.json')).write_bytes(json_bytes(summary))
    print(json.dumps({'case_id': case['case_id'], 'archive_bytes': len(compressed), 'archive_sha256': summary['archive_sha256']}))


if __name__ == '__main__':
    main()
