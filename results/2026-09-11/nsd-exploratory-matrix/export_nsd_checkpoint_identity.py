"""Export JSON identity from a trusted, locally generated NSD checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.case_dir/'checkpoint.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    value = {'identity': state['identity'], 'identity_hash': state['identity_hash'],
             'cursor': state['cursor'], 'optimizer_steps': state['optimizer_steps'],
             'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    target = args.case_dir/'checkpoint_identity.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(target)
    print(json.dumps({'case_dir': str(args.case_dir), 'identity_hash': value['identity_hash']}))


if __name__ == '__main__':
    main()
