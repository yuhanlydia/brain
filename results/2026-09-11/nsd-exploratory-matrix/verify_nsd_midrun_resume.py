"""Interrupt the first real case after an atomic update, then resume its CLI."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config['experiment']['output_dir'])
    checkpoint = output / 'checkpoint.pt'
    if checkpoint.exists():
        raise RuntimeError('Controlled interruption requires a fresh case; existing checkpoint preserved')
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    command = ['brain-npp', 'train', '--config', str(args.config)]
    environment = dict(os.environ, PYTHONPATH=str(args.checkout / 'src'), PYTHONUNBUFFERED='1')
    with (args.evidence_dir / 'interrupted.log').open('w') as log:
        process = subprocess.Popen(command, cwd=args.checkout, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None and not checkpoint.exists():
            time.sleep(1)
        if process.poll() is not None:
            raise RuntimeError('CLI exited before controlled interruption: ' + str(process.returncode))
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    steps = state['optimizer_steps']
    if not 0 < steps < config['experiment']['max_optimizer_steps']:
        raise RuntimeError('Checkpoint is not an intermediate optimizer boundary')
    evidence = {'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                'checkpoint_sha256_at_interruption': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'cursor_at_interruption': state['cursor'], 'steps_at_interruption': steps,
                'interrupted_process_returncode': process.returncode,
                'interrupted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'history_before': state['extra']['history']}
    del state
    (args.evidence_dir / 'interruption.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({k: v for k, v in evidence.items() if k != 'history_before'}), flush=True)
    with (args.evidence_dir / 'resumed.log').open('w') as log:
        completed = subprocess.run(command, cwd=args.checkout, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('Resumed CLI failed: ' + str(completed.returncode))
    result = json.loads((output / 'result.json').read_text())
    training = result['training']
    assert training['resumed'] is True
    assert training['history'][:steps] == evidence['history_before']
    assert training['optimizer_steps'] == config['experiment']['max_optimizer_steps']
    assert training['cursor'] == config['experiment']['max_examples']
    counts = {}
    for path in (output / 'evaluation').glob('*.jsonl'):
        if '.scores.' in path.name:
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == len({row['example_id'] for row in rows})
        counts[path.name] = len(rows)
    evidence.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    final_cursor=training['cursor'], final_steps=training['optimizer_steps'],
                    preserved_history_prefix=True, unique_prediction_counts=counts)
    (args.evidence_dir / 'resume_verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'status': 'verified_real_midtraining_resume', 'final_steps': training['optimizer_steps'],
                      'prediction_counts': counts}), flush=True)


if __name__ == '__main__':
    main()
