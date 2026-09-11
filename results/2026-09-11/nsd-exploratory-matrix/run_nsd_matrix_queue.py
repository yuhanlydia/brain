"""Operational sequential launcher for the fixed NSD case manifest.

Each case uses the normal CLI. A nonzero exit leaves the queue resumable for
controller debugging; numerical performance is never read as a stop condition.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--end-index', type=int)
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--require-evidence', type=Path)
    args = parser.parse_args()
    payload = args.manifest.read_bytes()
    manifest = json.loads(payload)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / 'queue-state.json'
    if args.wait_for_pid is not None:
        process_path = Path('/proc') / str(args.wait_for_pid) / 'cmdline'
        expected_command = process_path.read_bytes() if process_path.exists() else None
        atomic_json(state_path, {'status': 'waiting_for_first_case_verification', 'pid': args.wait_for_pid, 'manifest_sha256': hashlib.sha256(payload).hexdigest()})
        while process_path.exists():
            try:
                current_command = process_path.read_bytes()
            except FileNotFoundError:
                break
            if not current_command or current_command != expected_command:
                break
            time.sleep(2)
        if args.require_evidence is None or not args.require_evidence.is_file():
            atomic_json(state_path, {'status': 'first_case_needs_debug', 'pid': args.wait_for_pid})
            raise RuntimeError('First case did not produce required resume verification')
        evidence = json.loads(args.require_evidence.read_text())
        if not evidence.get('preserved_history_prefix'):
            raise RuntimeError('First-case resume verification incomplete')
    environment = dict(os.environ, PYTHONPATH=str(args.checkout / 'src'), PYTHONUNBUFFERED='1')
    cases = manifest['cases']
    end = args.end_index if args.end_index is not None else len(cases)
    for index in range(args.start_index, end):
        case = cases[index]
        config = Path(case['config'])
        if hashlib.sha256(config.read_bytes()).hexdigest() != case['config_sha256']:
            raise RuntimeError('Config changed after materialization: ' + case['case_id'])
        state = {'manifest_sha256': hashlib.sha256(payload).hexdigest(),
                 'case_index': index, 'case_id': case['case_id'], 'total_cases': len(cases),
                 'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        atomic_json(state_path, state)
        print(json.dumps(state), flush=True)
        log_path = args.state_dir / (case['case_id'] + '.log')
        with log_path.open('a') as log:
            run = subprocess.run(['brain-npp', 'train', '--config', str(config)],
                                 cwd=args.checkout, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state.update(returncode=run.returncode,
                     finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if run.returncode:
            state['status'] = 'needs_debug'
            atomic_json(state_path, state)
            print(json.dumps(state), flush=True)
            return run.returncode
        result = json.loads((Path(case['output_dir']) / 'result.json').read_text())
        if (result['training']['cursor'] != case['expected_examples'] or
                result['training']['optimizer_steps'] != case['expected_optimizer_updates']):
            state['status'] = 'incomplete_schedule'
            atomic_json(state_path, state)
            raise RuntimeError('Successful CLI did not finish declared schedule')
        subprocess.run([sys.executable, str(Path(__file__).with_name('export_nsd_checkpoint_identity.py')), '--case-dir', case['output_dir']], check=True)
        state['status'] = 'completed'
        atomic_json(state_path, state)
        print(json.dumps(state), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
