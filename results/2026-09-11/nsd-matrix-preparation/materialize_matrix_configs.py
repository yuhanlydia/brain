"""Materialize all predeclared single-case configs for the NSD experiment matrix."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--matrix-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--factory-checkout', type=Path, required=True)
    args = parser.parse_args()
    template = yaml.safe_load(args.template.read_text())
    budget_path = args.matrix_root / 'exploratory_budget.json'
    budget = json.loads(budget_path.read_text())
    p3 = json.loads((args.matrix_root/'p3_cohort.json').read_text())
    config_dir = args.matrix_root/'run-configs'
    config_dir.mkdir(exist_ok=True)
    cases = []
    def add(method, subject, seed, task, phase):
        config = copy.deepcopy(template)
        case_id = f'{phase}_{task}_{subject}_seed{seed}_{method}'
        config['method'], config['task'] = method, task
        experiment = config['experiment']
        experiment.pop('methods', None)
        experiment.pop('subjects', None)
        experiment.update(phase=phase, subject=subject, seed=seed,
                          output_dir=str(args.output_root/case_id),
                          max_examples=256, max_optimizer_steps=16, epochs=1)
        paths = config['paths']
        paths['vindex_checkout'] = str(args.factory_checkout)
        paths['budget'] = str(budget_path)
        paths['projector'] = '/root/brain-assets/vindex/src/llava/model_weights/llava-v1.5-7b/mm_projector.bin'
        paths['brainx_source'] = '/root/UMBRAE/umbrae/model.py'
        paths['shuffled_control'] = str(args.matrix_root/f'controls/shuffled_seed{seed}.json')
        paths['evaluation_cohort'] = str(args.matrix_root/'evaluation_cohort.json')
        paths['cached_noise_dir'] = str(args.matrix_root/'noise')
        if phase == 'P3':
            paths['training_cohort'] = str(args.matrix_root/'p3_cohort.json')
            experiment['image_group_fraction'] = 0.10
            experiment['max_examples'] = p3['subjects'][subject]['selected_trials']
            experiment['max_optimizer_steps'] = p3['subjects'][subject]['optimizer_updates']
        else:
            paths.pop('training_cohort', None)
        config['generation'].update(max_new_tokens=96 if task=='nsd_captioning' else 64,
                                    do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                    vqa_instruction='Answer with a short phrase.')
        config['trainer'].update(gradient_accumulation_steps=16, dapd_snapshot_interval=100, gradient_clip=None)
        config['optimizer'].update(name='AdamW', lr=5e-6, weight_decay=0.0)
        config['evaluation'].update(max_examples=128, main_repeat_prefix=1,
                                    diagnostic_repeat_prefixes=[1,2,3],
                                    pretrained_student_baseline=(task=='nsd_captioning' or (phase=='P2' and method=='ce')))
        config['cache'].update(clip_patch_dir=str(args.matrix_root/'clip-patches'), clip_patch_max_bytes=1<<30)
        path = config_dir/f'{case_id}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        cases.append({'case_id':case_id, 'config':str(path), 'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'output_dir':experiment['output_dir'], 'phase':phase, 'task':task, 'method':method,
                      'subject':subject, 'seed':seed, 'expected_examples':experiment['max_examples'],
                      'expected_optimizer_updates':experiment['max_optimizer_steps'],
                      'pretrained_baseline':config['evaluation']['pretrained_student_baseline']})
    for seed in budget['seeds']:
        for subject in budget['subjects']:
            for method in budget['vqa_methods']:
                phase = 'P4' if method.startswith('npp-') else 'P2'
                add(method,subject,seed,'nsd_vqa',phase)
    for subject in budget['subjects']:
        for method in budget['p3']['methods']:
            add(method,subject,41,'nsd_vqa','P3')
    for seed in budget['captioning']['seeds']:
        for subject in budget['captioning']['subjects']:
            add('npp-opsd',subject,seed,'nsd_captioning','captioning')
    if len(cases)!=164 or len({x['case_id'] for x in cases})!=164:
        raise ValueError('matrix coverage mismatch')
    manifest = {'schema_version':1, 'purpose':'complete predeclared available-backend exploratory matrix',
                'budget_sha256':hashlib.sha256(budget_path.read_bytes()).hexdigest(),
                'template_sha256':hashlib.sha256(args.template.read_bytes()).hexdigest(),
                'ordering':'fixed seed/subject/method primary comparisons, then full-subset P3, then captioning; no performance-dependent scheduling',
                'pretrained_baseline_policy':'one VQA pretrained baseline per subject/seed in P2 CE case, shared for analysis across other methods; caption baseline in each caption case',
                'case_count':len(cases), 'cases':cases}
    path = args.matrix_root/'run_manifest.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'training_examples':sum(x['expected_examples'] for x in cases),
                      'optimizer_updates':sum(x['expected_optimizer_updates'] for x in cases),
                      'manifest':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))

if __name__=='__main__':
    main()
