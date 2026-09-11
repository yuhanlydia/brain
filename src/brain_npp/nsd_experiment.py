"""Configuration-to-training/evaluation orchestration for real NSD matrix runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .nsd_runner import MatchedSchedule, MatrixExperiment, Method, select_image_group_fraction


def run_matrix_config(config: Mapping[str, Any], *, backend_builder=None) -> dict[str, Any]:
    experiment=_mapping(config.get("experiment"),"experiment"); trainer=_mapping(config.get("trainer"),"trainer")
    if experiment.get("epochs",1) != 1: raise ValueError("only experiment.epochs=1 is supported")
    if experiment.get("phase") not in {"P2","P3","P4","matrix","captioning"}: raise ValueError("full NSD phase is invalid")
    method=Method(str(config.get("method"))); output=Path(_string(experiment,"output_dir"));output.mkdir(parents=True,exist_ok=True)
    budget_path=config.get('paths',{}).get('budget')
    if budget_path and Path(budget_path).is_file():_validate_budget(config,json.loads(Path(budget_path).read_text()))
    builder=backend_builder
    if builder is None:
        from .adapters.nsd_backend import RealNSDBackend
        builder=RealNSDBackend
    backend=builder(config); accumulation=_positive_int(trainer,"gradient_accumulation_steps")
    ordered_p3=False
    if experiment["phase"]=="P3":
        if config.get('paths',{}).get('training_cohort'):
            records={key:(backend.records[key].subject_id,backend.records[key].image_id,backend.records[key].split) for key in backend.records}
            selected=load_p3_trial_ids(config['paths']['training_cohort'],subject=experiment['subject'],seed=int(experiment['seed']),manifest_path=config['paths']['manifest'],records=records,accumulation=accumulation);ordered_p3=True
        else:
            fraction=float(experiment.get("image_group_fraction",.1)); selected=select_image_group_fraction(backend.training_trial_images(),fraction=fraction,seed=int(experiment["seed"]))
        max_examples=len(selected); max_updates=math.ceil(max_examples/accumulation)
    else:
        selected=backend.training_trial_ids(); max_examples=_positive_int(experiment,"max_examples"); max_updates=_positive_int(experiment,"max_optimizer_steps")
    schedule_path=output/"schedule.json"
    if schedule_path.exists(): schedule=MatchedSchedule.load(schedule_path)
    else:
        schedule=(MatchedSchedule.ordered(selected,seed=int(experiment['seed']),accumulation_steps=accumulation,max_updates=max_updates) if ordered_p3 else
                  MatchedSchedule.create(selected,seed=int(experiment["seed"]),max_examples=max_examples,accumulation_steps=accumulation,max_updates=max_updates));schedule.save(schedule_path)
    if schedule.accumulation_steps!=accumulation or schedule.max_updates!=max_updates: raise ValueError("saved schedule budget mismatch")
    if hasattr(backend,"selected_example_manifest"):
        selected={"schema_version":1,"schedule_hash":schedule.schedule_hash,"method":method.value,
                  "view_policy":backend.selected_view_policy(method) if hasattr(backend,"selected_view_policy") else {},
                  "examples":backend.selected_example_manifest(schedule.trial_ids,method)}
        selected["manifest_hash"]=hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        selected_path=output/"selected_examples.json"
        if selected_path.exists() and json.loads(selected_path.read_text())!=selected:raise ValueError("selected-example manifest mismatch")
        if not selected_path.exists():_atomic_json(selected_path,selected)
    identity={"config":_identity_config(config),"backend":getattr(backend,"provenance",{}),"schedule":schedule.schedule_hash}
    training=MatrixExperiment(backend,schedule,method,output,identity=identity,
                              gradient_clip=trainer.get("gradient_clip")).train()
    evaluation=backend.evaluate(output)
    result={"status":"completed_nsd_matrix_case","phase":experiment["phase"],"method":method.value,
            "subject":experiment.get("subject"),"training":training,"evaluation":evaluation}
    _atomic_json(output/"result.json",result);return result


def _identity_config(config):
    # Output location is operational rather than scientific identity.
    value=json.loads(json.dumps(config));value.get("experiment",{}).pop("output_dir",None)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def load_p3_trial_ids(path,*,subject,seed,manifest_path,records,accumulation):
    payload=json.loads(Path(path).read_text());digest=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if payload.get('schema_version')!=1 or payload.get('seed')!=seed or payload.get('manifest_sha256')!=digest:raise ValueError('P3 cohort identity mismatch')
    row=payload.get('subjects',{}).get(subject)
    if not isinstance(row,dict):raise ValueError('P3 cohort subject missing')
    trials=tuple(row.get('trial_ids',()));images=set(row.get('selected_image_ids',()))
    if len(set(trials))!=len(trials) or row.get('selected_trials')!=len(trials):raise ValueError('P3 trial list invalid')
    for trial in trials:
        if trial not in records or records[trial]!=(subject,records[trial][1],'train') or records[trial][1] not in images:raise ValueError('P3 trial does not belong to selected image group')
    expected=math.ceil(len(trials)/accumulation);final=len(trials)%accumulation or accumulation
    if row.get('optimizer_updates')!=expected or row.get('final_window_examples')!=final:raise ValueError('P3 accumulation metadata mismatch')
    return trials


def _validate_budget(config,budget):
    experiment=config['experiment'];training=budget['training'];task=config.get('task')
    if experiment.get('seed') not in budget['seeds'] or experiment.get('subject') not in budget['subjects']:raise ValueError('run is outside predeclared subject/seed budget')
    if config.get('method') not in (budget['captioning']['methods'] if task=='nsd_captioning' else budget['vqa_methods']):raise ValueError('method is outside predeclared task budget')
    if experiment.get('phase')!='P3':
        if experiment.get('max_examples')!=training['max_examples'] or experiment.get('max_optimizer_steps')!=training['max_optimizer_steps']:raise ValueError('run training budget differs from predeclared budget')
    if config['trainer'].get('gradient_accumulation_steps')!=training['gradient_accumulation_steps']:raise ValueError('accumulation differs from predeclared budget')


def _mapping(value,name):
    if not isinstance(value,Mapping):raise ValueError(f"{name} must be a mapping")
    return value


def _string(mapping,key):
    value=mapping.get(key)
    if not isinstance(value,str) or not value:raise ValueError(f"{key} must be a nonempty string")
    return value


def _positive_int(mapping,key):
    value=mapping.get(key)
    if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f"{key} must be a positive integer")
    return value


def _atomic_json(path,payload):
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n");temporary.replace(path)
