#!/usr/bin/env python3
"""Report an NSD matrix from bounded snapshots of append-only artifacts."""
from __future__ import annotations
import argparse, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any
import yaml

EXPOSED={"nsd33245","nsd43156"}
CAPTION=("CIDEr","METEOR","ROUGE_L","SPICE","SPICE_object_precision","SPICE_object_recall")

def snap(path:Path):
    data=path.read_bytes(); return data,{"path":str(path),"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
def lines(data:bytes,tail=False):
    out=[]; truncated=False; source=data.splitlines(keepends=True)
    for i,line in enumerate(source):
        if not line.strip(): continue
        try: out.append(json.loads(line))
        except (UnicodeDecodeError,json.JSONDecodeError):
            if tail and i==len(source)-1 and not line.endswith((b"\n",b"\r")): truncated=True; break
            raise
    return out,truncated
def ident(r): return (str(r.get("source_image_id",r["image_id"])),str(r.get("question_id","")),str(r.get("answer_id","")),tuple(map(str,r.get("source_trial_ids",[]))))
def finite(x):
    try: x=float(x)
    except (TypeError,ValueError): return None
    return x if math.isfinite(x) else None
def vals(rows,metric): return {ident(r):finite(r["task_values"].get(metric)) for r in rows if finite(r["task_values"].get(metric)) is not None}
def boot(diff,samples,seed):
    if not diff:return None
    clusters=defaultdict(list)
    for k,v in diff.items():clusters[k[0]].append(v)
    images=[mean(clusters[k]) for k in sorted(clusters)];rng=random.Random(seed)
    draws=sorted(mean(rng.choice(images) for _ in images) for _ in range(samples)); q=lambda p:draws[min(len(draws)-1,int(p*len(draws)))]
    return {"difference":mean(images),"ci95":[q(.025),q(.975)],"matched_examples":len(diff),"matched_images":len(images),"bootstrap_unit":"image_id"}
def paired(a,b,metric,samples,seed):
    x,y=vals(a,metric),vals(b,metric); common=set(x)&set(y); out=boot({k:x[k]-y[k] for k in common},samples,seed)
    if out:out.update(left_defined=len(x),right_defined=len(y),unmatched_or_not_jointly_defined=len(set(x)|set(y))-len(common))
    return out
def four(mc,ms,cc,cs,metric,samples,seed):
    maps=[vals(x,metric) for x in (mc,ms,cc,cs)];common=set(maps[0]).intersection(*maps[1:]);out=boot({k:(maps[0][k]-maps[1][k])-(maps[2][k]-maps[3][k]) for k in common},samples,seed)
    if out:out["joint_four_way_examples"]=len(common)
    return out
def summary(rows,metric):
    x=list(vals(rows,metric).values());return {"mean":mean(x) if x else None,"defined_count":len(x),"undefined_count":len(rows)-len(x)}
def task_summary(rows,metric):
    out=summary(rows,metric);categories=defaultdict(list)
    for r in rows:
        value=finite(r["task_values"].get(metric))
        if value is not None:categories[str(r.get("category","undefined"))].append(value)
    out["category_accuracy"]={k:mean(v) for k,v in sorted(categories.items())}
    out["category_macro"]=mean(out["category_accuracy"].values()) if out["category_accuracy"] else None
    return out
def posterior(rows):
    ent=[finite(r["posterior"].get("entropy")) for r in rows]; ent=[x for x in ent if x is not None]
    cov=[r["posterior"].get("coverage") for r in rows if isinstance(r["posterior"].get("coverage"),bool)]
    cats=defaultdict(list)
    for r in rows:
        x=finite(r["posterior"].get("entropy"))
        if x is not None:cats[str(r.get("category","undefined"))].append(x)
    conditioned={}
    for name in ("true_target_nll","brier","ece"):
        x=[finite(r["posterior"].get(name)) for r in rows];x=[v for v in x if v is not None]
        conditioned[name]={"mean":mean(x) if x else None,"defined_count":len(x),"undefined_count":len(rows)-len(x)}
    return {"count":len(rows),"mean_entropy":mean(ent) if ent else None,"entropy_defined_count":len(ent),"entropy_undefined_count":len(rows)-len(ent),"entropy_by_category":{k:{"mean":mean(v),"defined_count":len(v)} for k,v in sorted(cats.items())},"coverage":mean(cov) if cov else None,"coverage_defined_count":len(cov),"coverage_undefined_count":len(rows)-len(cov),"target_conditioned":conditioned,"conditioning":"sampled TRAIN-only gallery"}
def metrics(task):return CAPTION if task=="nsd_captioning" else ("exact_match",)
def group(case):
    if case["task_name"]=="nsd_captioning":return "captioning-pretrained"
    return "P3-full-10pct" if case["phase"]=="P3" else "P2-P4-main"
def analyze(conditions,task,samples,seed):
    c,s=conditions.get("trained_correct_repeat1",[]),conditions.get("trained_shuffled_repeat1",[])
    controls={n:{"metrics":{m:summary(r,m) for m in metrics(task)},"posterior":posterior(r)} for n,r in sorted(conditions.items()) if "repeat1" in n}
    repeats={n.rsplit("_",1)[-1]:{"metrics":{m:summary(r,m) for m in metrics(task)},"posterior":posterior(r)} for n,r in conditions.items() if n.startswith("trained_correct_repeat") and not n.endswith("repeat1")}
    return {"primary":{m:task_summary(c,m) for m in metrics(task)},"controls":controls,"ndg":{m:paired(c,s,m,samples,seed) for m in metrics(task)},"posterior":posterior(c),"repeat_diagnostics":repeats}

def load_case(item,trial_meta,samples,seed):
    output=Path(item["output_dir"]);find=[];prov={};conditions={};tails={};journal_bindings=[]
    checks={"config":False,"schedule":False,"runtime_checkpoint_identity":False,"journals":True,"original_rows":True}
    row={k:item.get(k) for k in ("case_id","phase","method","subject","seed")};row["task_name"]=item["task"];row["budget_group"]=group(row)
    config=Path(item["config"])
    if not config.exists():find.append("config_missing")
    else:
        config_bytes,prov["config"]=snap(config)
        if prov["config"]["sha256"]!=item.get("config_sha256"):find.append("config_sha256_mismatch")
        else:checks["config"]=True
    parsed={}
    for name in ("result.json","schedule.json","evaluation/summary.json"):
        p=output/name
        if p.exists():
            data,prov[name]=snap(p)
            try:parsed[name]=json.loads(data)
            except (UnicodeDecodeError,json.JSONDecodeError):find.append(name.replace('/','_')+"_invalid")
    schedule=parsed.get("schedule.json")
    if schedule and (schedule.get("seed")!=item["seed"] or schedule.get("max_updates")!=item["expected_optimizer_updates"]):find.append("schedule_identity_mismatch")
    if schedule:
        material={k:schedule.get(k) for k in ("schema_version","seed","trial_ids","accumulation_steps","max_updates")}
        expected_schedule_hash=hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if schedule.get("schedule_hash")!=expected_schedule_hash or len(schedule.get("trial_ids",[]))!=item["expected_examples"]:find.append("schedule_hash_or_examples_mismatch")
        elif schedule.get("seed")==item["seed"] and schedule.get("max_updates")==item["expected_optimizer_updates"]:checks["schedule"]=True
    runtime_path=output/"checkpoint_identity.json";runtime=None
    if runtime_path.exists():
        rb,prov["checkpoint_identity.json"]=snap(runtime_path)
        try:runtime=json.loads(rb)
        except (UnicodeDecodeError,json.JSONDecodeError):find.append("checkpoint_identity_invalid")
    else:find.append("checkpoint_identity_missing")
    if runtime is not None:
        identity=runtime.get("identity");canonical=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest() if isinstance(identity,dict) else None
        try:
            config_value=yaml.safe_load(config_bytes);config_value=json.loads(json.dumps(config_value));config_value.get("experiment",{}).pop("output_dir",None)
            runtime_config_hash=hashlib.sha256(json.dumps(config_value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        except Exception:runtime_config_hash=None
        # This lightweight export is produced by the trusted operational queue from
        # the checkpoint; avoid loading a multi-GB checkpoint in the reporter.
        checkpoint_bound=isinstance(runtime.get("checkpoint_sha256"),str) and len(runtime["checkpoint_sha256"])==64
        prov["checkpoint_binding"]={"mode":"trusted-operational-export","checkpoint_sha256":runtime.get("checkpoint_sha256")}
        backend=identity.get("backend",{}) if isinstance(identity,dict) else {};backend_hash=hashlib.sha256(json.dumps(backend,sort_keys=True,separators=(",",":")).encode()).hexdigest() if backend else None
        runtime_ok=(runtime.get("identity_hash")==canonical and identity.get("config")==runtime_config_hash and identity.get("schedule")==schedule.get("schedule_hash") if schedule and isinstance(identity,dict) else False)
        runtime_ok=bool(runtime_ok and identity.get("schedule_hash",identity.get("schedule"))==schedule.get("schedule_hash") and identity.get("method")==item["method"] and runtime.get("cursor")==item["expected_examples"] and runtime.get("optimizer_steps")==item["expected_optimizer_updates"] and backend_hash and backend.get("cohort_sha256") and checkpoint_bound)
        if not runtime_ok:find.append("checkpoint_identity_binding_mismatch")
        else:checks["runtime_checkpoint_identity"]=True
    evaluation=output/"evaluation"
    for raw in sorted(evaluation.glob("*.jsonl")) if evaluation.exists() else []:
        if raw.name.endswith(".scores.jsonl"):continue
        rb,prov[f"evaluation/{raw.name}"]=snap(raw);rr,rt=lines(rb,True);score=raw.with_name(raw.name.replace(".jsonl",".scores.jsonl"))
        if not score.exists():find.append(raw.stem+":scores_missing");continue
        sb,prov[f"evaluation/{score.name}"]=snap(score);sr,st=lines(sb,True);tails[raw.stem]={"raw_truncated_tail_ignored":rt,"scores_truncated_tail_ignored":st}
        side=raw.with_suffix(raw.suffix+".identity.json");identity=None
        if not side.exists():find.append(raw.stem+":identity_missing")
        else:
            ib,prov[f"evaluation/{side.name}"]=snap(side)
            try:
                side_payload=json.loads(ib);identity=side_payload.get("identity",{})
                expected_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                if side_payload.get("identity_hash")!=expected_hash:find.append(raw.stem+":identity_hash_mismatch");checks["journals"]=False
            except (UnicodeDecodeError,json.JSONDecodeError):find.append(raw.stem+":identity_invalid")
        parts=raw.stem.split("_");repeat=int(parts[-1].replace("repeat",""));control="-".join(parts[1:-1]);expected={"variant":parts[0],"control":control,"repeat_prefix":repeat,"seed":item["seed"],"subject":item["subject"]}
        if identity is None:checks["journals"]=False
        if identity is not None and any(identity.get(k)!=v for k,v in expected.items()):find.append(raw.stem+":identity_mismatch");checks["journals"]=False
        if identity is not None:
            if not identity.get("backend_provenance") or not identity.get("cohort"):find.append(raw.stem+":provenance_binding_missing");checks["journals"]=False
            else:journal_bindings.append((identity["backend_provenance"],identity["cohort"]))
        sm={(str(x.get("image_id")),str(x.get("example_id"))):x for x in sr};merged=[]
        seen=set()
        for r in rr:
            s=sm.get((str(r.get("image_id")),str(r.get("example_id"))))
            if not s:continue
            required_identity=(r.get("image_id"),r.get("question_id"),r.get("answer_id"),r.get("source_trial_ids"))
            if any(x in (None,[],"") for x in required_identity) or ident(r) in seen:find.append(raw.stem+":invalid_or_duplicate_original_identity");checks["original_rows"]=False;continue
            seen.add(ident(r));source_trials=r.get("source_trial_ids",[])
            ok=all(str(trial_meta.get(str(t),{}).get("image_id"))==str(r.get("source_image_id",r.get("image_id"))) and trial_meta.get(str(t),{}).get("subject_id")==item["subject"] for t in source_trials)
            if not ok:find.append(raw.stem+":source_trial_identity_mismatch");checks["original_rows"]=False;continue
            merged.append({**r,"task_values":s.get("task",{}),"posterior":s.get("posterior",{})})
        conditions[raw.stem]=merged
    if not journal_bindings:checks["journals"]=False
    if journal_bindings and len(set(journal_bindings))!=1:find.append("journal_backend_or_cohort_binding_mismatch");checks["journals"]=False
    if checks["runtime_checkpoint_identity"] and journal_bindings:
        expected_binding=(backend_hash,runtime["identity"]["backend"]["cohort_sha256"])
        if any(binding!=expected_binding for binding in journal_bindings):find.append("journal_runtime_binding_mismatch");checks["journals"]=False
    result=parsed.get("result.json");started=any(k!="config" for k in prov) or (output/"checkpoint.pt").exists()
    if result is None:status="started" if started else "pending"
    else:
        if schedule is None:find.append("schedule_missing_or_invalid")
        if "evaluation/summary.json" not in parsed:find.append("evaluation_summary_missing_or_invalid")
        if result.get("status")!="completed_nsd_matrix_case" or any(result.get(k)!=item[k] for k in ("method","subject","phase")):find.append("result_identity_or_status_mismatch")
        train=result.get("training",{})
        if train.get("optimizer_steps")!=item["expected_optimizer_updates"] or train.get("cursor")!=item["expected_examples"] or train.get("method")!=item["method"]:find.append("training_completion_mismatch")
        ev=result.get("evaluation",{});expected=int(ev.get("cohort_examples",0));counts=ev.get("prediction_counts",{})
        disk_summary=parsed.get("evaluation/summary.json",{})
        if disk_summary.get("cohort_examples")!=ev.get("cohort_examples") or disk_summary.get("prediction_counts")!=counts or disk_summary.get("unmatched")!=ev.get("unmatched"):find.append("evaluation_summary_result_mismatch")
        unmatched=ev.get("unmatched",{})
        if not expected or not counts:find.append("evaluation_completion_mismatch")
        for key,count in counts.items():
            missing=unmatched.get(key,[])
            if not isinstance(missing,list) or int(count)+len(set(map(str,missing)))!=expected:find.append(key.replace(":","_")+":eligibility_count_mismatch")
            elif any(str(t) not in trial_meta or trial_meta[str(t)].get("subject_id")!=item["subject"] for t in missing):find.append(key.replace(":","_")+":unmatched_identity_invalid")
        controls=("correct","shuffled","zero","covariance-noise","wrong-subject")
        required={f"trained:{c}:repeat1" for c in controls}|{"trained:correct:repeat2","trained:correct:repeat3"}
        if item.get("pretrained_baseline"):required|={f"pretrained:{c}:repeat1" for c in controls}|{"pretrained:correct:repeat2","pretrained:correct:repeat3"}
        if not required.issubset(counts):find.append("required_evaluation_conditions_missing")
        for key,count in counts.items():
            stem=key.replace(":","_")
            if len(conditions.get(stem,[]))!=count:find.append(stem+":artifact_count_mismatch")
        status="complete" if not find else "incomplete"
    filtered={k:[r for r in v if str(r.get("image_id")) not in EXPOSED] for k,v in conditions.items()}
    scientifically_valid=all(checks.values())
    correct=conditions.get("trained_correct_repeat1",[]);sessions=defaultdict(list)
    for r in correct:
        for session in {trial_meta[str(t)].get("session_id") for t in r["source_trial_ids"] if str(t) in trial_meta}:sessions[str(session)].append(r)
    identities=[{k:r.get(k) for k in ("image_id","question_id","answer_id","source_trial_ids","category","original_question")} for r in correct]
    row.update(status=status,scientific_identity_valid=scientifically_valid,scientific_identity_checks=checks,available_metrics_verification="verified" if scientifically_valid else "unverified",validation_findings=sorted(set(find)),provenance=prov,append_tails=tails,available_conditions={k:len(v) for k,v in conditions.items()},identities=identities,session_strata={s:{m:task_summary(rs,m) for m in metrics(item["task"])} for s,rs in sorted(sessions.items())},analysis=analyze(conditions,item["task"],samples,seed+item["seed"]),exposure_sensitivity={"excluded_images":sorted({str(r.get("image_id")) for v in conditions.values() for r in v}&EXPOSED),"analysis":analyze(filtered,item["task"],samples,seed+item["seed"])},_conditions=conditions)
    if result:
        training=result.get("training",{});history=training.get("history",[]);row["compute"]={"optimizer_steps":training.get("optimizer_steps"),"training_cursor":training.get("cursor"),"training_examples":sum(x.get("micro_examples",0) for x in history),"training_walltime_seconds":sum(x.get("update_seconds",0) for x in history) if any("update_seconds" in x for x in history) else None,"peak_cuda_bytes":max((x.get("peak_cuda_bytes",0) for x in history),default=None),**training.get("accounting",{})}
        for key in ("walltime_seconds","peak_memory_bytes","peak_gpu_memory_bytes","peak_cuda_bytes"):
            if key in result:row["compute"][key]=result[key]
    return row

def compare(method,baseline,filtered,samples,seed,*,baseline_type="ce"):
    a,b=method["_conditions"],baseline["_conditions"]
    if filtered:a={k:[r for r in v if str(r.get("image_id")) not in EXPOSED] for k,v in a.items()};b={k:[r for r in v if str(r.get("image_id")) not in EXPOSED] for k,v in b.items()}
    difference_name="trained_minus_pretrained" if baseline_type=="pretrained" else "method_minus_ce"
    delta_name="trained_minus_pretrained_delta_ndg" if baseline_type=="pretrained" else "delta_ndg"
    out={"baseline_type":baseline_type,"baseline_case_id":baseline["case_id"],"metrics":{}}
    for m in metrics(method["task_name"]):out["metrics"][m]={difference_name:paired(a.get("trained_correct_repeat1",[]),b.get("trained_correct_repeat1",[]),m,samples,seed),delta_name:four(a.get("trained_correct_repeat1",[]),a.get("trained_shuffled_repeat1",[]),b.get("trained_correct_repeat1",[]),b.get("trained_shuffled_repeat1",[]),m,samples,seed+1)}
    return out

def build_report(manifest_path,nsd_manifest_path,*,bootstrap_samples=2000,bootstrap_seed=1729,external_retrieval=None):
    mp,np=Path(manifest_path),Path(nsd_manifest_path);mb,mpv=snap(mp);nb,npv=snap(np);manifest=json.loads(mb);trials,_=lines(nb);meta={str(x["trial_id"]):x for x in trials}
    cases=[load_case(x,meta,bootstrap_samples,bootstrap_seed) for x in manifest["cases"]];base={}
    for r in cases:
        if r["scientific_identity_valid"] and r["task_name"]!="nsd_captioning" and r["method"]=="ce":base[(r["subject"],r["seed"],r["budget_group"])]=r
    for r in cases:
        ce=base.get((r["subject"],r["seed"],r["budget_group"]))
        baseline_type="ce"
        if r["task_name"]=="nsd_captioning" and r["scientific_identity_valid"]:
            pc={k.replace("pretrained_","trained_",1):v for k,v in r["_conditions"].items() if k.startswith("pretrained_")};ce={**r,"case_id":r["case_id"]+":pretrained","_conditions":pc}
            baseline_type="pretrained"
        if ce and r["scientific_identity_valid"] and (r["method"]!="ce" or r["task_name"]=="nsd_captioning"):
            label="comparison_to_pretrained" if baseline_type=="pretrained" else "comparison_to_ce"
            r[label]=compare(r,ce,False,bootstrap_samples,bootstrap_seed+r["seed"],baseline_type=baseline_type);r["exposure_sensitivity"][label]=compare(r,ce,True,bootstrap_samples,bootstrap_seed+r["seed"]+2,baseline_type=baseline_type)
    grouped=defaultdict(list)
    for r in cases:
        if r["status"]!="complete" or not r["scientific_identity_valid"]:continue
        for m,s in r["analysis"]["primary"].items():
            if s["mean"] is not None:grouped[(r["phase"],r["task_name"],r["method"],r["subject"],m)].append(s["mean"])
    seeds=[{"phase":k[0],"task":k[1],"method":k[2],"subject":k[3],"metric":k[4],"seed_count":len(v),"mean":mean(v),"sample_sd":stdev(v) if len(v)>1 else None,"aggregation":"per-seed mean and sample SD; images not pooled across seeds"} for k,v in sorted(grouped.items())]
    for r in cases:r.pop("_conditions")
    counts=Counter(r["status"] for r in cases);expected=int(manifest.get("case_count",len(cases)));provenance={"run_manifest":mpv,"nsd_manifest":npv}
    if external_retrieval:
        p=Path(external_retrieval);_,provenance["external_full_gallery_retrieval"]=snap(p)
    return {"schema_version":2,"coverage":{"expected_cases":expected,"observed_cases":len(cases),"complete_cases":counts["complete"],"complete":counts["complete"]==expected,"statuses":dict(sorted(counts.items()))},"cases":cases,"seed_summary":seeds,"provenance":provenance,"analysis_policy":{"no_performance_gate":True,"pairing":"exact original image/question/answer/source-trial identities before image clustering","caption":"jointly finite canonical metrics; CIDEr IDF is fixed-corpus dependent","retrieval":"TRAIN-only posterior metrics are not substituted for optional full-gallery diagnostics"}}

def write_report(report,json_path,markdown_path):
    jp,md=Path(json_path),Path(markdown_path);jp.parent.mkdir(parents=True,exist_ok=True);md.parent.mkdir(parents=True,exist_ok=True);jp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n");c=report["coverage"]
    text=["# NSD exploratory matrix report","",f"Matrix status: **{'complete' if c['complete'] else 'partial'}** ({c['complete_cases']}/{c['expected_cases']} validated complete).","No performance threshold filters this report.","","| Case | Status | Available conditions | Validation findings |","|---|---|---:|---|"]
    for r in report["cases"]:text.append(f"| {r['case_id']} | {r['status']} | {len(r['available_conditions'])} | {', '.join(r['validation_findings']) or 'none'} |")
    text += ["","Exact original identities are joined before image-cluster bootstrap. Cross-seed summaries use seed means and sample SD.","Caption comparisons retain jointly defined canonical values. CIDEr IDF is fixed-corpus dependent. SPICE object metrics are reference-grounded proxies.","TRAIN-only gallery target-conditioned metrics remain explicitly undefined when unavailable.",""];md.write_text("\n".join(text))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-manifest",required=True,type=Path);p.add_argument("--nsd-manifest",required=True,type=Path);p.add_argument("--output-json",required=True,type=Path);p.add_argument("--output-markdown",required=True,type=Path);p.add_argument("--bootstrap-samples",type=int,default=2000);p.add_argument("--bootstrap-seed",type=int,default=1729);p.add_argument("--external-retrieval",type=Path);a=p.parse_args()
    if a.bootstrap_samples<1:p.error("--bootstrap-samples must be positive")
    write_report(build_report(a.run_manifest,a.nsd_manifest,bootstrap_samples=a.bootstrap_samples,bootstrap_seed=a.bootstrap_seed,external_retrieval=a.external_retrieval),a.output_json,a.output_markdown)
if __name__=="__main__":main()
