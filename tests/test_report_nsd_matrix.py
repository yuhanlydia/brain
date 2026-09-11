import hashlib, importlib.util, json
from pathlib import Path

P=Path(__file__).parents[1]/"scripts/report_nsd_matrix.py";S=importlib.util.spec_from_file_location("report",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
BACKEND={"cohort_sha256":"cohort-hash","code":{"runner":"abc"}};BACKEND_HASH=hashlib.sha256(json.dumps(BACKEND,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def put(path,value,raw=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if raw:path.write_bytes(value);return
    path.write_text("".join(json.dumps(x)+"\n" for x in value) if path.suffix==".jsonl" else json.dumps(value))
def pred(image,q,value,metric="exact_match",posterior=None):
    raw={"image_id":image,"source_image_id":image,"question_id":q,"answer_id":q+10,"example_id":f"{image}:{q}","source_trial_ids":[f"t-{image}"],"category":"object"}
    score={"image_id":image,"example_id":raw["example_id"],"task":{metric:value},"posterior":posterior or {}}
    return raw,score
def condition(case,name,pairs,identity=True,tail=b""):
    root=Path(case["output_dir"])/"evaluation";raw,scores=zip(*pairs);rp=root/f"{name}.jsonl";sp=root/f"{name}.scores.jsonl";put(rp,list(raw));put(sp,list(scores))
    if tail:rp.write_bytes(rp.read_bytes()+tail);sp.write_bytes(sp.read_bytes()+tail)
    if identity:
        p=name.split("_");value={"variant":p[0],"control":"-".join(p[1:-1]),"repeat_prefix":int(p[-1][6:]),"seed":case["seed"],"subject":case["subject"],"backend_provenance":BACKEND_HASH,"cohort":"cohort-hash"};identity_hash=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest();put(rp.with_suffix(".jsonl.identity.json"),{"schema_version":1,"identity":value,"identity_hash":identity_hash})
def make_case(tmp,cid,phase,method,task="nsd_vqa",seed=41,complete=True):
    out=tmp/cid;cfg=tmp/f"{cid}.yaml";config={"task":task,"method":method,"experiment":{"phase":phase,"seed":seed,"subject":"subj01","output_dir":str(out)}};put(cfg,config);case={"case_id":cid,"phase":phase,"task":task,"method":method,"subject":"subj01","seed":seed,"expected_examples":4,"expected_optimizer_updates":2,"output_dir":str(out),"config":str(cfg),"config_sha256":hashlib.sha256(cfg.read_bytes()).hexdigest()}
    schedule={"schema_version":1,"seed":seed,"trial_ids":[f"train-{i}" for i in range(4)],"accumulation_steps":2,"max_updates":2};schedule["schedule_hash"]=hashlib.sha256(json.dumps(schedule,sort_keys=True,separators=(",",":")).encode()).hexdigest();put(out/"schedule.json",schedule)
    identity_config=json.loads(json.dumps(config));identity_config["experiment"].pop("output_dir");config_hash=hashlib.sha256(json.dumps(identity_config,sort_keys=True,separators=(",",":")).encode()).hexdigest();identity={"config":config_hash,"backend":BACKEND,"schedule":schedule["schedule_hash"],"schedule_hash":schedule["schedule_hash"],"method":method};identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest();checkpoint=b"checkpoint";put(out/"checkpoint.pt",checkpoint,raw=True);put(out/"checkpoint_identity.json",{"identity":identity,"identity_hash":identity_hash,"cursor":4,"optimizer_steps":2,"checkpoint_sha256":hashlib.sha256(checkpoint).hexdigest()})
    if complete:put(out/"result.json",{"status":"completed_nsd_matrix_case","phase":phase,"method":method,"subject":"subj01","training":{"optimizer_steps":2,"cursor":4,"method":method,"accounting":{"generated_tokens":8}},"evaluation":{"cohort_examples":4,"prediction_counts":{}}})
    return case
def finalize(case,names):
    p=Path(case["output_dir"])/"result.json";x=json.loads(p.read_text());counts={n.replace("_",":"):len((Path(case["output_dir"])/"evaluation"/f"{n}.scores.jsonl").read_text().splitlines()) for n in names};unmatched={k:[f"missing-{i}" for i in range(4-v)] for k,v in counts.items()};x["evaluation"].update(prediction_counts=counts,unmatched=unmatched);put(p,x);put(Path(case["output_dir"])/"evaluation/summary.json",x["evaluation"])
def manifest(tmp,cases,count=None):
    p=tmp/"manifest.json";put(p,{"case_count":count or len(cases),"cases":cases});n=tmp/"nsd.jsonl";put(n,[{"trial_id":f"t-{i}","image_id":i,"subject_id":"subj01","session_id":"s1"} for i in ("nsd33245","i2","i3","i4")]);return p,n

def test_started_case_safe_tail_and_completion_validation(tmp_path):
    active=make_case(tmp_path,"active","P2","ce",complete=False); names=[]
    condition(active,"trained_correct_repeat1",[pred(i,q,v,posterior={"entropy":q+1}) for q,(i,v) in enumerate(zip(("nsd33245","i2","i3","i4"),(1,1,0,0)))])
    condition(active,"trained_shuffled_repeat1",[pred("i2",1,0)],tail=b'{"unfinished"');p,n=manifest(tmp_path,[active])
    r=M.build_report(p,n,bootstrap_samples=20)["cases"][0]
    assert r["status"]=="started" and r["available_conditions"]["trained_correct_repeat1"]==4
    assert r["append_tails"]["trained_shuffled_repeat1"]["scores_truncated_tail_ignored"]
    bad=make_case(tmp_path,"bad","P2","ce");condition(bad,"trained_correct_repeat1",[pred("i2",1,1)]*4,identity=False);finalize(bad,["trained_correct_repeat1"]);p,n=manifest(tmp_path,[bad])
    r=M.build_report(p,n,bootstrap_samples=10)["cases"][0]
    assert r["status"]=="incomplete" and "trained_correct_repeat1:identity_missing" in r["validation_findings"]

def test_budget_baseline_exact_four_way_and_full_sensitivity(tmp_path):
    p2=make_case(tmp_path,"p2ce","P2","ce");p3=make_case(tmp_path,"p3ce","P3","ce");method=make_case(tmp_path,"p4m","P4","npp-opsd")
    images=("nsd33245","i2","i3","i4")
    data={"p2ce":((1,0,0,0),(0,0,0,0)),"p3ce":((1,1,1,1),(0,0,0,0)),"p4m":((1,1,0,0),(0,0,0,0))}
    for c in (p2,p3,method):
        names=[]
        for ctl,values in zip(("correct","shuffled"),data[c["case_id"]]):
            name=f"trained_{ctl}_repeat1";condition(c,name,[pred(i,q,v,posterior={"entropy":1.0,"coverage":False}) for q,(i,v) in enumerate(zip(images,values))]);names.append(name)
        finalize(c,names)
    p,n=manifest(tmp_path,[p2,p3,method]);r=M.build_report(p,n,bootstrap_samples=50);m={x["case_id"]:x for x in r["cases"]}["p4m"]
    comp=m["comparison_to_ce"];assert comp["baseline_case_id"]=="p2ce"
    assert comp["metrics"]["exact_match"]["method_minus_ce"]["matched_examples"]==4
    assert comp["metrics"]["exact_match"]["delta_ndg"]["joint_four_way_examples"]==4
    assert m["exposure_sensitivity"]["comparison_to_ce"]["metrics"]["exact_match"]["method_minus_ce"]["difference"]==1/3
    assert m["analysis"]["posterior"]["entropy_defined_count"]==4

def test_real_caption_task_joint_defined_controls_and_pretrained_baseline(tmp_path):
    c=make_case(tmp_path,"cap","captioning","npp-opsd","nsd_captioning");images=("nsd33245","i2","i3","i4");names=[]
    for variant,ctl,values in (("trained","correct",(1,None,3,4)),("trained","shuffled",(0,2,2,4)),("pretrained","correct",(0,2,2,4)),("pretrained","shuffled",(0,1,2,None))):
        name=f"{variant}_{ctl}_repeat1"; pairs=[]
        for q,(i,v) in enumerate(zip(images,values)):
            raw,score=pred(i,q,v,"CIDEr",{"entropy":q if q else None});score["task"].update({"SPICE":None,"SPICE_object_precision":.5 if q==0 else None});pairs.append((raw,score))
        condition(c,name,pairs);names.append(name)
    finalize(c,names);p,n=manifest(tmp_path,[c]);r=M.build_report(p,n,bootstrap_samples=30)["cases"][0]
    assert {k:r["analysis"]["primary"]["CIDEr"][k] for k in ("mean","defined_count","undefined_count")}=={"mean":8/3,"defined_count":3,"undefined_count":1}
    assert r["analysis"]["ndg"]["CIDEr"]["matched_examples"]==3
    assert "comparison_to_ce" not in r
    assert r["comparison_to_pretrained"]["baseline_type"]=="pretrained"
    assert r["comparison_to_pretrained"]["metrics"]["CIDEr"]["trained_minus_pretrained"]["matched_examples"]==3
    assert r["comparison_to_pretrained"]["metrics"]["CIDEr"]["trained_minus_pretrained_delta_ndg"]["joint_four_way_examples"]==2
    assert r["analysis"]["primary"]["SPICE"]["defined_count"]==0
    assert r["analysis"]["posterior"]["entropy_undefined_count"]==1

def test_config_and_result_identity_rejected_and_outputs_written(tmp_path):
    c=make_case(tmp_path,"bad","P2","ce");Path(c["config"]).write_text("changed");condition(c,"trained_correct_repeat1",[pred(i,q,1) for q,i in enumerate(("nsd33245","i2","i3","i4"))]);finalize(c,["trained_correct_repeat1"])
    x=json.loads((Path(c["output_dir"])/"result.json").read_text());x["subject"]="subj02";put(Path(c["output_dir"])/"result.json",x);p,n=manifest(tmp_path,[c],2);report=M.build_report(p,n,bootstrap_samples=10)
    assert report["cases"][0]["status"]=="incomplete";assert "config_sha256_mismatch" in report["cases"][0]["validation_findings"]
    assert report["cases"][0]["scientific_identity_valid"] is False and report["seed_summary"]==[]
    j,m=tmp_path/"r.json",tmp_path/"r.md";M.write_report(report,j,m);assert json.loads(j.read_text())["coverage"]["complete"] is False;assert "partial" in m.read_text().lower() and "CHAIR" not in m.read_text()

def test_actual_runtime_shape_validates_128_125_complete_and_config_only_pending(tmp_path):
    complete=make_case(tmp_path,"complete","P2","ce");complete.update(expected_examples=256,expected_optimizer_updates=16,pretrained_baseline=True)
    trial_ids=[f"trial-{i}" for i in range(128)];images=[f"image-{i}" for i in range(128)]
    schedule={"schema_version":1,"seed":41,"trial_ids":[f"train-{i}" for i in range(256)],"accumulation_steps":16,"max_updates":16}
    schedule["schedule_hash"]=hashlib.sha256(json.dumps(schedule,sort_keys=True,separators=(",",":")).encode()).hexdigest();put(Path(complete["output_dir"])/"schedule.json",schedule)
    config=json.loads(Path(complete["config"]).read_text());config["experiment"].pop("output_dir");config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest();identity={"config":config_hash,"backend":BACKEND,"schedule":schedule["schedule_hash"],"schedule_hash":schedule["schedule_hash"],"method":"ce"};identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest();checkpoint=Path(complete["output_dir"])/"checkpoint.pt";put(Path(complete["output_dir"])/"checkpoint_identity.json",{"identity":identity,"identity_hash":identity_hash,"cursor":256,"optimizer_steps":16,"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    names=[];counts={};unmatched={}
    for variant in ("trained","pretrained"):
        for control in ("correct","shuffled","zero","covariance-noise","wrong-subject"):
            name=f"{variant}_{control}_repeat1";limit=125 if control=="shuffled" else 128;pairs=[]
            for i in range(limit):
                raw,score=pred(images[i],i,float(i%2),posterior={"entropy":.5,"coverage":False});raw["source_trial_ids"]=[trial_ids[i]];pairs.append((raw,score))
            condition(complete,name,pairs);names.append(name);counts[name.replace("_",":")]=limit;unmatched[name.replace("_",":")]=trial_ids[125:] if control=="shuffled" else []
        for prefix in (2,3):
            name=f"{variant}_correct_repeat{prefix}";pairs=[]
            for i in range(128):
                raw,score=pred(images[i],i,float(i%2));raw["source_trial_ids"]=[trial_ids[i]];pairs.append((raw,score))
            condition(complete,name,pairs);names.append(name);counts[name.replace("_",":")]=128;unmatched[name.replace("_",":")]=[]
    evaluation={"cohort_examples":128,"prediction_counts":counts,"unmatched":unmatched,"metrics":{}}
    result={"status":"completed_nsd_matrix_case","phase":"P2","method":"ce","subject":"subj01","training":{"optimizer_steps":16,"cursor":256,"method":"ce","history":[{"micro_examples":16} for _ in range(16)],"accounting":{"generated_tokens":256}},"evaluation":evaluation}
    put(Path(complete["output_dir"])/"result.json",result);put(Path(complete["output_dir"])/"evaluation/summary.json",evaluation)
    pending=make_case(tmp_path,"pending","P2","npp-opsd",complete=False)
    for name in ("schedule.json","checkpoint.pt","checkpoint_identity.json"):(Path(pending["output_dir"])/name).unlink()
    mp=tmp_path/"actual-manifest.json";put(mp,{"case_count":2,"cases":[complete,pending]});np=tmp_path/"actual-nsd.jsonl";put(np,[{"trial_id":t,"image_id":i,"subject_id":"subj01","session_id":"session01"} for t,i in zip(trial_ids,images)])
    report=M.build_report(mp,np,bootstrap_samples=10);by={r["case_id"]:r for r in report["cases"]}
    assert by["complete"]["status"]=="complete" and by["complete"]["validation_findings"]==[]
    assert by["complete"]["available_conditions"]["trained_shuffled_repeat1"]==125
    assert by["pending"]["status"]=="pending"
    assert by["complete"]["analysis"]["primary"]["exact_match"]["category_macro"]==.5
    assert by["complete"]["session_strata"]["session01"]["exact_match"]["defined_count"]==128
    assert len(by["complete"]["identities"])==128 and by["complete"]["compute"]["training_examples"]==256

def test_missing_sidecar_and_bad_schedule_never_enter_comparisons(tmp_path):
    ce=make_case(tmp_path,"ce","P2","ce",complete=False);method=make_case(tmp_path,"method","P4","npp-opsd",complete=False)
    rows=[pred(i,q,float(q%2)) for q,i in enumerate(("nsd33245","i2","i3","i4"))]
    condition(ce,"trained_correct_repeat1",rows);condition(method,"trained_correct_repeat1",rows,identity=False)
    schedule=Path(method["output_dir"])/"schedule.json";value=json.loads(schedule.read_text());value["schedule_hash"]="bad";put(schedule,value)
    p,n=manifest(tmp_path,[ce,method]);report=M.build_report(p,n,bootstrap_samples=10);by={r["case_id"]:r for r in report["cases"]}
    assert by["ce"]["scientific_identity_valid"] is True
    assert by["method"]["scientific_identity_valid"] is False
    assert by["method"]["available_metrics_verification"]=="unverified"
    assert "comparison_to_ce" not in by["method"] and report["seed_summary"]==[]
