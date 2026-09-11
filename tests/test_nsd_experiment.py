import json
from pathlib import Path

import torch
import pytest

from brain_npp.nsd_experiment import load_p3_trial_ids, run_matrix_config


class TinyBackend:
    def __init__(self):
        self.model=torch.nn.Linear(1,1,bias=False); self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.01)
    def training_trial_ids(self): return ("1","2","3")
    def loss(self,trial_id,method): return self.model(torch.tensor([float(trial_id[-1])])).square().mean()
    def evaluate(self,output_dir):
        Path(output_dir,"evaluated").write_text("yes"); return {"predictions":2}
    def selected_example_manifest(self,trial_ids,method):return [{"trial_id":x,"image_id":"i"+x,"question_id":x,"answer_id":x,"reference":"r","credit_negative_image_id":"n"} for x in trial_ids]


def test_run_matrix_config_executes_train_resume_and_evaluate(tmp_path):
    config={"backend":"vindex","task":"nsd_vqa","method":"ce","paths":{"budget":"budget"},
            "experiment":{"phase":"matrix","seed":41,"subject":"subj01","output_dir":str(tmp_path),
                          "max_examples":3,"max_optimizer_steps":2},
            "trainer":{"gradient_accumulation_steps":2}}
    result=run_matrix_config(config,backend_builder=lambda c:TinyBackend())
    assert result["training"]["optimizer_steps"]==2 and result["evaluation"]=={"predictions":2}
    assert (tmp_path/"schedule.json").exists() and (tmp_path/"checkpoint.pt").exists()
    assert (tmp_path/"selected_examples.json").exists()
    resumed=run_matrix_config(config,backend_builder=lambda c:TinyBackend())
    assert resumed["training"]["resumed"] is True
    selected=json.loads((tmp_path/"selected_examples.json").read_text());selected["examples"][0]["reference"]="tampered"
    (tmp_path/"selected_examples.json").write_text(json.dumps(selected))
    with pytest.raises(ValueError,match="selected-example"):run_matrix_config(config,backend_builder=lambda c:TinyBackend())

def test_run_matrix_rejects_unsupported_epochs(tmp_path):
    config={"method":"ce","experiment":{"phase":"matrix","epochs":2,"output_dir":str(tmp_path)},"trainer":{}}
    with pytest.raises(ValueError,match="epochs=1"):run_matrix_config(config,backend_builder=lambda c:TinyBackend())


def test_run_matrix_config_p3_consumes_complete_ten_percent_group_subset(tmp_path):
    class P3(TinyBackend):
        def training_trial_images(self): return {f"a{i}":"a" for i in range(2)}|{f"{g}1":g for g in "bcdefghij"}
    config={"backend":"vindex","task":"nsd_vqa","method":"ce","paths":{"budget":"budget"},
            "experiment":{"phase":"P3","seed":41,"subject":"subj01","output_dir":str(tmp_path),"image_group_fraction":.1},
            "trainer":{"gradient_accumulation_steps":16}}
    result=run_matrix_config(config,backend_builder=lambda c:P3())
    selected=json.loads((tmp_path/"schedule.json").read_text())["trial_ids"]
    assert result["training"]["optimizer_steps"]==1
    assert ("a1" in selected)==("a0" in selected)


def test_p3_loader_preserves_predeclared_order_and_validates_groups(tmp_path):
    import hashlib
    manifest=tmp_path/"m";manifest.write_text("manifest")
    payload={"schema_version":1,"seed":41,"image_fraction":.1,"manifest_sha256":hashlib.sha256(b"manifest").hexdigest(),
             "subjects":{"subj01":{"selected_image_ids":["i2","i1"],"trial_ids":["b","a"],"selected_trials":2,"optimizer_updates":1,"final_window_examples":2}}}
    path=tmp_path/"p3.json";path.write_text(json.dumps(payload))
    records={"a":("subj01","i1","train"),"b":("subj01","i2","train")}
    assert load_p3_trial_ids(path,subject="subj01",seed=41,manifest_path=manifest,records=records,accumulation=16)==("b","a")
    records["b"]=("subj01","other","train")
    with pytest.raises(ValueError,match="image group"):
        load_p3_trial_ids(path,subject="subj01",seed=41,manifest_path=manifest,records=records,accumulation=16)
