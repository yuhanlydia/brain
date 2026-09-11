"""Real frozen brain-only inference pilot; no comparative training outcomes."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import BitsAndBytesConfig,LlamaConfig,LlamaForCausalLM,LlamaTokenizer
from brain_npp.nsd_encoder import load_brainx,encode_subject_brain
from brain_npp.data import load_jsonl_manifest
from brain_npp.adapters.vindex import VindexP1Adapter

root=Path('/root/brain-assets');nsd=root/'nsd-full';out=Path('/root/brain/artifacts/verification/pretrained_inference_pilot');out.mkdir(exist_ok=True)
torch.set_num_threads(8);torch.manual_seed(41)
cohort=json.loads((nsd/'matrix/evaluation_cohort.json').read_text());records={r.trial_id:r for r in load_jsonl_manifest(nsd/'nsd_manifest.jsonl')}
qa={(r['question_id'],r['answer_id']):r for r in map(json.loads,(nsd/'vqa_test.jsonl').open())}
model_path=root/'llava-v1.5-7b';start=time.perf_counter()
model=LlamaForCausalLM.from_pretrained(model_path,config=LlamaConfig(**json.loads((model_path/'config.json').read_text())),local_files_only=True,device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4'))
model.requires_grad_(False);model.eval();model.config.use_cache=True
tokenizer=LlamaTokenizer.from_pretrained(model_path,local_files_only=True)
projector=torch.nn.Sequential(torch.nn.Linear(1024,4096),torch.nn.GELU(),torch.nn.Linear(4096,4096)).cuda()
state=torch.load(root/'vindex/src/llava/model_weights/llava-v1.5-7b/mm_projector.bin',map_location='cpu',weights_only=True)
projector.load_state_dict({k.split('mm_projector.',1)[1]:v for k,v in state.items() if 'mm_projector.' in k},strict=True);projector.requires_grad_(False);projector.eval()
encoder=load_brainx(root/'umbrae/train_logs/brainx/last.pth',model_source='/root/UMBRAE/umbrae/model.py',device='cuda');encoder.requires_grad_(False);encoder.eval()
print('MODELS_READY',time.perf_counter()-start,flush=True)
prefix="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
results=[]
for subject in ['subj01','subj02','subj05','subj07']:
    canonical={r['trial_id']:r['brain_array_sha256'] for r in map(json.loads,(nsd/f'{subject}.trial_provenance.jsonl').open())}
    for example in cohort['examples'][:2]:
        trial_id=example['trial_ids_by_subject'][subject][0];record=records[trial_id];annotation=qa[(example['question_id'],example['answer_id'])]
        brain=np.load(record.brain_path,allow_pickle=False)
        assert hashlib.sha256(brain.tobytes()).hexdigest()==canonical[trial_id]
        pre=tokenizer(prefix,return_tensors='pt').input_ids
        post=tokenizer('\n'+annotation['question']+' ASSISTANT:',add_special_tokens=False,return_tensors='pt').input_ids
        adapter=VindexP1Adapter(model,tokenizer,projector,projector,pre,post,'cuda')
        torch.cuda.synchronize();begin=time.perf_counter()
        with torch.no_grad():
            features=encode_subject_brain(encoder,torch.from_numpy(brain[None]).cuda(),subject)
            rollout=adapter.generate_student({'brain_features':features},{'max_new_tokens':64,'do_sample':False,'use_cache':True})
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        text=tokenizer.decode(rollout.token_ids[0],skip_special_tokens=True)
        result={'subject':subject,'trial_id':trial_id,'image_id':example['image_id'],'question_id':example['question_id'],'answer_id':example['answer_id'],'question':annotation['question'],'references':annotation['answers'],'prediction':text,'token_ids':rollout.token_ids[0].tolist(),'active_tokens':int(rollout.token_mask.sum()),'seconds':elapsed,'brain_array_sha256':canonical[trial_id]}
        results.append(result);(out/'predictions.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
report={'purpose':'first2predeclaredcohort images x4subjects; frozen originalweights; engineering inference pilot, not fullbenchmark','cohort_sha256':hashlib.sha256((nsd/'matrix/evaluation_cohort.json').read_bytes()).hexdigest(),'generation':{'max_new_tokens':64,'do_sample':False,'use_cache':True},'torch':torch.__version__,'cases':len(results),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'mean_seconds':sum(r['seconds'] for r in results)/len(results),'encoder_provenance':getattr(encoder,'brain_npp_provenance',None)}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',json.dumps(report),flush=True)
