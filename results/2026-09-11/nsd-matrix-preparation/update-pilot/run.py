"""Task-3 engineering pilot: one real CE/NPP/DAPD update, no benchmark claims."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig,CLIPImageProcessor,CLIPVisionModel,LlamaConfig,LlamaForCausalLM,LlamaTokenizer
from brain_npp.adapters.vindex import rollout_mask
from brain_npp.baselines import dapd_loss,masked_cross_entropy
from brain_npp.nsd_data import FullCalibrationStore,FullNSDData
from brain_npp.nsd_encoder import load_brainx,encode_subject_brain
from brain_npp.objective import build_npp_target,forward_kl_loss
from brain_npp.posterior import build_neural_posterior

assets=Path('/root/brain-assets'); nsd=assets/'nsd-full'; out=Path('/root/brain/artifacts/verification/task3_method_pilot');out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(41);torch.cuda.manual_seed_all(41);torch.set_num_threads(8)
model_path=assets/'llava-v1.5-7b'
model=LlamaForCausalLM.from_pretrained(model_path,config=LlamaConfig(**json.loads((model_path/'config.json').read_text())),local_files_only=True,device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4'))
model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'));model.config.use_cache=False
tok=LlamaTokenizer.from_pretrained(model_path,local_files_only=True)
projector=torch.nn.Sequential(torch.nn.Linear(1024,4096),torch.nn.GELU(),torch.nn.Linear(4096,4096)).cuda()
pstate=torch.load(assets/'vindex/src/llava/model_weights/llava-v1.5-7b/mm_projector.bin',map_location='cpu',weights_only=True)
projector.load_state_dict({k.split('mm_projector.',1)[1]:v for k,v in pstate.items() if 'mm_projector.' in k},strict=True)
teacher_projector=__import__('copy').deepcopy(projector).requires_grad_(False).eval()
encoder=load_brainx(assets/'umbrae/train_logs/brainx/last.pth',model_source='/root/UMBRAE/umbrae/model.py',device='cuda').requires_grad_(False).eval()
clip=CLIPVisionModel.from_pretrained(assets/'clip-vit-large-patch14',local_files_only=True).cuda().requires_grad_(False).eval(); proc=CLIPImageProcessor.from_pretrained(assets/'clip-vit-large-patch14',local_files_only=True)
data=FullNSDData.load(nsd/'nsd_manifest.jsonl',features_path=nsd/'features/image_pooled.npy',image_ids_path=nsd/'features/image_ids.json',feature_provenance_path=nsd/'features/clip_provenance.json',vqa_paths={'train':nsd/'vqa_train.jsonl'},row_mapping_binding_path=nsd/'features/row_mapping_binding.v1.json')
calibration=FullCalibrationStore(nsd/'calibration',data); example=data.vqa_examples('train',subject_id='subj01')[0]; record=example.trial
brain=np.load(record.brain_path,allow_pickle=False); canonical={r['trial_id']:r['brain_array_sha256'] for r in map(json.loads,(nsd/'subj01.trial_provenance.jsonl').open())};assert hashlib.sha256(brain.tobytes()).hexdigest()==canonical[record.trial_id]
brain_features=encode_subject_brain(encoder,torch.from_numpy(brain[None]).cuda(),'subj01').detach()
prefix_text="A chat between a curious user and an artificial intelligence assistant. USER: "
instruction="Answer with a short phrase."
embed=model.get_input_embeddings()
def prompt(features,aux=None,teacher=False):
    pre=tok(prefix_text,return_tensors='pt').input_ids.cuda(); suffix='\n'+example.question+'\n'+instruction
    if aux is not None:suffix+='\nAuxiliary text: '+aux
    post=tok(suffix+' ASSISTANT:',add_special_tokens=False,return_tensors='pt').input_ids.cuda()
    proj=teacher_projector if teacher else projector
    return torch.cat((embed(pre),proj(features.to(proj[0].weight.dtype)).to(embed.weight.dtype),embed(post)),1)
def score(features,tokens,aux=None,base=False,snapshot=False):
    context=model.disable_adapter() if base else __import__('contextlib').nullcontext()
    with context:
        p=prompt(features,aux,teacher=base); t=embed(tokens); logits=model(inputs_embeds=torch.cat((p,t),1),use_cache=False).logits
    return logits[:,p.shape[1]-1:p.shape[1]-1+tokens.shape[1]]
def rollout():
    p=prompt(brain_features);model.eval()
    with torch.no_grad(): ids=model.generate(inputs_embeds=p,attention_mask=torch.ones(p.shape[:2],device='cuda',dtype=torch.long),max_new_tokens=64,do_sample=True,temperature=1.0,top_p=1.0,pad_token_id=tok.eos_token_id)
    return ids,rollout_mask(ids,tok.eos_token_id)
reference=tok(example.answers[0],add_special_tokens=False,return_tensors='pt').input_ids.cuda();reference=torch.cat((reference,torch.tensor([[tok.eos_token_id]],device='cuda')),1);refmask=rollout_mask(reference,tok.eos_token_id)
trainable=[p for p in projector.parameters()]+[p for p in model.parameters() if p.requires_grad]
initial={str(i):p.detach().cpu().clone() for i,p in enumerate(trainable)}
def reset():
    with torch.no_grad():
        for i,p in enumerate(trainable):p.copy_(initial[str(i)].to(p.device,p.dtype))
def image_patches(image_id):
    image=Image.open(nsd/'images'/f'{image_id}.jpg').convert('RGB'); pixels=proc(images=image,return_tensors='pt').pixel_values.cuda()
    with torch.no_grad():return clip(pixel_values=pixels,output_hidden_states=True).hidden_states[-2][:,1:]
results=[]
for method in ('ce','npp','dapd'):
    reset();optimizer=torch.optim.AdamW(trainable,lr=5e-6);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    if method=='ce':
        live=score(brain_features,reference);loss=masked_cross_entropy(live,reference,refmask);views=1;generated=0
    elif method=='npp':
        ids,mask=rollout();gallery=data.sample_gallery(subject_id='subj01',experiment_seed=41,data_cursor=0,count=4);pooled=data.image_features(gallery.image_ids);scores=torch.from_numpy(calibration.score(record,brain[None],pooled)).cuda();prior=torch.tensor([gallery.log_prior],device='cuda')
        teachers=[]
        for image_id in gallery.image_ids:
            with torch.no_grad():teachers.append(score(image_patches(image_id),ids,base=True))
        teachers=torch.stack(teachers,2);live=score(brain_features,ids);posterior=build_neural_posterior(scores.float(),prior.float());target=build_npp_target(teachers.float(),live.detach().float(),posterior);loss=forward_kl_loss(target.log_probabilities,live.float(),mask).loss;views=5;generated=int(mask.sum())
    else:
        ids,mask=rollout();rolltext=tok.decode(ids[0],skip_special_tokens=True);reftext=example.answers[0]
        live={'rollout_none':score(brain_features,ids),'reference_reference':score(brain_features,reference,reftext),'rollout_rollout':score(brain_features,ids,rolltext),'reference_none':score(brain_features,reference)}
        with torch.no_grad():
            anchors={'entangled_rollout':score(brain_features,ids,reftext,snapshot=True),'inference_reference':score(brain_features,reference,base=True),'privileged_rollout':score(brain_features,ids,reftext,base=True),'entangled_reference':score(brain_features,reference,rolltext,snapshot=True),'inference_rollout':score(brain_features,ids,base=True),'privileged_reference':score(brain_features,reference,rolltext,base=True)}
        loss=dapd_loss(live,anchors,{'rollout':mask,'reference':refmask}).loss;views=10;generated=int(mask.sum())
    loss.backward();gnorm=float(torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in trainable if p.grad is not None])));optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
    delta=float(torch.linalg.vector_norm(torch.stack([(p.detach().cpu()-initial[str(i)]).float().norm() for i,p in enumerate(trainable)])))
    row={'method':method,'seconds':elapsed,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'loss':float(loss.detach()),'gradient_norm':gnorm,'delta_norm':delta,'sequence_views':views,'generated_tokens':generated};assert all(np.isfinite(row[k]) and row[k]>0 for k in ('seconds','loss','gradient_norm','delta_norm'));results.append(row);print(row,flush=True)
(out/'report.json').write_text(json.dumps({'purpose':'real one-update engineering pilot, not comparison','trial_id':record.trial_id,'question_id':example.question_id,'prompt_instruction':instruction,'results':results},indent=2)+'\n')
