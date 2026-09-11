"""Runnable real BrainX/NF4-LLaVA backend for the NSD experiment matrix."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy, hashlib, json, math, random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from brain_npp.adapters.nsd_vindex import PatchCache, ParameterContexts
from brain_npp.adapters.vindex import rollout_mask
from brain_npp.nsd_data import FullCalibrationStore, FullNSDData
from brain_npp.nsd_encoder import encode_subject_brain, load_brainx
from brain_npp.nsd_evaluation import VQA_NORMALIZATION_POLICY, caption_scores, load_evaluation_cohort, score_candidate_posterior, score_vqa_predictions
from brain_npp.nsd_runner import (ComputeAccounting, Method, PredictionJournal,
    credit_negative_mapping, method_loss, select_question_ids,
    validate_brain_array, validate_control_artifact)
from brain_npp.provenance import sha256_file, sha256_tree


class _TrainableContainer(torch.nn.Module):
    def __init__(self,llm,projector):super().__init__();self.llm=llm;self.projector=projector


class RealNSDBackend:
    """Streams real examples and constructs each method's exact fixed-token views."""
    def __init__(self,config:Mapping[str,Any]):
        self.config=dict(config);self.paths=_mapping(config.get('paths'),'paths');self.experiment=_mapping(config.get('experiment'),'experiment')
        self.subject=str(self.experiment.get('subject'));self.seed=int(self.experiment.get('seed'))
        if self.subject not in {'subj01','subj02','subj05','subj07'}:raise ValueError('experiment.subject must select one supported subject')
        random.seed(self.seed);np.random.seed(self.seed);torch.manual_seed(self.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(self.seed)
        self.task=str(config.get('task'));self.generation=dict(_mapping(config.get('generation',{}),'generation'))
        _validate_vad_generation(str(config.get('method')),self.generation)
        self.accounting=ComputeAccounting();self._load_data();self._load_models();self._prepare_training()

    def _load_data(self):
        p=self.paths
        self.data=FullNSDData.load(p['manifest'],features_path=p['image_features'],image_ids_path=p['image_ids'],feature_provenance_path=p['feature_provenance'],
            vqa_paths={'train':p['vqa_train'],'test':p['vqa_test']},caption_paths={'train':p['captions_train'],'test':p['captions_test']},row_mapping_binding_path=p['row_mapping_binding'])
        self.records={r.trial_id:r for r in self.data.records};self.calibration=FullCalibrationStore(p['calibration'],self.data)
        self.image_paths={r.image_id:r.image_path for r in self.data.records}
        self.canonical={r['trial_id']:r['brain_array_sha256'] for r in map(json.loads,Path(p.get('trial_provenance',Path(p['data'])/f'{self.subject}.trial_provenance.jsonl')).open())}
        self.all_canonical={s:{r['trial_id']:r['brain_array_sha256'] for r in map(json.loads,(Path(p['data'])/f'{s}.trial_provenance.jsonl').open())} for s in ('subj01','subj02','subj05','subj07')}
        valid={r.trial_id:(r.subject_id,r.image_id,r.repeat_id) for r in self.data.records}
        self.cohort=load_evaluation_cohort(p['evaluation_cohort'],manifest_path=p['manifest'],vqa_path=p['vqa_test'],caption_path=p['captions_test'],valid_trials=valid)
        selected=[x for e in self.cohort.for_subject(self.subject,repeat_prefix=3) for x in e.trial_ids]
        tuples={key:(r.subject_id,r.image_id,r.repeat_id) for key,r in self.records.items()}
        self.shuffle=validate_control_artifact(p['shuffled_control'],expected_control='shuffled',manifest_sha256=self.data.provenance['manifest_sha256'],records=tuples,selected_source_ids=selected)
        self.wrong=validate_control_artifact(p['wrong_subject_control'],expected_control='wrong-subject',manifest_sha256=self.data.provenance['manifest_sha256'],records=tuples,selected_source_ids=selected)
        self.qa_rows={(int(x['question_id']),int(x['answer_id'])):x for x in map(json.loads,Path(p['vqa_test']).open())}
        self.caption_rows={}
        for x in self.data.caption_examples('test',subject_id=self.subject):self.caption_rows.setdefault(x.trial.image_id,[r.caption for r in x.references])

    def _load_models(self):
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig,CLIPImageProcessor,CLIPVisionModel,LlamaConfig,LlamaForCausalLM,LlamaTokenizer
        model_path=Path(self.paths['model']);device=self.experiment.get('device','cuda')
        llm=LlamaForCausalLM.from_pretrained(model_path,config=LlamaConfig(**json.loads((model_path/'config.json').read_text())),local_files_only=True,device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4'))
        llm=prepare_model_for_kbit_training(llm,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        self.llm=get_peft_model(llm,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'));self.llm.config.use_cache=False
        self.tokenizer=LlamaTokenizer.from_pretrained(model_path,local_files_only=True)
        self.projector=torch.nn.Sequential(torch.nn.Linear(1024,4096),torch.nn.GELU(),torch.nn.Linear(4096,4096)).to(device)
        state=torch.load(self.paths['projector'],map_location='cpu',weights_only=True);self.projector.load_state_dict({k.split('mm_projector.',1)[1]:v for k,v in state.items() if 'mm_projector.' in k},strict=True)
        self.teacher_projector=copy.deepcopy(self.projector).requires_grad_(False).eval()
        self.brainx=load_brainx(self.paths['brainx_checkpoint'],model_source=self.paths['brainx_source'],device=device)
        clip_path=self.paths.get('clip_model',str(Path(self.paths['data']).parent/'clip-vit-large-patch14'))
        self.clip=CLIPVisionModel.from_pretrained(clip_path,local_files_only=True).to(device).requires_grad_(False).eval();self.image_processor=CLIPImageProcessor.from_pretrained(clip_path,local_files_only=True)
        self.device=torch.device(device);self.model=_TrainableContainer(self.llm,self.projector)
        trainable=dict((n,p) for n,p in self.model.named_parameters() if p.requires_grad);projector_names={n for n in trainable if n.startswith('projector.')}
        self.contexts=ParameterContexts(trainable,projector_names=projector_names,adapter_context=self.llm.disable_adapter);self.contexts.capture_snapshot(update_index=0)
        opt=_mapping(self.config.get('optimizer',{}),'optimizer');self.optimizer=torch.optim.AdamW(trainable.values(),lr=float(opt.get('lr',5e-6)),weight_decay=float(opt.get('weight_decay',0)))
        self._initial={n:p.detach().cpu().clone() for n,p in trainable.items()};self._trainable=trainable
        cache=_mapping(self.config.get('cache',{}),'cache');clip_hash=sha256_tree(clip_path);pre_hash=hashlib.sha256(json.dumps(self.image_processor.to_dict(),sort_keys=True).encode()).hexdigest()
        self.patch_cache=PatchCache(cache['clip_patch_dir'],max_bytes=int(cache['clip_patch_max_bytes']),model_hash=clip_hash,preprocessing_hash=pre_hash)
        material={name:sha256_file(self.paths[name]) for name in ('manifest','image_features','image_ids','feature_provenance','row_mapping_binding','vqa_train','vqa_test','captions_train','captions_test','evaluation_cohort','shuffled_control','wrong_subject_control')}
        for name in ('budget','training_cohort'):
            if self.paths.get(name) and Path(self.paths[name]).is_file():material[name]=sha256_file(self.paths[name])
        noise_root=Path(self.paths['cached_noise_dir']);material['noise_provenance']=sha256_file(noise_root/f'{self.subject}.provenance.json');material['noise_array']=sha256_file(noise_root/f'{self.subject}_seed{self.seed}.npy')
        source_root=Path(__file__).parents[1];names=('baselines.py','objective.py','nsd_runner.py','nsd_evaluation.py','nsd_experiment.py','posterior.py','nsd_data.py','nsd_encoder.py','data.py')
        code={name:sha256_file(source_root/name) for name in names if (source_root/name).is_file()};code['adapters/nsd_backend.py']=sha256_file(__file__);code['adapters/nsd_vindex.py']=sha256_file(Path(__file__).with_name('nsd_vindex.py'));code['adapters/vindex.py']=sha256_file(Path(__file__).with_name('vindex.py'))
        factory=Path(__file__).parents[3]/'integrations/vindex/brain_npp_factory.py'
        if factory.is_file():code['integrations/vindex/brain_npp_factory.py']=sha256_file(factory)
        self.provenance={'model_sha256':sha256_tree(model_path),'projector_sha256':sha256_file(self.paths['projector']),'brainx':self.brainx.checkpoint_provenance,'clip_sha256':clip_hash,'cohort_sha256':self.cohort.sha256,'data':material,'calibration_sha256':sha256_tree(self.paths['calibration']),'code':code,'normalization_policy':VQA_NORMALIZATION_POLICY,'vad_importance_policy':'on-policy-unit' if str(self.config.get('method'))==Method.VAD.value else None}
        self.provenance_hash=hashlib.sha256(json.dumps(self.provenance,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def _prepare_training(self):
        examples=self.data.vqa_examples('train',subject_id=self.subject) if self.task=='nsd_vqa' else self.data.caption_examples('train',subject_id=self.subject)
        grouped={}
        for e in examples:grouped.setdefault(e.trial.trial_id,[]).append(e)
        if self.task=='nsd_vqa':
            chosen=select_question_ids({k:[e.question_id for e in v] for k,v in grouped.items()},seed=self.seed)
            self.training={k:next(e for e in v if e.question_id==chosen[k]) for k,v in grouped.items()}
        else:
            self.training={k:v[0] for k,v in grouped.items()}
            self.caption_reference={k:int.from_bytes(hashlib.sha256(f'caption-v1:{self.seed}:{k}'.encode()).digest()[:8],'big')%len(v[0].references) for k,v in grouped.items()}
        ordered=sorted(self.training);images=[self.training[x].trial.image_id for x in ordered];mapping=credit_negative_mapping(images,seed=self.seed)
        self.credit_negative={ordered[i]:self.training[ordered[j]].trial.image_id for i,j in mapping.items()}

    def training_trial_ids(self):return tuple(sorted(self.training))
    def training_trial_images(self):return {k:v.trial.image_id for k,v in self.training.items()}
    def selected_example_manifest(self,trial_ids,method):
        rows=[]
        for trial_id in trial_ids:
            example=self.training[trial_id]
            row={'trial_id':trial_id,'image_id':example.trial.image_id,'reference':self._reference(example),'credit_negative_image_id':self.credit_negative[trial_id]}
            if self.task=='nsd_vqa':row.update(question_id=example.question_id,answer_id=example.answer_id,question=example.question)
            else:row.update(caption_reference_index=self.caption_reference[trial_id])
            if method is Method.VAD:row['vad_importance']=1.0
            rows.append(row)
        return rows
    def selected_view_policy(self,method):
        policy={"method":method.value,"rollout_reused_across_views":True}
        if method is Method.VAD:policy.update(clear_view="original RGB",degraded_view="PIL bilinear downsample then nearest upsample",importance=1.0,sampling="unwarped on-policy")
        if method is Method.DAPD:policy.update(live_views=4,detached_anchor_views=6,snapshot_interval_updates=100)
        return policy

    def _brain(self,record):return validate_brain_array(record.brain_path,expected_sha256=self.canonical[record.trial_id]).float()
    def _brain_features(self,brain,subject):
        with torch.no_grad():return encode_subject_brain(self.brainx,brain[None].to(self.device),subject).detach()
    def _question(self,example):return example.question if self.task=='nsd_vqa' else 'Describe the image.'
    def _reference(self,example):return example.answers[0] if self.task=='nsd_vqa' else example.references[self.caption_reference[example.trial.trial_id]].caption
    def _tokens(self,text):
        ids=self.tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
        return torch.cat((ids,torch.tensor([[self.tokenizer.eos_token_id]],device=self.device)),1)
    def _prompt(self,features,question,condition=None,projector=None):
        prefix='A chat between a curious user and an artificial intelligence assistant. USER: '
        instruction=self.generation.get('vqa_instruction','Answer with a short phrase.') if self.task=='nsd_vqa' else 'Provide a concise caption.'
        suffix='\n'+question+'\n'+instruction+(('\nAuxiliary text: '+condition) if condition is not None else '')+' ASSISTANT:'
        emb=self.llm.get_input_embeddings();pre=self.tokenizer(prefix,return_tensors='pt').input_ids.to(self.device);post=self.tokenizer(suffix,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
        projector=projector or self.projector
        return torch.cat((emb(pre),projector(features.to(next(projector.parameters()).dtype)).to(emb.weight.dtype),emb(post)),1)
    def _score(self,features,tokens,question,condition=None,context='live'):
        self.llm.train(context=='live')
        if context=='base':scope=self.contexts.base();projector=self.projector
        elif context=='snapshot':scope=self.contexts.snapshot();projector=self.projector
        else:scope=nullcontext();projector=self.projector
        with scoring_grad_context(context):
          with scope:
            prompt=self._prompt(features,question,condition,projector);embedded=self.llm.get_input_embeddings()(tokens)
            logits=self.llm(inputs_embeds=torch.cat((prompt,embedded),1),use_cache=False).logits
        self.accounting.physical_forwards+=1
        return logits[:,prompt.shape[1]-1:prompt.shape[1]-1+tokens.shape[1]]
    def _generate(self,features,question,*,evaluation=False):
        prompt=self._prompt(features,question);options={k:v for k,v in self.generation.items() if k in {'max_new_tokens','do_sample','temperature','top_p','top_k'}};options.setdefault('max_new_tokens',64);options.setdefault('do_sample',True);options['pad_token_id']=self.tokenizer.eos_token_id
        if evaluation:options['do_sample']=False;options.pop('temperature',None);options.pop('top_p',None);options.pop('top_k',None)
        self.llm.eval()
        with torch.no_grad():ids=self.llm.generate(inputs_embeds=prompt,attention_mask=torch.ones(prompt.shape[:2],dtype=torch.long,device=self.device),use_cache=True,**options)
        mask=rollout_mask(ids,self.tokenizer.eos_token_id);count=int(mask.sum());self.accounting.generated_tokens+=count;self.accounting.physical_forwards+=count
        return ids,mask
    def _patches(self,image_id,view='clear'):
        from PIL import Image
        path=Path(self.image_paths[image_id]);image_hash=sha256_file(path)
        def compute():
            image=Image.open(path).convert('RGB')
            if view=='degraded':
                width,height=image.size;reduced=(max(1,round(width*.10)),max(1,round(height*.10)))
                image=image.resize(reduced,resample=Image.Resampling.BILINEAR).resize((width,height),resample=Image.Resampling.NEAREST)
            pixels=self.image_processor(images=image,return_tensors='pt').pixel_values.to(self.device)
            with torch.no_grad():return self.clip(pixel_values=pixels,output_hidden_states=True).hidden_states[-2][:,1:]
        return self.patch_cache.get_or_compute(image_id,view,image_hash,compute).to(self.device)

    def loss(self,trial_id,method):
        e=self.training[trial_id];r=e.trial;q=self._question(e);reference_text=self._reference(e);reference=self._tokens(reference_text);refmask=rollout_mask(reference,self.tokenizer.eos_token_id)
        brain=self._brain(r);features=self._brain_features(brain,self.subject);common={'gold_tokens':reference,'teacher_probabilities':torch.empty(0),'posterior_weights':torch.empty(0),'prior_weights':torch.empty(0)}
        if method is Method.CE:
            live=self._score(features,reference,q);self.accounting.add(student_tokens=int(refmask.sum()),sequence_views=1);return method_loss(method,student_logits=live,mask=refmask,**common)
        ids,mask=self._generate(features,q);common['gold_tokens']=ids;y=int(mask.sum());rlen=int(refmask.sum());self.accounting.add(student_tokens=y)
        if method is Method.CREDIT:
            positive=torch.softmax(self._score(self._patches(r.image_id),ids,q,context='base').float(),-1);negative=torch.softmax(self._score(self._patches(self.credit_negative[trial_id]),ids,q,context='base').float(),-1)
            live=self._score(features,ids,q)
            self.accounting.add(teacher_tokens=2*y,sequence_views=3);return method_loss(method,student_logits=live.float(),mask=mask,positive_teacher_probabilities=positive,negative_teacher_probabilities=negative,**common)
        if method is Method.VAD:
            clear=torch.softmax(self._score(self._patches(r.image_id),ids,q,context='base').float(),-1);degraded=torch.softmax(self._score(self._patches(r.image_id,'degraded'),ids,q,context='base').float(),-1);live=self._score(features,ids,q);sampled=torch.log_softmax(live.detach().float(),-1).gather(-1,ids[...,None]).squeeze(-1)
            self.accounting.add(teacher_tokens=2*y,sequence_views=3);return method_loss(method,student_logits=live.float(),mask=mask,clear_teacher_probabilities=clear,degraded_teacher_probabilities=degraded,sampled_log_probs=sampled,old_log_probs=sampled.clone(),**common)
        if method is Method.DAPD:
            rollout_text=self.tokenizer.decode(ids[0],skip_special_tokens=True)
            with torch.no_grad():anchors={'entangled_rollout':self._score(features,ids,q,reference_text,'snapshot').float(),'inference_reference':self._score(features,reference,q,context='base').float(),'privileged_rollout':self._score(features,ids,q,reference_text,'base').float(),'entangled_reference':self._score(features,reference,q,rollout_text,'snapshot').float(),'inference_rollout':self._score(features,ids,q,context='base').float(),'privileged_reference':self._score(features,reference,q,rollout_text,'base').float()}
            live_views={'rollout_none':self._score(features,ids,q).float(),'reference_reference':self._score(features,reference,q,reference_text).float(),'rollout_rollout':self._score(features,ids,q,rollout_text).float(),'reference_none':self._score(features,reference,q).float()};live=live_views['rollout_none']
            self.accounting.add(student_tokens=y+2*rlen,teacher_tokens=3*y+3*rlen,sequence_views=10);return method_loss(method,student_logits=live.float(),mask=mask,live_logits=live_views,anchor_logits=anchors,completion_masks={'rollout':mask,'reference':refmask},**common)
        if method is Method.EXACT:
            exact_teacher=torch.softmax(self._score(self._patches(r.image_id),ids,q,context='base').float(),-1);teachers=exact_teacher[:,:,None].expand(-1,-1,4,-1)
            scores=prior=posterior=torch.zeros((1,4),device=self.device);exact=0;self.accounting.add(teacher_tokens=y,sequence_views=2)
        else:
            gallery=self.data.sample_gallery(subject_id=self.subject,experiment_seed=self.seed,data_cursor=self.training_trial_ids().index(trial_id),count=int(self.config.get('candidate_count',4)))
            teacher_logits=torch.stack([self._score(self._patches(x),ids,q,context='base').float() for x in gallery.image_ids],2);scores=torch.from_numpy(self.calibration.score(r,brain[None].numpy(),self.data.image_features(gallery.image_ids))).to(self.device).float();prior=torch.tensor([gallery.log_prior],device=self.device).float();posterior=torch.softmax(scores+prior,-1);teachers=torch.softmax(teacher_logits,-1);exact=0;self.accounting.add(teacher_tokens=4*y,sequence_views=5)
        live=self._score(features,ids,q)
        return method_loss(method,student_logits=live.float(),mask=mask,teacher_probabilities=teachers,posterior_weights=posterior,prior_weights=prior.exp(),exact_index=exact,candidate_scores=scores,log_prior=prior,gold_tokens=ids)

    @contextmanager
    def _pretrained(self):
        current={n:p.detach().cpu().clone() for n,p in self._trainable.items()}
        try:
            with torch.no_grad():
                for n,p in self._trainable.items():p.copy_(self._initial[n].to(p.device,p.dtype))
            yield
        finally:
            with torch.no_grad():
                for n,p in self._trainable.items():p.copy_(current[n].to(p.device,p.dtype))

    def evaluate(self,output_dir):
        output=Path(output_dir)/'evaluation';output.mkdir(parents=True,exist_ok=True);cfg=_mapping(self.config.get('evaluation',{}),'evaluation');limit=int(cfg.get('max_examples',len(self.cohort.examples)));counts={};unmatched={};metrics={}
        variants=('trained','pretrained') if cfg.get('pretrained_student_baseline',True) else ('trained',)
        for variant in variants:
            scope=self._pretrained() if variant=='pretrained' else nullcontext()
            with scope:
              for repeat_prefix in cfg.get('diagnostic_repeat_prefixes',[int(cfg.get('main_repeat_prefix',1))]):
                repeat_prefix=int(repeat_prefix)
                controls=('correct','shuffled','zero','covariance-noise','wrong-subject') if repeat_prefix==int(cfg.get('main_repeat_prefix',1)) else ('correct',)
                for control in controls:
                    key=f'{variant}:{control}:repeat{repeat_prefix}';journal=PredictionJournal(output/f'{variant}_{control}_repeat{repeat_prefix}.jsonl',identity={'cohort':self.cohort.sha256,'backend_provenance':self.provenance_hash,'variant':variant,'control':control,'subject':self.subject,'seed':self.seed,'repeat_prefix':repeat_prefix,'normalization_policy':VQA_NORMALIZATION_POLICY if self.task=='nsd_vqa' else 'canonical-coco-scorers-v1'})
                    control_mapping=self.shuffle.mapping if control=='shuffled' else (self.wrong.mapping if control=='wrong-subject' else None)
                    missing=([source for cohort_row in self.cohort.examples[:limit] for source in cohort_row['trial_ids_by_subject'][self.subject][:repeat_prefix] if source not in control_mapping] if control_mapping is not None else [])
                    for row_index,cohort_row in enumerate(self.cohort.examples[:limit]):
                        example_id=f"{cohort_row['image_id']}:{cohort_row['question_id']}"
                        if not pending_example(cohort_row,journal.completed_ids):continue
                        trial_ids=cohort_row['trial_ids_by_subject'][self.subject][:repeat_prefix];brains=[];source_subject=self.subject;source_trials=[]
                        for source in trial_ids:
                            target=source
                            if control=='shuffled':
                                if source not in self.shuffle.mapping:continue
                                target=self.shuffle.mapping[source]
                            elif control=='wrong-subject':
                                if source not in self.wrong.mapping:continue
                                target=self.wrong.mapping[source];source_subject=self.records[target].subject_id
                            source_trials.append(target);brains.append(self._brain_for_any(target))
                        if control in {'shuffled','wrong-subject'} and len(brains)!=len(trial_ids):continue
                        brain=torch.stack(brains).mean(0)
                        if control=='zero':brain=torch.zeros_like(brain)
                        if control=='covariance-noise':brain=self._noise(row_index)
                        features=self._brain_features(brain,source_subject)
                        if self.task=='nsd_vqa':qa=self.qa_rows[(int(cohort_row['question_id']),int(cohort_row['answer_id']))];question=qa['question'];references=qa['answers']
                        else:question='Describe the image.';references=self.caption_rows[cohort_row['image_id']]
                        ids,mask=self._generate(features,question,evaluation=True);text=self.tokenizer.decode(ids[0][mask[0]],skip_special_tokens=True)
                        gallery=self.data.sample_gallery(subject_id=self.subject,experiment_seed=self.seed,data_cursor=row_index,count=int(self.config.get('candidate_count',4)));candidate_features=self.data.image_features(gallery.image_ids)
                        scoring_record=self.records[source_trials[0]] if control=='wrong-subject' else self.records[trial_ids[0]]
                        probabilities=posterior_probabilities(self.calibration,scoring_record,brain[None].numpy(),candidate_features,gallery.log_prior);posterior_policy='subject-own-gaussian-likelihood'
                        journal.append({'example_id':f"{cohort_row['image_id']}:{cohort_row['question_id']}",'source_image_id':cohort_row['image_id'],'image_id':cohort_row['image_id'],'question_id':cohort_row['question_id'],'answer_id':cohort_row['answer_id'],'category':cohort_row['category'],'original_question':question,'references':references,'token_ids':ids[0].tolist(),'text':text,'prediction':text,'control':control,'variant':variant,'repeat_prefix':repeat_prefix,'source_trial_ids':list(trial_ids),'resolved_trial_ids':source_trials,'brain_subject_id':source_subject,'candidate_ids':list(gallery.image_ids),'probabilities':probabilities,'posterior_policy':posterior_policy,'gallery_coverage':gallery.coverage(cohort_row['image_id'])})
                    counts[key]=len(journal.completed_ids);unmatched[key]=sorted(set(missing))
                    rows=list(journal.rows.values())
                    if rows:
                        task_metrics=score_vqa_predictions(rows) if self.task=='nsd_vqa' else self._caption_metrics(rows)
                        posterior_metrics=score_candidate_posterior(rows);task_rows=task_metrics.pop('per_example',[]);posterior_rows=posterior_metrics.pop('per_example',[])
                        score_rows=[]
                        for source_row,task_row,posterior_row in zip(rows,task_rows,posterior_rows):
                            task_values=({k:task_row[k] for k in ('exact_match','normalized_prediction','normalized_references') if k in task_row} if self.task=='nsd_vqa' else {k:v for k,v in task_row.items() if k!='example_id'})
                            score_rows.append({'example_id':source_row['example_id'],'image_id':source_row['image_id'],'task':task_values,'posterior':{k:posterior_row[k] for k in ('entropy','coverage','true_target_nll','brier')}})
                        (output/f'{variant}_{control}_repeat{repeat_prefix}.scores.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in score_rows))
                        metrics[key]={'task':task_metrics,'posterior':posterior_metrics}
        summary={'prediction_counts':counts,'unmatched':unmatched,'metrics':metrics,'cohort_examples':min(limit,len(self.cohort.examples)),'repeat_policy':'all controls at main prefix; correct-brain mean only at prefixes2/3; repeated-mean posterior entropy is conditional under the fixed single-trial likelihood'};(output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n');return summary

    def _caption_metrics(self,rows):
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice
        raw_refs={i:[{'caption':x} for x in row['references']] for i,row in enumerate(rows)};raw_hyp={i:[{'caption':row['prediction']}] for i,row in enumerate(rows)};tokenizer=PTBTokenizer();refs=tokenizer.tokenize(raw_refs);hyp=tokenizer.tokenize(raw_hyp)
        return caption_scores(refs,hyp,spice_scorer=Spice(),standard_scorers=[(Cider(),'CIDEr'),(Meteor(),'METEOR'),(Rouge(),'ROUGE_L')])

    def _brain_for_any(self,trial_id):
        r=self.records[trial_id]
        return validate_brain_array(r.brain_path,expected_sha256=self.all_canonical[r.subject_id][trial_id]).float()
    def _noise(self,row):
        root=Path(self.paths['cached_noise_dir']);provenance=json.loads((root/f'{self.subject}.provenance.json').read_text());path=root/f'{self.subject}_seed{self.seed}.npy'
        if sha256_file(path)!=provenance['outputs'][str(self.seed)]['sha256'] or provenance['identity']['cohort_sha256']!=self.cohort.sha256:raise ValueError('cached noise provenance mismatch')
        return torch.from_numpy(np.load(path,mmap_mode='r')[row].copy()).float()


def _mapping(value,name):
    if not isinstance(value,Mapping):raise ValueError(f'{name} must be a mapping')
    return value


def _validate_vad_generation(method,generation):
    if method == Method.VAD.value and (generation.get('vad_importance_policy','on-policy-unit')!='on-policy-unit' or
       generation.get('temperature',1)!=1 or generation.get('top_p',1)!=1 or generation.get('top_k',0)!=0):
        raise ValueError('VAD supports only the declared unwarped on-policy unit-importance policy')


def pending_example(cohort_row,completed_ids):
    return f"{cohort_row['image_id']}:{cohort_row['question_id']}" not in completed_ids


def posterior_probabilities(calibration,record,brain,candidate_features,log_prior):
    scores=calibration.score(record,brain,candidate_features)[0]
    return torch.softmax(torch.from_numpy(np.asarray(scores)).double()+torch.tensor(log_prior),-1).tolist()


def scoring_grad_context(context):
    if context not in {'live','base','snapshot'}:raise ValueError('unknown scoring context')
    return nullcontext() if context=='live' else torch.no_grad()
