"""Resumable frozen CLIP pooled-feature extraction; no brain fitting or labels."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

assets=Path('/root/brain-assets')
root=assets/'nsd-full'
out=root/'features';out.mkdir(exist_ok=True)
records=[json.loads(line) for line in (root/'image_provenance.jsonl').read_text().splitlines()]
records.sort(key=lambda row:row['image_id'])
model_path=assets/'clip-vit-large-patch14'
inputs={'images':hashlib.sha256((root/'image_provenance.jsonl').read_bytes()).hexdigest()}
for name in ['config.json','preprocessor_config.json','pytorch_model.bin']:
    inputs[name]=hashlib.sha256((model_path/name).read_bytes()).hexdigest()
for row in records:
    path=root/'images'/f"{row['image_id']}.jpg"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('source image checksum mismatch')
status_path=out/'clip_progress.json'
start=0
if status_path.exists():
    status=json.loads(status_path.read_text())
    if status['input_hashes']!=inputs:raise ValueError('incompatible feature checkpoint inputs')
    start=status['completed_images']
feature_path=out/'image_pooled.npy'
features=np.lib.format.open_memmap(feature_path,mode='r+' if start else 'w+',dtype=np.float32,shape=(len(records),1024))
(out/'image_ids.json').write_text(json.dumps([r['image_id'] for r in records])+'\n')
processor=CLIPImageProcessor.from_pretrained(model_path,local_files_only=True)
model=CLIPVisionModel.from_pretrained(model_path,local_files_only=True).eval().cuda()
torch.set_float32_matmul_precision('highest')
with torch.inference_mode():
    for i in range(start,len(records),8):
        batch=records[i:i+8]
        images=[]
        for row in batch:
            with Image.open(root/'images'/f"{row['image_id']}.jpg") as image:
                images.append(image.convert('RGB'))
        pixels=processor(images=images,return_tensors='pt').pixel_values.cuda()
        values=model(pixels).pooler_output.float().cpu().numpy()
        if not np.isfinite(values).all():raise ValueError('nonfinite CLIP features')
        features[i:i+len(batch)]=values
        if (i+len(batch))%256==0 or i+len(batch)==len(records):
            features.flush()
            status={'input_hashes':inputs,'completed_images':i+len(batch),'total_images':len(records),
                    'purpose':'fixed image features only; no brain/label fitting',
                    'clip_revision':'32bd64288804d66eefd0ccbe215aa642df71cc41'}
            temporary=out/'clip_progress.tmp'
            temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_path)
            print(f"{i+len(batch)}/{len(records)}",flush=True)
features.flush()
status['feature_sha256']=hashlib.sha256(feature_path.read_bytes()).hexdigest()
status['all_finite']=bool(np.isfinite(features).all())
status['max_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
(out/'clip_provenance.json').write_text(json.dumps(status,indent=2)+'\n')
print('COMPLETE',flush=True)
