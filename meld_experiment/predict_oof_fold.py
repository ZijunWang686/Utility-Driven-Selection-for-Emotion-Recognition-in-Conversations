#!/usr/bin/env python3
"""Score one dialogue-heldout train fold with its independently trained adapter."""
import argparse,hashlib,json,logging,math
from pathlib import Path
from common import LABELS,keyed,parse_prediction,read_jsonl
from multimodal_data import MODES,encode_multimodal_prompt,load_split
from zero_shot_meld import label_constraint


def ids(path):
    items=[s.strip() for s in path.read_text().splitlines() if s.strip()]
    if not items or len(items)!=len(set(items)):raise ValueError(f'Empty or duplicate IDs: {path}')
    return items


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--media-root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('data/processed'));p.add_argument('--train-ids',type=Path,required=True)
    p.add_argument('--heldout-ids',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--limit',type=int)
    p.add_argument('--allow-smoke',action='store_true',help='Permit smoke-trained adapters for limited tests only')
    a=p.parse_args()
    training,held=ids(a.train_ids),ids(a.heldout_ids)
    rows=load_split(a.data_dir,a.media_root,'train',include_labels=True);index=keyed(rows)
    if set(training)&set(held) or set(training)|set(held)!=set(index):raise ValueError('Train and held-out IDs must partition train split')
    train_dialogues={index[x]['dialogue_id'] for x in training};held_dialogues={index[x]['dialogue_id'] for x in held}
    if train_dialogues & held_dialogues:raise ValueError('Dialogue crosses folds')
    config=json.loads((a.adapter/'run_config.json').read_text())
    sha=hashlib.sha256(a.train_ids.read_bytes()).hexdigest()
    if config.get('train_id_sha256')!=sha or config.get('imbalance')!='weighted':raise ValueError('Adapter is not the weighted model trained on this fold')
    if config.get('smoke') and (not a.allow_smoke or not a.limit):raise ValueError('Smoke adapter requires --allow-smoke and --limit')
    if a.limit:
        if a.limit<1:raise ValueError('limit must be positive')
        held=held[:a.limit]
    a.output_dir.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(a.output_dir/'run.log'),logging.StreamHandler()])
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    tok=AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    targets=[tok(label+tok.eos_token,add_special_tokens=False)['input_ids'] for label in LABELS]
    if any(not t or t[-1]!=tok.eos_token_id or len(t)>16 for t in targets) or len({tuple(t) for t in targets})!=len(targets):raise ValueError('Invalid label sequences')
    model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa'),str(a.adapter)).eval()
    metadata={'adapter':str(a.adapter),'adapter_sha256':hashlib.sha256((a.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
              'train_ids_sha256':sha,'heldout_ids_sha256':hashlib.sha256(a.heldout_ids.read_bytes()).hexdigest(),
              'samples':len(held),'limit':a.limit,'labels':LABELS,'sequence_definition':'full label plus EOS log likelihood; softmax over seven labels',
              'prediction':'seven-label constrained greedy decoding','selection':'Dev T+A+V weighted F1'}
    (a.output_dir/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (a.output_dir/'probabilities.jsonl').open('w') as f,torch.inference_mode():
        for i,k in enumerate(held,1):
            row=index[k];record={'id':k,'true_label':row['label'],'dialogue_id':row['dialogue_id'],'modes':{}}
            for mode in MODES:
                prompt,dropped=encode_multimodal_prompt(tok,row,mode,1008);x=torch.tensor([prompt],device='cuda')
                gen=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=16,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,prefix_allowed_tokens_fn=label_constraint(tok,len(prompt),16))
                pred=parse_prediction(tok.decode(gen[0,len(prompt):],skip_special_tokens=True).strip())
                if pred not in LABELS:raise ValueError('Invalid prediction')
                sequences=[prompt+t for t in targets];width=max(map(len,sequences))
                tokens=torch.tensor([s+[tok.pad_token_id]*(width-len(s)) for s in sequences],device='cuda')
                masks=torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in sequences],device='cuda')
                logits=model(input_ids=tokens,attention_mask=masks,use_cache=False).logits.float();scores=[]
                for j,target in enumerate(targets):
                    pos=torch.arange(len(prompt)-1,len(prompt)-1+len(target),device='cuda')
                    lp=torch.log_softmax(logits[j,pos,:],dim=-1)
                    scores.append(lp[torch.arange(len(target),device='cuda'),torch.tensor(target,device='cuda')].sum())
                scores=torch.stack(scores);logp=torch.log_softmax(scores,dim=0).cpu().tolist()
                if not all(math.isfinite(v) for v in logp):raise ValueError('Non-finite score')
                record['modes'][mode]={'prediction':pred,'sequence_argmax':LABELS[int(scores.argmax())],
                      'log_probabilities':dict(zip(LABELS,logp)),'probabilities':dict(zip(LABELS,map(math.exp,logp))),'dropped_history':dropped}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            if i%100==0 or i==len(held):logging.info('Completed %d/%d',i,len(held))
    logging.info('Completed held-out fold')

if __name__=='__main__':main()
