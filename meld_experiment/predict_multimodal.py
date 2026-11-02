#!/usr/bin/env python3
"""Evaluate one shared classifier and modality combination on MELD Dev."""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from common import keyed, parse_prediction, read_jsonl
from multimodal_data import MODES, encode_multimodal_prompt, load_split
from score import evaluate
from zero_shot_meld import label_constraint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--media-root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    p.add_argument('--split',choices=['dev'],default='dev')
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--max-new-tokens',type=int,default=16)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit<1: p.error('--limit must be positive')
    rows=load_split(args.data_dir,args.media_root,args.split,include_labels=False)
    if len(rows)!=1109: raise ValueError(f'Expected 1109 Dev examples, got {len(rows)}')
    if args.limit: rows=rows[:args.limit]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(args.output_dir/'run.log'),logging.StreamHandler()])
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 is required')
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,str(args.adapter)).eval()
    model.config.use_cache=True
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter(); records=[]
    with (args.output_dir/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for i,row in enumerate(rows,1):
            tokens,dropped=encode_multimodal_prompt(tokenizer,row,args.mode,args.max_length-args.max_new_tokens)
            ids=torch.tensor([tokens],device='cuda')
            with torch.inference_mode():
                out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                    max_new_tokens=args.max_new_tokens,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=label_constraint(tokenizer,len(tokens),args.max_new_tokens))
            text=tokenizer.decode(out[0,len(tokens):],skip_special_tokens=True).strip()
            record={'id':row['id'],'prediction':parse_prediction(text),'raw_output':text,'mode':args.mode,
                    'audio_status':row['audio_status'],'video_status':row['video_status'],'dropped_history':dropped}
            records.append(record); f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
            if i%100==0 or i==len(rows): logging.info('Completed %d/%d',i,len(rows))
    gold_map=keyed(read_jsonl(args.data_dir/'dev.gold.jsonl'))
    gold=[gold_map[r['id']] for r in rows]
    metrics=evaluate(gold,records)
    metrics.update({'mode':args.mode,'elapsed_seconds':time.perf_counter()-started,
                    'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (args.output_dir/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','true_label','pred_label']); w.writeheader()
        w.writerows({'sample_id':r['id'],'true_label':gold_map[r['id']]['label'],'pred_label':r['prediction']} for r in records)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from evaluate_meld import evaluate_meld
    evaluate_meld(args.output_dir/'predictions.csv',args.output_dir/'evaluation',expected_count=len(rows),reference_rows=gold)
    (args.output_dir/'run_config.json').write_text(json.dumps({'model':args.model,'adapter':str(args.adapter),
        'split':args.split,'mode':args.mode,'limit':args.limit,'max_length':args.max_length,'max_new_tokens':args.max_new_tokens,
        'enable_thinking':False,'do_sample':False,'output_constraint':'seven_label_token_trie'},indent=2)+'\n')
    logging.info('Metrics: %s',json.dumps(metrics,ensure_ascii=False))

if __name__=='__main__': main()
