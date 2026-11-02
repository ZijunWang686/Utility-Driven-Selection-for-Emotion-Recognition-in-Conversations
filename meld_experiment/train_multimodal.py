#!/usr/bin/env python3
"""Train a shared BF16 LoRA classifier using per-epoch modality dropout."""
import argparse
import gc
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from multimodal_data import MODES,WEIGHTS,encode_training_row,load_split,mode_for_sample


class MixedDataset:
    def __init__(self,rows,tokenizer,cfg): self.rows,self.tokenizer,self.cfg,self.epoch=rows,tokenizer,cfg,0
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        mode=mode_for_sample(self.cfg['seed'],self.epoch,row['id'])
        item=encode_training_row(self.tokenizer,row,mode,self.cfg['max_length'],self.cfg['max_new_tokens'])
        if self.cfg.get('imbalance') == 'weighted': item['sample_weight']=self.cfg['class_weights'][row['label']]
        return item


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/text_lora_qwen3_8b.json'))
    p.add_argument('--model',required=True)
    p.add_argument('--media-root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--imbalance',choices=['none','weighted','oversample'],default='none')
    p.add_argument('--train-id-file',type=Path,help='Optional strict subset of train sample IDs for OOF training')
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    if cfg['use_4bit']: raise ValueError('This first run is BF16 LoRA; use_4bit must be false')
    train=load_split(args.data_dir,args.media_root,'train',include_labels=True)
    dev=load_split(args.data_dir,args.media_root,'dev',include_labels=False)
    if len(train)!=9989 or len(dev)!=1109: raise ValueError('Unexpected MELD split counts')
    if args.train_id_file:
        selected=[line.strip() for line in args.train_id_file.read_text().splitlines() if line.strip()]
        if len(set(selected))!=len(selected) or not selected: raise ValueError('Invalid training ID subset')
        available={row['id'] for row in train}
        if not set(selected)<=available: raise ValueError('Training ID subset includes unknown IDs')
        selected_ids=set(selected)
        train=[row for row in train if row['id'] in selected_ids]
    counts=Counter(row['label'] for row in train)
    raw={label:math.sqrt(len(train)/count) for label,count in counts.items()}
    mean=sum(counts[label]*weight for label,weight in raw.items())/len(train)
    weights={label:weight/mean for label,weight in raw.items()}
    if args.smoke: train=train[:32]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(args.output_dir/'run.log'),logging.StreamHandler()])
    cfg.update({'model_name':args.model,'modality_probabilities':dict(zip(MODES,WEIGHTS)),
                'media_root':str(args.media_root.resolve()),'smoke':args.smoke,'checkpoint_metric':'Dev T+A+V Weighted-F1',
                'missing_modality_marker':'[unavailable]', 'prompt':'MELD Prompt A with selected description fields',
                'imbalance':args.imbalance,'class_counts':dict(counts),'class_weights':weights,
                'imbalance_formula':'normalized inverse square root class frequency'})
    if args.train_id_file:
        cfg.update({'train_id_file':str(args.train_id_file.resolve()),
                    'train_id_sha256':hashlib.sha256(args.train_id_file.read_bytes()).hexdigest(),
                    'train_samples':len(train)})
    (args.output_dir/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,TrainerCallback,set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA with BF16 required')
    set_seed(cfg['seed'])
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    dataset=MixedDataset(train,tokenizer,cfg)
    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self,args,state,control,**kwargs):
            dataset.epoch=int(math.floor((state.epoch or 0)+1e-6))
            counts=Counter(mode_for_sample(cfg['seed'],dataset.epoch,r['id']) for r in dataset.rows)
            logging.info('Modality sampling epoch=%d seed=%d counts=%s',dataset.epoch,cfg['seed'],json.dumps(counts))
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs:
                logging.info('Training step=%d %s',state.global_step,json.dumps(logs,default=str))
    def collate(rows):
        width=max(len(r['input_ids']) for r in rows)
        batch={'input_ids':torch.tensor([r['input_ids']+[tokenizer.pad_token_id]*(width-len(r['input_ids'])) for r in rows]),
                'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows]),
                'labels':torch.tensor([r['labels']+[-100]*(width-len(r['labels'])) for r in rows])}
        if 'sample_weight' in rows[0]: batch['sample_weight']=torch.tensor([r['sample_weight'] for r in rows])
        return batch
    class ImbalanceTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            if args.imbalance != 'oversample': return super()._get_train_sampler(train_dataset)
            data=train_dataset if train_dataset is not None else self.train_dataset
            generator=torch.Generator().manual_seed(cfg['seed'])
            return torch.utils.data.WeightedRandomSampler([weights[row['label']] for row in data.rows],len(data),replacement=True,generator=generator)
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            sample_weight=inputs.pop('sample_weight',None)
            if sample_weight is None: return super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
            outputs=model(**inputs)
            loss=outputs.loss*sample_weight.mean()
            return (loss,outputs) if return_outputs else loss
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],
        bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj']))
    model.print_trainable_parameters()
    training_args=TrainingArguments(output_dir=str(args.output_dir),num_train_epochs=cfg['epochs'],max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=1,gradient_accumulation_steps=2 if args.smoke else cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],warmup_ratio=0.03,weight_decay=0.01,lr_scheduler_type='cosine',
        bf16=True,tf32=True,optim='adamw_torch',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='no',save_strategy='steps' if args.smoke else 'epoch',save_steps=2 if args.smoke else 500,
        save_only_model=True,save_total_limit=None,logging_steps=1 if args.smoke else 10,report_to='none',
        seed=cfg['seed'],data_seed=cfg['seed'],dataloader_num_workers=0,remove_unused_columns=False)
    trainer=ImbalanceTrainer(model=model,args=training_args,train_dataset=dataset,data_collator=collate,processing_class=tokenizer,callbacks=[EpochCallback])
    torch.cuda.reset_peak_memory_stats(); trainer.train(); trainer.save_state()
    cfg['gpu_peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
    (args.output_dir/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    if args.smoke:
        final=args.output_dir/'adapter'; trainer.save_model(str(final)); tokenizer.save_pretrained(str(final))
        (final/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        logging.info('Smoke training finished; adapter=%s',final)
        return
    checkpoints=sorted(args.output_dir.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
    if not checkpoints: raise RuntimeError('No epoch checkpoints found')
    del trainer,model; gc.collect(); torch.cuda.empty_cache()
    predictor=Path(__file__).with_name('predict_multimodal.py')
    results=[]
    for ck in checkpoints:
        out=ck/'dev_full_mode'
        cmd=[sys.executable,str(predictor),'--adapter',str(ck),'--model',args.model,'--data-dir',str(args.data_dir),
             '--media-root',str(args.media_root),'--mode','T+A+V','--output-dir',str(out),
             '--max-length',str(cfg['max_length']),'--max-new-tokens',str(cfg['max_new_tokens'])]
        logging.info('Evaluating checkpoint %s',ck); subprocess.run(cmd,check=True)
        metrics=json.loads((out/'metrics.json').read_text())
        results.append({'checkpoint':str(ck),'weighted_f1':metrics['weighted_f1'],'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1']})
    best=max(results,key=lambda r:r['weighted_f1'])
    final=args.output_dir/'adapter'; final.mkdir()
    ck=Path(best['checkpoint'])
    for name in ('adapter_config.json','adapter_model.safetensors','adapter_model.bin'):
        if (ck/name).is_file(): shutil.copy2(ck/name,final/name)
    tokenizer.save_pretrained(str(final))
    cfg.update({'best_checkpoint':str(ck),'checkpoint_results':results})
    (final/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (args.output_dir/'checkpoint_results.json').write_text(json.dumps(results,indent=2)+'\n')
    logging.info('Selected checkpoint: %s',json.dumps(best))
    for mode in MODES:
        out=args.output_dir/'dev_modes'/mode.replace('+','_')
        subprocess.run([sys.executable,str(predictor),'--adapter',str(final),'--model',args.model,
            '--data-dir',str(args.data_dir),'--media-root',str(args.media_root),'--mode',mode,'--output-dir',str(out),
            '--max-length',str(cfg['max_length']),'--max-new-tokens',str(cfg['max_new_tokens'])],check=True)
    logging.info('Completed training, checkpoint selection, and four Dev modality evaluations')

if __name__=='__main__': main()
