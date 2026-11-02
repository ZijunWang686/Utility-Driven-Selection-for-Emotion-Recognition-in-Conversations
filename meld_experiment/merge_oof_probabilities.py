#!/usr/bin/env python3
"""Validate and merge five OOF folds; preserve continuous gains for later cost tuning."""
import argparse,hashlib,json,math
from pathlib import Path
from common import LABELS,keyed,read_jsonl
from multimodal_data import MODES
COSTS={'T':0.0,'T+A':0.33,'T+V':0.33,'T+A+V':1.0}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold-dir',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--data-dir',type=Path,default=Path('data/processed'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.fold_dir/'manifest.json').read_text());gold=keyed(read_jsonl(a.data_dir/'train.gold.jsonl'));inputs=keyed(read_jsonl(a.data_dir/'train.inputs.jsonl'))
 if manifest['folds']!=5 or sha(a.data_dir/'train.inputs.jsonl')!=manifest['input_sha256'] or sha(a.data_dir/'train.gold.jsonl')!=manifest['gold_sha256'] or set(gold)!=set(inputs):raise ValueError('Invalid source data')
 combined={};sources=[]
 for fold in range(5):
  root=a.run_root/f'fold_{fold}'/'oof_predictions';cfg=json.loads((root/'config.json').read_text());records=read_jsonl(root/'probabilities.jsonl')
  held_path=a.fold_dir/f'fold_{fold}_heldout_ids.txt';train_path=a.fold_dir/f'fold_{fold}_train_ids.txt';expected=set(held_path.read_text().splitlines());actual=[r['id'] for r in records]
  if cfg['limit'] is not None or cfg['samples']!=len(expected) or cfg['train_ids_sha256']!=sha(train_path) or cfg['heldout_ids_sha256']!=sha(held_path) or len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError(f'Fold {fold} invalid coverage/config')
  if set(actual)&set(combined):raise ValueError('Duplicate OOF sample')
  for r in records:
   k=r['id'];y=gold[k]['label']
   if r['true_label']!=y or r['dialogue_id']!=inputs[k]['dialogue_id'] or set(r['modes'])!=set(MODES):raise ValueError(f'Invalid record {k}')
   for mode in MODES:
    m=r['modes'][mode];pmap=m['probabilities'];lm=m['log_probabilities']
    if set(pmap)!=set(LABELS) or set(lm)!=set(LABELS) or abs(sum(pmap.values())-1)>1e-5 or any(not math.isfinite(lm[t]) or abs(math.exp(lm[t])-pmap[t])>1e-5 for t in LABELS):raise ValueError(f'Invalid probabilities {k} {mode}')
   base=r['modes']['T']['log_probabilities'][y]
   r['fold']=fold;r['log_gain_vs_T']={m:r['modes'][m]['log_probabilities'][y]-base for m in MODES}
   combined[k]=r
  sources.append({'fold':fold,'heldout_count':len(records),'adapter_sha256':cfg['adapter_sha256'],'probabilities_sha256':sha(root/'probabilities.jsonl')})
 if set(combined)!=set(inputs):raise ValueError('Incomplete OOF coverage')
 a.output_dir.mkdir(parents=True,exist_ok=False)
 with (a.output_dir/'utility_inputs.jsonl').open('w') as f:
  for k in inputs:f.write(json.dumps(combined[k],ensure_ascii=False)+'\n')
 (a.output_dir/'manifest.json').write_text(json.dumps({'samples':len(combined),'folds':sources,'costs':COSTS,'definition':'utility(m)=log_gain_vs_T(m)-lambda*cost(m); lambda and epsilon not yet selected','data_manifest_sha256':sha(a.fold_dir/'manifest.json')},indent=2)+'\n')
 print('Validated and merged',len(combined),'OOF samples')
if __name__=='__main__':main()
