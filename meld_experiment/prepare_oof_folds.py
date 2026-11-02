#!/usr/bin/env python3
"""Create reproducible dialogue-disjoint folds for MELD training examples."""
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    inputs_path=args.data_dir/'train.inputs.jsonl';gold_path=args.data_dir/'train.gold.jsonl'
    rows=read(inputs_path);gold=read(gold_path)
    ids=[r['id'] for r in rows];labels={r['id']:r['label'] for r in gold}
    if len(ids)!=9989 or len(set(ids))!=len(ids) or len(gold)!=len(labels) or set(ids)!=set(labels):raise ValueError('Unexpected train IDs or references')
    groups=defaultdict(list)
    for row in rows:
        if row.get('split')!='train' or not row['id'].startswith('train/'):raise ValueError('Non-training sample')
        groups[row['dialogue_id']].append(row['id'])
    if args.folds<2 or len(groups)<args.folds:raise ValueError('Invalid fold count')
    order=sorted(groups);random.Random(args.seed).shuffle(order);order.sort(key=lambda d:len(groups[d]),reverse=True)
    folds=[[] for _ in range(args.folds)]; sizes=[0]*args.folds
    for dialogue_id in order:
        k=min(range(args.folds),key=lambda j:(sizes[j],j));folds[k].append(dialogue_id);sizes[k]+=len(groups[dialogue_id])
    assignments={sample_id:k for k,dialogues in enumerate(folds) for did in dialogues for sample_id in groups[did]}
    if len(assignments)!=len(ids):raise ValueError('Fold coverage mismatch')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    config={'folds':args.folds,'seed':args.seed,'method':'shuffle dialogue IDs, stable sort by descending utterance count, assign each to smallest fold',
            'input_sha256':hashlib.sha256(inputs_path.read_bytes()).hexdigest(),'gold_sha256':hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            'samples':len(ids),'dialogues':len(groups),'fold_summaries':[]}
    for k,dialogues in enumerate(folds):
        held=set(dialogues); held_ids=[r['id'] for r in rows if r['dialogue_id'] in held];train_ids=[r['id'] for r in rows if r['dialogue_id'] not in held]
        if set(train_ids)&set(held_ids) or len(held_ids)+len(train_ids)!=len(rows):raise ValueError('Split mismatch')
        (args.output_dir/f'fold_{k}_heldout_ids.txt').write_text('\n'.join(held_ids)+'\n')
        (args.output_dir/f'fold_{k}_train_ids.txt').write_text('\n'.join(train_ids)+'\n')
        config['fold_summaries'].append({'fold':k,'dialogues':len(dialogues),'heldout_samples':len(held_ids),'train_samples':len(train_ids),
                  'heldout_labels':dict(Counter(labels[i] for i in held_ids)),'train_labels':dict(Counter(labels[i] for i in train_ids))})
    (args.output_dir/'assignment.json').write_text(json.dumps(assignments,indent=2)+'\n')
    (args.output_dir/'manifest.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(config,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
