#!/usr/bin/env python3
"""Reproducible descriptive diagnostics for dialogue-cross-fitted MELD teachers."""
import csv, json, math, statistics, re
from collections import Counter
from pathlib import Path
from common import LABELS, keyed, read_jsonl
from multimodal_data import MODES

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'reports/oof_5fold_v1_download'
OUT=ROOT/'reports/oof_diagnostics_v1'
COST={'T':0.,'T+A':.33,'T+V':.33,'T+A+V':1.}
OPTIONS=dict(zip(MODES,'ABCD'))


def metrics(rows, mode):
    n=len(rows); truth=Counter(r['true_label'] for r in rows); pred=Counter(r['modes'][mode]['prediction'] for r in rows)
    tp=Counter(r['true_label'] for r in rows if r['true_label']==r['modes'][mode]['prediction'])
    f={k:(2*tp[k]/(truth[k]+pred[k]) if truth[k]+pred[k] else 0.) for k in LABELS}
    probs=[r['modes'][mode]['probabilities'] for r in rows]
    # Calibration is for argmax sequence probability, not constrained greedy prediction.
    bins=[[] for _ in range(15)]
    for r,p in zip(rows,probs):
        best=max(LABELS,key=lambda x:p[x]);conf=p[best];bins[min(14,int(conf*15))].append((conf,best==r['true_label']))
    ece=sum(len(b)/n*abs(sum(c for c,_ in b)/len(b)-sum(y for _,y in b)/len(b)) for b in bins if b)
    return {'n':n,'accuracy':sum(tp.values())/n,'weighted_f1':sum(f[k]*truth[k] for k in LABELS)/n,'macro_f1':sum(f.values())/len(LABELS),
            'per_class':{k:{'support':truth[k],'precision':tp[k]/pred[k] if pred[k] else 0.,'recall':tp[k]/truth[k] if truth[k] else 0.,'f1':f[k]} for k in LABELS},
            'nll':sum(-r['modes'][mode]['log_probabilities'][r['true_label']] for r in rows)/n,'ece_15':ece,
            'sequence_vs_greedy_disagreements':sum(r['modes'][mode]['sequence_argmax']!=r['modes'][mode]['prediction'] for r in rows)}


def label_for(row,lam,eps):
    utilities={m:row['log_gain_vs_T'][m]-lam*COST[m] for m in MODES}
    best=max(MODES,key=lambda m:utilities[m]); selected=best
    # Epsilon rule only compares actions that both classify correctly.
    if row['modes'][best]['prediction']==row['true_label']:
        eligible=[m for m in MODES if row['modes'][m]['prediction']==row['true_label'] and utilities[best]-utilities[m] < eps]
        selected=min(eligible,key=lambda m:(COST[m],-utilities[m],MODES.index(m)))
    return selected


def read_dev_csv(path,refs):
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    by={r['sample_id']:r for r in rows}
    if len(by)!=len(rows) or set(by)!=set(refs) or any(v['true_label']!=refs[k]['label'] for k,v in by.items()):raise ValueError(f'Invalid Dev file: {path}')
    return by


def main():
    rows=read_jsonl(DATA/'merged/utility_inputs.jsonl'); gold=keyed(read_jsonl(ROOT/'data/processed/train.gold.jsonl'));inputs=keyed(read_jsonl(ROOT/'data/processed/train.inputs.jsonl'))
    if len(rows)!=9989 or len({r['id'] for r in rows})!=9989 or set(gold)!={r['id'] for r in rows} or set(inputs)!=set(gold):raise ValueError('Invalid training coverage')
    for r in rows:
        if r['true_label']!=gold[r['id']]['label'] or r['dialogue_id']!=inputs[r['id']]['dialogue_id']:raise ValueError('Invalid reference')
        for m in MODES:
            probs=r['modes'][m]['probabilities'];logs=r['modes'][m]['log_probabilities']
            if set(probs)!=set(LABELS) or set(logs)!=set(LABELS) or abs(sum(probs.values())-1)>1e-5 or any(not math.isfinite(logs[l]) or abs(math.exp(logs[l])-probs[l])>1e-5 for l in LABELS):raise ValueError('Invalid probabilities')
    report={'oof':{},'folds':{},'utility':{},'action_distributions':{},'dev_comparison':{},'oracle':{}}
    for m in MODES:report['oof'][m]=metrics(rows,m)
    for k in range(5):
        part=[r for r in rows if r['fold']==k]
        if not part:raise ValueError('Missing fold')
        report['folds'][str(k)]={m:metrics(part,m) for m in MODES}
    for m in MODES:
        gains=[r['log_gain_vs_T'][m] for r in rows]
        report['utility'][m]={'mean_gain':statistics.mean(gains),'median_gain':statistics.median(gains),'positive':sum(g>1e-6 for g in gains),'negative':sum(g< -1e-6 for g in gains),'near_zero_abs_le_0_01':sum(abs(g)<=.01 for g in gains),
          'mean_cost_adjusted':{str(lam):statistics.mean(gains)-lam*COST[m] for lam in [0.,.05,.1,.2,.5]},'positive_cost_adjusted':{str(lam):sum(g-lam*COST[m]>1e-6 for g in gains) for lam in [0.,.05,.1,.2,.5]}}
    for lam in [0.,.05,.1,.2,.5]:
        for eps in [.01,.05]:
            c=Counter(OPTIONS[label_for(r,lam,eps)] for r in rows);report['action_distributions'][f'lambda={lam},epsilon={eps}']={v:c[v] for v in 'ABCD'}
    refs=keyed(read_jsonl(ROOT/'data/processed/dev.gold.jsonl'));devroot=DATA/'dev_comparisons'
    for mode in MODES:
        md=mode.replace('+','_');mainpred=read_dev_csv(devroot/'main_dev_modes'/md/'predictions.csv',refs)
        main_correct=sum(v['true_label']==v['pred_label'] for v in mainpred.values())
        data=[]
        for k in range(5):
            sub=read_dev_csv(devroot/f'fold_{k}'/'dev_modes'/md/'predictions.csv',refs)
            mainrows=[{'id':i,'true_label':x['true_label'],'modes':{mode:{'prediction':x['pred_label'],'probabilities':{l:1. if l==x['pred_label'] else 0. for l in LABELS},'log_probabilities':{l:0. for l in LABELS},'sequence_argmax':x['pred_label']}}} for i,x in mainpred.items()]
            foldrows=[{'id':i,'true_label':x['true_label'],'modes':{mode:{'prediction':x['pred_label'],'probabilities':{l:1. if l==x['pred_label'] else 0. for l in LABELS},'log_probabilities':{l:0. for l in LABELS},'sequence_argmax':x['pred_label']}}} for i,x in sub.items()]
            mm=metrics(mainrows,mode);fm=metrics(foldrows,mode)
            data.append({'fold':k,'main_accuracy':mm['accuracy'],'main_weighted_f1':mm['weighted_f1'],'main_macro_f1':mm['macro_f1'],'fold_accuracy':fm['accuracy'],'fold_weighted_f1':fm['weighted_f1'],'fold_macro_f1':fm['macro_f1'],
                'prediction_disagreements':sum(sub[i]['pred_label']!=mainpred[i]['pred_label'] for i in refs),'fold_correct_main_wrong':sum(sub[i]['pred_label']==refs[i]['label'] and mainpred[i]['pred_label']!=refs[i]['label'] for i in refs),'main_correct_fold_wrong':sum(sub[i]['pred_label']!=refs[i]['label'] and mainpred[i]['pred_label']==refs[i]['label'] for i in refs)})
        report['dev_comparison'][mode]=data
    oracle=[r for r in rows if r['modes']['T']['prediction']!=r['true_label'] and any(r['modes'][m]['prediction']==r['true_label'] for m in MODES[1:])]
    baseline=[r for r in rows if r['modes']['T']['prediction']!=r['true_label'] and all(r['modes'][m]['prediction']!=r['true_label'] for m in MODES[1:])]
    def features(r):
        text=inputs[r['id']]['current']['text'];words=re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",text.lower())
        return {'short_le_5_words':len(words)<=5,'question_mark':'?' in text,'exclamation':'!' in text,'ellipsis':'...' in text or '…' in text,
                'first_person':bool(set(words)&{'i','me','my','i’m',"i'm"}),'second_person':bool(set(words)&{'you','your','you’re',"you're"}),
                'no_previous_context':len(inputs[r['id']]['history'])==0}
    report['oracle']={'count':len(oracle),'T_wrong_no_modality_correct':len(baseline),'text_features':{name:{'oracle_n':sum(features(r)[name] for r in oracle),'baseline_n':sum(features(r)[name] for r in baseline)} for name in features(rows[0])},
      'oracle_by_class':dict(Counter(r['true_label'] for r in oracle))}
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'diagnostics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'oracle_text_candidates.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['id','fold','true_label','text','history','T','T+A','T+V','T+A+V','correct_modes'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in oracle:
            inp=inputs[r['id']];w.writerow({'id':r['id'],'fold':r['fold'],'true_label':r['true_label'],'text':inp['current']['text'],'history':json.dumps(inp['history'],ensure_ascii=False),**{m:r['modes'][m]['prediction'] for m in MODES},'correct_modes':','.join(m for m in MODES[1:] if r['modes'][m]['prediction']==r['true_label'])})
    print(json.dumps({'oof':{m:{k:report['oof'][m][k] for k in ('n','accuracy','weighted_f1','macro_f1','nll','ece_15','sequence_vs_greedy_disagreements')} for m in MODES},'utility':report['utility'],'action_distributions':report['action_distributions'],'dev_comparison':report['dev_comparison'],'oracle':report['oracle']},indent=2))

if __name__=='__main__':main()
