#!/usr/bin/env python3
"""CPU proxy cascade grid over OOF utility labels; freeze protocol before Qwen V1."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from common import LABELS,keyed,read_jsonl
from multimodal_data import MODES
from diagnose_oof import COST,OPTIONS,label_for,read_dev_csv
ROOT=Path(__file__).resolve().parent
LAMBDAS=[0.,.05,.1,.2,.5];EPS=[.01,.05];ORDER={a:i for i,a in enumerate(MODES)}

def text_input(p):
 current=p['current'];history=' '.join('speaker '+str(u['speaker'])+' says '+u['text'] for u in p['history'])
 return 'history '+history+' target_speaker '+str(current['speaker'])+' current '+current['text']

def cascade(actions,gold,dev):
 y=[gold[k]['label'] for k in gold];yp=[dev[actions[i]][k]['pred_label'] for i,k in enumerate(gold)]
 counts=Counter(actions);n=len(actions);p,r,f,s=precision_recall_fscore_support(y,yp,labels=list(LABELS),zero_division=0)
 return {'n':n,'accuracy':float(accuracy_score(y,yp)),'weighted_f1':float(f1_score(y,yp,labels=list(LABELS),average='weighted',zero_division=0)),
  'macro_f1':float(f1_score(y,yp,labels=list(LABELS),average='macro',zero_division=0)),
  'cost':sum(COST[m] for m in actions)/n,'audio_rate':(counts['T+A']+counts['T+A+V'])/n,'video_rate':(counts['T+V']+counts['T+A+V'])/n,
  'either_rate':(n-counts['T'])/n,'mean_modalities':(n+counts['T+A']+counts['T+V']+2*counts['T+A+V'])/n,
  'action_counts':{OPTIONS[m]:counts[m] for m in MODES},
  'per_class':{k:{'support':int(s[i]),'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i])} for i,k in enumerate(LABELS)}}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--oof-root',type=Path,required=True,help='Contains merged/utility_inputs.jsonl')
 parser.add_argument('--dev-predictions',type=Path,required=True,help='Contains T, T_A, T_V, T_A_V prediction directories')
 parser.add_argument('--data-dir',type=Path,default=Path('data/processed'))
 parser.add_argument('--output-dir',type=Path,required=True)
 args=parser.parse_args();OUT=args.output_dir
 rows=read_jsonl(args.oof_root/'merged/utility_inputs.jsonl');train=keyed(read_jsonl(args.data_dir/'train.inputs.jsonl'));di=keyed(read_jsonl(args.data_dir/'dev.inputs.jsonl'));gold=keyed(read_jsonl(args.data_dir/'dev.gold.jsonl'))
 if len(rows)!=9989 or len({r['id'] for r in rows})!=9989 or set(train)!={r['id'] for r in rows} or len(gold)!=1109 or set(di)!=set(gold):raise ValueError('data coverage')
 for r in rows:
  if r['dialogue_id']!=train[r['id']]['dialogue_id']:raise ValueError('dialogue mismatch')
 dev={m:read_dev_csv(args.dev_predictions/m.replace('+','_')/'predictions.csv',gold) for m in MODES}
 if OUT.exists():raise ValueError('Output exists; never overwrite')
 OUT.mkdir(parents=True)
 vector=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,dtype=np.float64)
 x=vector.fit_transform([text_input(train[r['id']]) for r in rows]);z=vector.transform([text_input(di[k]) for k in gold]);results=[];labels_cache={};actions_cache={}
 for lam in LAMBDAS:
  for eps in EPS:
   labels=[label_for(r,lam,eps) for r in rows];lc=Counter(labels)
   clf=LogisticRegression(C=1.,max_iter=500,random_state=42)
   clf.fit(x,labels);actions=list(clf.predict(z));score=cascade(actions,gold,dev)
   item={'lambda':lam,'epsilon':eps,'train_label_counts':{OPTIONS[m]:lc[m] for m in MODES},'dev':score,'n_iter':int(clf.n_iter_.max())}
   results.append(item);labels_cache[(lam,eps)]=labels;actions_cache[(lam,eps)]=actions
   print('lambda',lam,'epsilon',eps,'dev_f1',round(score['weighted_f1']*100,3),'cost',round(score['cost'],3),'actions',score['action_counts'],flush=True)
 best=max(r['dev']['weighted_f1'] for r in results);eligible=[r for r in results if r['dev']['weighted_f1']>=best-.01-1e-12]
 chosen=min(eligible,key=lambda r:(r['dev']['cost'],-r['dev']['weighted_f1'],r['lambda'],r['epsilon']))
 key=(chosen['lambda'],chosen['epsilon']);fixed={m:cascade([m]*len(gold),gold,dev) for m in MODES}
 rng=np.random.RandomState(42);rand=[cascade(list(rng.choice(MODES,size=len(gold))),gold,dev) for _ in range(100)]
 random_baseline={k:{'mean':float(np.mean([r[k] for r in rand])),'std':float(np.std([r[k] for r in rand],ddof=1))} for k in ['accuracy','weighted_f1','macro_f1','cost','either_rate','audio_rate','video_rate']}
 oracle=[]
 for id in gold:
  correct=[m for m in MODES if dev[m][id]['pred_label']==gold[id]['label']]
  oracle.append(min(correct,key=lambda m:(COST[m],ORDER[m])) if correct else 'T')
 oracle_result=cascade(oracle,gold,dev)
 protocol={'selection':'Dev weighted F1 within 1.00 percentage point of best grid value; minimize mean C; ties higher weighted F1 then lower lambda then lower epsilon',
 'label_tie_rule':'argmax U, deterministic T,T+A,T+V,T+A+V on exact tie; if argmax prediction is correct, consider other correct actions within strict epsilon of maximum, minimize cost, then maximize utility, then fixed mode order',
 'lambda_grid':LAMBDAS,'epsilon_grid':EPS,'costs':COST,'model':'TF-IDF word 1-2 grams min_df=2 sublinear_tf, multinomial LogisticRegression C=1, unweighted, max_iter=500, random_state=42',
 'model_input':'up to 12 preceding text utterances and speakers plus current speaker and text; no label, descriptions, future turns, probabilities, utilities',
 'checkpoint':'frozen weighted teacher, checkpoint-1875','dev_used_for':'fold teacher checkpoint selection and this proxy grid; not a held-out final test',
 'warning':'Choice optimized for this lightweight model; use as a fixed starting configuration for Qwen V1, not proof Qwen grid is optimized. No test used.'}
 out={'protocol':protocol,'results':results,'best_grid_weighted_f1':best,'eligible_count':len(eligible),'selected':chosen,'fixed':fixed,'uniform_random_100':random_baseline,'oracle_lowest_cost_correct_else_T':oracle_result}
 (OUT/'grid_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'selected_train_actions.jsonl').open('w') as f:
  for r,act in zip(rows,labels_cache[key]):
   util={m:r['log_gain_vs_T'][m]-key[0]*COST[m] for m in MODES}
   f.write(json.dumps({'id':r['id'],'fold':r['fold'],'action':OPTIONS[act],'mode':act,'utility':util},ensure_ascii=False)+'\n')
 with (OUT/'selected_dev_actions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['sample_id','action','mode','true_label','selected_prediction']);w.writeheader()
  for i,id in enumerate(gold):
   action=actions_cache[key][i];w.writerow({'sample_id':id,'action':OPTIONS[action],'mode':action,'true_label':gold[id]['label'],'selected_prediction':dev[action][id]['pred_label']})
 print('SELECTED',json.dumps(chosen,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
