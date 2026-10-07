"""AUROC-only reporting, all preset candidates retained; no best-checkpoint selection."""
import csv,io,json
from pathlib import Path
import numpy as np
from common import ROOT,TARGETS,digest,read_json,release_identity,sha,write_json
from model import CANDIDATES

PAPER={'ped1':.7614,'ped2':.9578,'avenue':.8228}

def normalize(raw,video):
    result=np.zeros(len(raw),dtype=np.float64)
    for v in np.unique(video):
        use=video==v; values=raw[use]; span=values.max()-values.min()
        if span>0: result[use]=(values-values.min())/span
    return result

def auroc(y,score):
    y=np.asarray(y); score=np.asarray(score)
    if not np.isin(y,[0,1]).all() or not np.isfinite(score).all(): raise RuntimeError('Invalid labels/scores')
    order=np.argsort(score,kind='stable'); s=score[order]; labels=y[order]
    starts=np.r_[0,1+np.flatnonzero(s[1:]!=s[:-1])]
    positive=np.add.reduceat(labels.astype(np.float64),starts)
    negative=np.add.reduceat((1-labels).astype(np.float64),starts)
    denominator=positive.sum()*negative.sum()
    if denominator==0: raise RuntimeError('AUROC requires both classes')
    return float((positive*(np.cumsum(negative)-.5*negative)).sum()/denominator)

def generate_report():
    out=ROOT/'outputs'; freeze=read_json(out/'freeze.json')
    if freeze['release']!=release_identity(): raise RuntimeError('Code changed before reporting')
    rows=[]; macro={}
    for candidate in CANDIDATES:
        values=[]
        for name,(_,_,nframes) in TARGETS.items():
            path=out/candidate/(name+'.npz'); meta=read_json(path.with_suffix('.json'))
            if meta['freeze_sha256']!=sha(out/'freeze.json') or meta['scores_sha256']!=sha(path): raise RuntimeError('Score provenance changed')
            with np.load(path,allow_pickle=False) as z:
                y,raw,vid=z['label'],z['raw'],z['video']
                if len(y)!=nframes or len(raw)!=nframes or len(vid)!=nframes: raise RuntimeError('Wrong scored count')
                expected=freeze['targets'][name]
                if z['video_names'].tolist()!=expected['video_names']: raise RuntimeError('Video order changed')
                if not np.array_equal(np.unique(vid),np.arange(len(expected['video_names']))): raise RuntimeError('Wrong video IDs')
                if y.ndim!=1 or raw.ndim!=1 or vid.ndim!=1 or z['frame'].ndim!=1: raise RuntimeError('Scores must be vectors')
                for i,count in enumerate(expected['video_total_frames']):
                    if not np.array_equal(z['frame'][vid==i],np.arange(4,count)): raise RuntimeError('Scored frame alignment changed')
                value=auroc(y,normalize(raw,vid)); values.append(value)
            rows.append({'study':freeze['study'],'candidate':candidate,'seed':17,'checkpoint_step':5000,'target':name,'frames':nframes,'AUROC':value,'paper_AUROC':PAPER[name],'delta_pp':100*(value-PAPER[name])})
        macro[candidate]=float(np.mean(values))
    buff=io.StringIO(); writer=csv.DictWriter(buff,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'results.csv').write_text(buff.getvalue(),encoding='utf-8')
    summary={'status':'COMPLETED','study':freeze['study'],'author_confirmed':False,'kind':'PREDECLARED_IMPLEMENTATION_AUDIT','metric':'AUROC only','candidate_order':list(CANDIDATES),'rows':rows,'macro_AUROC':macro,'paper_comparison_row':'Table4 G=SHT O=SHT','paper_macro_AUROC':float(np.mean(list(PAPER.values()))),'freeze_sha256':sha(out/'freeze.json'),'no_target_training_or_checkpoint_selection':True,'interpretation':'All four candidates are development diagnostics. Numerical proximity does not identify or validate the unavailable authors implementation.'}
    write_json(out/'summary.json',summary)
    lines=['# zxVAD independent V2 implementation audit','','AUROC only; all preset candidates; final step5000. Author implementation remains unconfirmed.','','| Candidate | Ped1 AUROC (%) | Ped2 | Avenue | Macro |','|---|---:|---:|---:|---:|']
    for c in CANDIDATES:
        vals=[next(r['AUROC'] for r in rows if r['candidate']==c and r['target']==t)*100 for t in TARGETS]
        lines.append('| '+c+' | '+' | '.join(f'{v:.4f}' for v in vals+[macro[c]*100])+' |')
    lines+=['','Paper Table4 SHT/SHT:76.14/95.78/82.28%. Pipeline differences and reference conflicts remain disclosed in protocol and reproduction ledger.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'COMPLETED','macro_AUROC':macro,'summary':str(out/'summary.json')},indent=2),flush=True)

if __name__=='__main__': generate_report()
