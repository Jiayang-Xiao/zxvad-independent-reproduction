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


def factorial_contrasts(values):
    # Interactions at the all-other-factors-off corner; unit: AUROC percentage points.
    b=values['B']
    return {'T_N':100*(values['TN']-values['T']-values['N']+b),
            'T_Q':100*(values['TQ']-values['T']-values['Q']+b),
            'N_Q':100*(values['NQ']-values['N']-values['Q']+b),
            'T_N_Q':100*(values['TNQ']-values['TN']-values['TQ']-values['NQ']+values['T']+values['N']+values['Q']-b)}

def generate_report():
    out=ROOT/'outputs';freeze=read_json(out/'freeze.json')
    if freeze['release']!=release_identity():raise RuntimeError('Code changed before reporting')
    rows=[];macro={};target_values={t:{} for t in TARGETS}
    for c in CANDIDATES:
        values=[]
        for name,(_,_,nframes) in TARGETS.items():
            p=out/c/(name+'.npz');m=read_json(p.with_suffix('.json'))
            if m['freeze_sha256']!=sha(out/'freeze.json') or m['scores_sha256']!=sha(p):raise RuntimeError('Score provenance changed')
            with np.load(p,allow_pickle=False) as z,np.load(ROOT/'reference'/('c04_'+name+'.npz'),allow_pickle=False) as ref:
                for k in ('label','video','frame','video_names'):
                    if not np.array_equal(z[k],ref[k]):raise RuntimeError('Audited frame/label alignment differs: '+c+'/'+name+'/'+k)
                if z['raw'].shape!=(nframes,):raise RuntimeError('Wrong scored count')
                expected=freeze['targets'][name]
                if z['video_names'].tolist()!=expected['video_names']:raise RuntimeError('Freeze video identity changed')
                for vi,count in enumerate(expected['video_total_frames']):
                    if not np.array_equal(z['frame'][z['video']==vi],np.arange(4,count)):raise RuntimeError('Scored frame alignment changed')
                value=auroc(z['label'],normalize(z['raw'],z['video']))
            values.append(value);target_values[name][c]=value
            rows.append({'study':freeze['study'],'candidate':c,'seed':17,'checkpoint_step':5000,'target':name,'frames':nframes,'AUROC':value})
        macro[c]=float(np.mean(values))
    for row in rows:row['delta_vs_B_pp']=100*(row['AUROC']-target_values[row['target']]['B'])
    buff=io.StringIO();w=csv.DictWriter(buff,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'results.csv').write_text(buff.getvalue(),encoding='utf-8')
    summary={'status':'COMPLETED','study':freeze['study'],'author_confirmed':False,'kind':'PREDECLARED_DEVELOPMENT_FACTORIAL','metric':'AUROC only','candidate_order':list(CANDIDATES),'rows':rows,'macro_AUROC':macro,'macro_delta_vs_B_pp':{c:100*(v-macro['B']) for c,v in macro.items()},'primary_comparison':'TNQ versus paired B; report even if negative','primary_delta_pp':100*(macro['TNQ']-macro['B']),'factorial_interaction_pp':{'macro':factorial_contrasts(macro),**{t:factorial_contrasts(v) for t,v in target_values.items()}},'freeze_sha256':sha(out/'freeze.json'),'no_target_training_or_checkpoint_selection':True,'prior_target_feedback_used_for_design':True,'interpretation':'Seed17 development evidence only. Historical control labels preserved. Cross-domain target AUROC guided this prospective design; any winning arm requires fresh predeclared confirmation.'}
    write_json(out/'summary.json',summary)
    lines=['# c04 T/N/Q factorial','','AUROC only; all eight predeclared arms; final step5000; same-host paired B.','','| Arm | Ped1 (%) | Ped2 | Avenue | Macro | Delta vs B (pp) |','|---|---:|---:|---:|---:|---:|']
    for c in CANDIDATES:
        vals=[target_values[t][c]*100 for t in TARGETS]+[macro[c]*100,100*(macro[c]-macro['B'])]
        lines.append('| '+c+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','Primary comparison TNQ-B: '+f"{summary['primary_delta_pp']:+.4f}"+' pp. No cross-host baseline subtraction.','All target gains and losses remain visible. Prior results informed design; this is not a blind final claim.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'COMPLETED','macro_AUROC':macro,'primary_delta_pp':summary['primary_delta_pp'],'summary':str(out/'summary.json')},indent=2),flush=True)

if __name__=='__main__':generate_report()
