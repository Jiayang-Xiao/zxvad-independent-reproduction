"""Retain all planned arms, incomplete evaluations and paired AUROC contrasts."""
import csv,io
import numpy as np
from common import ROOT,TARGETS,read_json,release_identity,sha,write_json,digest
from model import ORDER,RECIPES,SEEDS,parse
from schedule import QUEUES,lane_for

def normalize(raw,video):
    result=np.zeros(len(raw),dtype=np.float64)
    for v in np.unique(video):
        use=video==v;values=raw[use];span=values.max()-values.min()
        if span>0:result[use]=(values-values.min())/span
    return result

def auroc(y,score):
    y=np.asarray(y);score=np.asarray(score)
    if not np.isin(y,[0,1]).all() or not np.isfinite(score).all():raise RuntimeError('Invalid scores/labels')
    order=np.argsort(score,kind='stable');s=score[order];labels=y[order]
    starts=np.r_[0,1+np.flatnonzero(s[1:]!=s[:-1])]
    positive=np.add.reduceat(labels.astype(np.float64),starts);negative=np.add.reduceat((1-labels).astype(np.float64),starts)
    denominator=positive.sum()*negative.sum()
    if not denominator:raise RuntimeError('Both classes needed')
    return float((positive*(np.cumsum(negative)-.5*negative)).sum()/denominator)

def generate_report():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    freeze=read_json(out/'freeze.json') if (out/'freeze.json').exists() else None
    if freeze and freeze['release']!=release_identity():raise RuntimeError('Frozen code changed')
    rows=[];macro={};missing=[];trained=[];target_values={}
    for arm in ORDER:
        recipe,seed=parse(arm);donepath=out/arm/'completed.json'
        if not donepath.exists():
            state=read_json(out/arm/'state.json') if (out/arm/'state.json').exists() else None
            missing.append({'arm':arm,'reason':'not trained to final5000','state':state});continue
        trained.append(arm);done=read_json(donepath);cfg=read_json(out/arm/'config.json')
        if done['step']!=5000 or done['config_hash']!=digest(cfg):raise RuntimeError('Invalid final fit')
        if not freeze or freeze['models'].get(arm)!=done:
            missing.append({'arm':arm,'reason':'completed fit not yet evaluated/frozen'});continue
        values={}
        for target,(_,_,count) in TARGETS.items():
            p=out/arm/(target+'.npz');mp=p.with_suffix('.json')
            if not p.exists() or not mp.exists():
                missing.append({'arm':arm,'target':target,'reason':'target evaluation incomplete'});continue
            meta=read_json(mp)
            if meta['freeze_sha256']!=sha(out/'freeze.json') or meta['scores_sha256']!=sha(p):raise RuntimeError('Score binding changed')
            with np.load(p,allow_pickle=False) as z,np.load(ROOT/'reference'/('c04_'+target+'.npz'),allow_pickle=False) as ref:
                for k in ('label','video','frame','video_names'):
                    if not np.array_equal(z[k],ref[k]):raise RuntimeError('Frame/label alignment changed: '+arm+'/'+target+'/'+k)
                if z['raw'].shape!=(count,):raise RuntimeError('Scored length changed')
                value=auroc(z['label'],normalize(z['raw'],z['video']))
            values[target]=value
            rows.append({'study':'ZXVAD_C04_SCREEN_DUALGPU_3090_20261008','arm':arm,'recipe':recipe,'seed':seed,'physical_gpu':cfg['physical_gpu'],'checkpoint_step':5000,'target':target,'frames':count,'AUROC':value})
        target_values[arm]=values
        if len(values)==3:macro[arm]=float(np.mean(list(values.values())))
    for row in rows:
        b=target_values.get(f"B_s{row['seed']}",{}).get(row['target'])
        row['delta_vs_paired_B_pp']=None if b is None else 100*(row['AUROC']-b)
    fieldnames=['study','arm','recipe','seed','physical_gpu','checkpoint_step','target','frames','AUROC','delta_vs_paired_B_pp']
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    (out/'results.csv').write_text(buf.getvalue(),encoding='utf-8')
    b=macro.get('B_s17');screen={}
    for arm in ORDER:
        recipe,seed=parse(arm);v=target_values.get(arm,{})
        deltas={t:100*(a-target_values['B_s17'][t]) for t,a in v.items() if t in target_values.get('B_s17',{})}
        screen[recipe]={'arm':arm,'AUROC':v,'target_delta_pp':deltas,'positive_targets':[t for t,d in deltas.items() if d>0],'macro_AUROC':macro.get(arm),'macro_delta_pp':100*(macro[arm]-b) if arm in macro and b is not None else None,'all3_targets_complete':len(v)==3}
    ranked=sorted([r for r in RECIPES if r!='B' and screen[r]['macro_delta_pp'] is not None],key=lambda r:(-screen[r]['macro_delta_pp'],RECIPES.index(r)))
    positive_macro=[r for r in ranked if screen[r]['macro_delta_pp']>0]
    positive_domain=[r for r in RECIPES if r!='B' and screen[r]['positive_targets']]
    contrasts={a+'-'+bb:100*(macro[a+'_s17']-macro[bb+'_s17']) if a+'_s17' in macro and bb+'_s17' in macro else None for a,bb in [('S25','SR25'),('M20','MR20'),('F9','U9'),('CM7','C7'),('R50','R100'),('QC03','QC01')]}
    complete=len(trained)==len(ORDER) and len(rows)==3*len(ORDER)
    summary={'status':'COMPLETED' if complete else 'INCOMPLETE','study':'ZXVAD_C04_SCREEN_DUALGPU_3090_20261008','author_confirmed':False,'metric':'AUROC only','kind':'SINGLE_SEED_DEVELOPMENT_SCREEN','seeds':[17],'gpu_queues':QUEUES,'cross_gpu_training_trajectory_identity_not_guaranteed':True,'planned_fits':len(ORDER),'completed_fits':len(trained),'trained_arms':trained,'candidate_order':ORDER,'rows':rows,'macro_AUROC':macro,'screening_table':screen,'ranked_macro_candidates':ranked,'positive_macro_candidates':positive_macro,'positive_any_domain_candidates':positive_domain,'control_macro_contrasts_pp':contrasts,'baseline_all3_complete':'B_s17' in macro,'missing':missing,'freeze_sha256':sha(out/'freeze.json') if freeze else None,'code_release':release_identity(),'no_target_training_or_checkpoint_selection':True,'prior_target_feedback_used_for_design':True,'interpretation':'Fixed single-seed broad screen; every negative, control and missing result retained. Rank only complete3-target candidates versus the same B. Individual-domain gain is distinct from macro gain. Positives need later confirmation.'}
    write_json(out/'summary.json',summary)
    lines=['# c04 single-seed screening AUROC','','Seed17 only; all predeclared settings; final5000 only; no bootstrap reporting.','','| Setting | Ped1 % | Ped2 % | Avenue % | Macro % | Macro delta pp | Positive targets |','|---|---:|---:|---:|---:|---:|---|']
    for recipe in RECIPES:
        item=screen[recipe];v=item['AUROC'];cells=[f'{100*v[t]:.4f}' if t in v else 'missing' for t in TARGETS]
        cells.append(f'{100*item["macro_AUROC"]:.4f}' if item['macro_AUROC'] is not None else 'missing')
        cells.append(f'{item["macro_delta_pp"]:+.4f}' if item['macro_delta_pp'] is not None else 'missing')
        cells.append(','.join(item['positive_targets']) or '-')
        lines.append('| '+recipe+' | '+' | '.join(cells)+' |')
    lines+=['','Positive macro candidates: '+(', '.join(positive_macro) or 'none'),'', 'Positive in at least one target: '+(', '.join(positive_domain) or 'none'),'', 'Macro ranking: '+', '.join(ranked),'','Control contrasts pp: '+str(contrasts),'','Missing items: '+str(len(missing))+'. See summary.json.','Single-seed positives are screening signals; development feedback is not a blind benchmark.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary['status'],'trained',len(trained),'AUROC rows',len(rows),'macro positives',positive_macro,flush=True)
    return summary

if __name__=='__main__':generate_report()
